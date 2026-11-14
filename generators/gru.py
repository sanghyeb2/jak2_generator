"""기초 GRU 분자 생성 모델과 학습·생성 함수."""
import copy
import random
import re
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
TOKEN = re.compile(r"\[[^\]]+\]|Br|Cl|%\d{2}|.")  # Cl, Br, [nH] 등은 한 토큰


# 1. 모델: 이전 토큰들을 읽고 다음 토큰의 확률을 예측합니다.
class SmilesGRU(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=PAD)
        self.gru = nn.GRU(128, 256, batch_first=True)
        self.output = nn.Linear(256, vocab_size)

    def forward(self, tokens, hidden=None):
        output, hidden = self.gru(self.embedding(tokens), hidden)
        return self.output(output), hidden


def pad_batch(batch):
    return pad_sequence(batch, batch_first=True, padding_value=PAD)


def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        # 예: [BOS, C, C, O]를 입력하면 [C, C, O, EOS]를 맞힙니다.
        with torch.set_grad_enabled(optimizer is not None):
            logits, _ = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), ignore_index=PAD, reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError("학습 loss가 유한하지 않습니다.")
            n_tokens = int((targets != PAD).sum())
            if optimizer is not None:
                optimizer.zero_grad()
                (loss / n_tokens).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item()
        total_tokens += n_tokens
    return total_loss / total_tokens


# 2. 학습: train.csv 내부의 10%를 검증에 사용합니다.
def train_model(train_csv, epochs=150, patience=20, batch_size=128, device=None, full_train=False):
    """train.csv의 smiles로 학습하고 (모델, 토큰 목록)을 반환합니다."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)
    tokens = [TOKEN.findall(s) for s in pd.read_csv(train_csv)["smiles"]]
    if len(tokens) < 10 or epochs < 1:
        raise ValueError("학습 분자 10개 이상, epochs 1 이상이 필요합니다.")
    random.shuffle(tokens)
    n_valid = 0 if full_train else max(1, round(len(tokens) * 0.1))
    fit, valid = tokens[n_valid:], tokens[:n_valid]
    vocab = SPECIAL + sorted({t for sequence in fit for t in sequence})
    token_id = {token: i for i, token in enumerate(vocab)}

    def encode(sequences):
        return [torch.tensor([BOS] + [token_id.get(t, UNK) for t in s] + [EOS])
                for s in sequences]

    fit_loader = DataLoader(encode(fit), batch_size, shuffle=True, collate_fn=pad_batch)
    valid_loader = DataLoader(encode(valid), batch_size, collate_fn=pad_batch)
    model = SmilesGRU(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    best_loss, stale, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, fit_loader, optimizer)
        if full_train:
            print(f"epoch {epoch}: train={train_loss:.3f}", flush=True)
            scheduler.step()
            continue
        valid_loss = run_epoch(model, valid_loader)
        print(f"epoch {epoch}: train={train_loss:.3f}, valid={valid_loss:.3f}", flush=True)
        if valid_loss < best_loss - 1e-4:
            best_loss, stale = valid_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break
    if not full_train:
        model.load_state_dict(best_state)
    return model, vocab


# 3. 생성: BOS부터 시작해서 모델이 EOS를 낼 때까지 토큰을 뽑습니다.
@torch.inference_mode()
def sample_batch(model, vocab, size, rng, temperature=1.0):
    device = next(model.parameters()).device
    current = torch.full((size, 1), BOS, dtype=torch.long, device=device)
    hidden, finished = None, [False] * size
    sequences = [[] for _ in range(size)]
    for _ in range(256):
        logits, hidden = model(current, hidden)
        logits = logits[:, -1] / temperature
        logits[:, [PAD, BOS, UNK]] = -torch.inf
        current = torch.multinomial(logits.softmax(dim=-1), 1, generator=rng)
        for i, token in enumerate(current[:, 0].tolist()):
            if not finished[i]:
                if token == EOS:
                    finished[i] = True
                else:
                    sequences[i].append(vocab[token])
        if all(finished):
            break
    return ["".join(s) if done else None for s, done in zip(sequences, finished)]




def main():
    import argparse
    from pathlib import Path
    from common import ROOT, standardize_smiles
    parser = argparse.ArgumentParser(description="GRU: fixed 6029-row training and 10k generation")
    parser.add_argument('--mode', choices=('train', 'generate', 'all'), default='generate')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'weights/gru/final.pt')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint
    if args.mode in ('train', 'all'):
        checkpoint = args.output_dir/'checkpoint.pt'
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        model, vocab = train_model(ROOT/'data/train.csv', epochs=150, batch_size=128, full_train=True)
        torch.save({'state_dict': model.state_dict(), 'vocab': vocab}, checkpoint)
    if args.mode == 'train':
        return
    output = args.output_dir/'gru_10000.csv'
    if output.exists():
        raise FileExistsError(output)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    vocab = saved['vocab']
    model = SmilesGRU(len(vocab)).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    rows, seen = [], set()
    for seed, quota in zip((123, 124, 125), (3334, 3333, 3333)):
        rng = torch.Generator(device=device).manual_seed(seed)
        accepted = 0
        for _ in range(400):
            for raw in sample_batch(model, vocab, 256, rng, temperature=1.0):
                key = standardize_smiles(raw)
                if key is None or key in seen:
                    continue
                seen.add(key)
                rows.append((raw, 'gru', seed))
                accepted += 1
                if accepted == quota:
                    break
            if accepted == quota:
                break
        if accepted != quota:
            raise RuntimeError(f'Seed {seed}: insufficient valid unique candidates')
    pd.DataFrame(rows, columns=['smiles', 'source', 'seed']).to_csv(output, index=False)


if __name__ == '__main__':
    main()
