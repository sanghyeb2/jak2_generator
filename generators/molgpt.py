"""MolGPT checkpoint IO and fixed-seed generation, without HPO dependencies."""
import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from common import ROOT, digest, standardize_smiles, collate, epoch_loss
WORK = ROOT / "weights/molgpt"
PRETRAINED = WORK / "pretrained"


def write_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    pending.replace(path)



def save_checkpoint(path, value):
    pending = path.with_suffix(".pending.pt")
    torch.save(value, pending)
    pending.replace(path)



def load_model(tokenizer, device):
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED, local_files_only=True)
    for config in (model.config, model.generation_config):
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
    return model.to(device)



@torch.inference_mode()
def generate(checkpoint, temperature, tokenizer, device, path,
             target_count=1000, max_draws=30000, resume=False):
    if path.exists():
        smiles = pd.read_csv(path).smiles.tolist()
        if len(smiles) != target_count:
            raise ValueError(f"Incomplete generated pool: {path}")
        return smiles
    model = load_model(tokenizer, device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(123)
    selected, seen, completed_draws = [], set(), 0
    progress = path.with_suffix(".progress.pt")
    identity = dict(checkpoint_sha256=digest(checkpoint), temperature=temperature,
                    target_count=target_count, max_draws=max_draws, seed=123,
                    device=str(device), torch_version=torch.__version__)
    if resume and progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Generation resume settings differ from the saved run")
        selected, seen, completed_draws = saved["selected"], set(saved["seen"]), saved["draws"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for start in range(completed_draws, max_draws, 64):
        size = min(64, max_draws - start)
        prompt = torch.full((size, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
        output = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                do_sample=True, top_k=0, temperature=temperature,
                                max_length=256, use_cache=True,
                                bad_words_ids=[[tokenizer.pad_token_id],
                                               [tokenizer.bos_token_id], [3]])
        for sequence in output.tolist():
            if tokenizer.eos_token_id not in sequence[1:]:
                continue
            raw = tokenizer.decode(sequence, skip_special_tokens=True).strip()
            key = standardize_smiles(raw)
            if key is None or key in seen:
                continue
            seen.add(key)
            selected.append(raw)
            if len(selected) == target_count:
                pending = path.with_suffix(".pending.csv")
                pd.DataFrame({"smiles": selected}).to_csv(pending, index=False)
                pending.replace(path)
                write_json(path.with_suffix(".generation.json"),
                           dict(**identity, draws=start + size, valid_unique=target_count))
                progress.unlink(missing_ok=True)
                print(f"Saved {path}: {target_count:,} unique after {start + size} draws", flush=True)
                return selected
        if resume:
            save_checkpoint(progress, dict(identity=identity, selected=selected, seen=list(seen),
                                          draws=start + size, torch_rng=torch.get_rng_state(),
                                          cuda_rng=torch.cuda.get_rng_state_all()
                                          if device.type == "cuda" else []))
        if start % 640 == 0:
            print(f"{path.name}: draws={start + size}, unique={len(selected)}", flush=True)
    raise RuntimeError(f"{path}: only {len(selected)} valid unique molecules")



"""Refit the selected MolGPT settings on all JAK2 rows and generate 10k candidates."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import random

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf-cache"))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


FULL = ROOT / "runs/molgpt"
POOL = FULL / "molgpt_6029_pool_10k.csv"


def train_full(tokenizer, settings, device):
    selected = settings["selected"]
    lr = float(selected["learning_rate"])
    epochs = selected["epoch"]
    identity = dict(train_sha256=digest(ROOT / "data/train.csv"), learning_rate=lr,
                    epochs=epochs, batch_size=64, seed=42,
                    source=settings["source"], device=str(device),
                    torch_version=torch.__version__)
    random.seed(42)
    torch.manual_seed(42)
    model = load_model(tokenizer, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    smiles = pd.read_csv(ROOT / "data/train.csv").smiles.tolist()
    if len(smiles) != 6029:
        raise ValueError("Expected all 6,029 training rows")
    ids = tokenizer(smiles, add_special_tokens=False)["input_ids"]
    if any(3 in sequence or len(sequence) + 2 > 256 for sequence in ids):
        raise ValueError("Unencodable full-training rows")
    sequences = [[tokenizer.bos_token_id] + sequence + [tokenizer.eos_token_id] for sequence in ids]
    loader = DataLoader(sequences, batch_size=64, shuffle=True,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    checkpoint = FULL / "checkpoint.pt"
    history, start_epoch = [], 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Full-training resume settings differ")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, history = saved["epoch"], saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with open(os.devnull, "w") as quiet:
        for epoch in range(start_epoch + 1, epochs + 1):
            loss = epoch_loss(model, loader, device, optimizer, mixed_precision, quiet)
            history.append(dict(epoch=epoch, train_loss=loss))
            save_checkpoint(checkpoint, dict(identity=identity, model=model.state_dict(),
                                            optimizer=optimizer.state_dict(), epoch=epoch,
                                            history=history, torch_rng=torch.get_rng_state(),
                                            python_rng=random.getstate(),
                                            cuda_rng=torch.cuda.get_rng_state_all()
                                            if device.type == "cuda" else []))
            write_json(FULL / "history.json", history)
            print(f"Full 6029 epoch={epoch}/{epochs} train={loss:.5f}", flush=True)
    write_json(FULL / "training_complete.json", identity)
    return checkpoint



def main():
    import argparse
    global FULL
    parser = argparse.ArgumentParser(description="MolGPT: fixed epoch18 full training and 10k generation")
    parser.add_argument('--mode', choices=('train', 'generate', 'all'), default='generate')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'weights/molgpt/checkpoint.pt')
    args = parser.parse_args()
    FULL = args.output_dir
    FULL.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT/'data/molgpt_settings.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = args.checkpoint
    if args.mode in ('train', 'all'):
        checkpoint = train_full(tokenizer, settings, device)
    if args.mode != 'train':
        generate(checkpoint, settings['selected']['temperature'], tokenizer, device,
                 FULL/'molgpt_10000.csv', target_count=10000, max_draws=100000, resume=True)


if __name__ == '__main__':
    main()
