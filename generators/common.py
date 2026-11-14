"""Generator-only utilities. No historical selector state or parent-project imports."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from functools import lru_cache
import argparse,contextlib,os,random,hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer
from rdkit import Chem,rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize as MS
ALLOWED=set("H B C N O F Si P S Cl Br I".split())
UNCHARGER=MS.Uncharger()
TAUTOMER=MS.TautomerEnumerator()
TAUTOMER.SetMaxTautomers(1000)
TAUTOMER.SetMaxTransforms(2000)
MODEL_ID=str(ROOT.parent/'assets/GP-MoLFormer-Uniq')
MODEL_REVISION='6eca879581e2302b4e1ab07bb02908636bddb4a2'
TOKENIZER_ID=str(ROOT.parent/'assets/MoLFormer-XL-both-10pct')
TOKENIZER_REVISION='361063d0ad524ef77cf39b08469f6be770dc550f'

@lru_cache(maxsize=50000)
def _standardize(smiles):
    """문자열 하나 → (제출용 SMILES, 중복 판정용 SMILES). 채점 기준과 동일."""
    if len(smiles) > 1000:
        return None
    smiles = smiles.strip()
    if not smiles or any(c.isspace() for c in smiles) or "|" in smiles:
        return None
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not 0 < mol.GetNumHeavyAtoms() <= 128:
                return None
            # 염·전하·동위원소·atom map을 정리합니다.
            mol = UNCHARGER.uncharge(MS.FragmentParent(MS.Cleanup(mol)))
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
                atom.SetAtomMapNum(0)
            mol = Chem.RemoveHs(mol)
            # 동일한 tautomer 계열을 하나의 대표 구조로 정리합니다.
            forms = TAUTOMER.Enumerate(mol)
            if forms.status != MS.TautomerEnumeratorStatus.Completed:
                return None
            mol = TAUTOMER.PickCanonical(forms)
            Chem.SanitizeMol(mol)
            if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
                return None
            if any(a.GetSymbol() not in ALLOWED or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms()):
                return None
            canonical = Chem.MolToSmiles(mol)
            if len(canonical) > 349:
                return None
            # 제출용 입체정보는 보존하되, 중복 판정에서는 입체 차이를 무시합니다.
            Chem.RemoveStereochemistry(mol)
            identity = Chem.MolToSmiles(mol, isomericSmiles=False)
            return canonical, identity
    except (ValueError, RuntimeError, OverflowError):
        return None


def standardize_smiles(smiles):
    """None이면 제외, 반환 문자열이 같으면 중복입니다."""
    result = _standardize(smiles) if isinstance(smiles, str) else None
    return None if result is None else result[1]


def make_submission(generated_smiles):
    """유효·고유 분자 2,000개를 입력 순서대로 선택합니다. 부족하면 오류."""
    if isinstance(generated_smiles, (str, bytes)):
        raise TypeError("SMILES 하나가 아니라 SMILES 목록을 넣어 주세요.")
    unique = {}  # 중복 판정용 SMILES → 제출용 SMILES
    for raw in generated_smiles:
        result = _standardize(raw) if isinstance(raw, str) else None
        if result is not None:
            canonical, identity = result
            unique.setdefault(identity, canonical)
        if len(unique) == 2000:
            return pd.DataFrame({"id": range(2000), "smiles": list(unique.values())})
    raise ValueError(f"유효·고유 분자가 {len(unique)}개입니다. 2,000개까지 추가 생성하세요.")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collate(sequences, pad_id):
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence)
        mask[row, : len(sequence)] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    return ids, mask, labels


def epoch_loss(model, loader, device, optimizer, mixed_precision, quiet):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    token_count = 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext():
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        count = int(mask[:, 1:].sum())
        loss_sum += loss.item() * count
        token_count += count
    return loss_sum / token_count


@torch.inference_mode()
def generate_submission(model, tokenizer, device, out, max_draws, batch_size, temperature, quiet,
                        target_count=2000, seed=123):
    model.eval()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    selected, seen = [], set()
    forbidden = [tokenizer.bos_token_id, tokenizer.pad_token_id,
                 tokenizer.mask_token_id, tokenizer.unk_token_id]
    for start in range(0, max_draws, batch_size):
        size = min(batch_size, max_draws - start)
        prompt = torch.full((size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            output = model.generate(
                prompt, do_sample=True, top_k=0, temperature=temperature,
                max_length=202, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[token] for token in forbidden],
            )
        for raw in tokenizer.batch_decode(output, skip_special_tokens=True):
            key = standardize_smiles(raw)
            if key is None or key in seen:
                continue
            seen.add(key)
            selected.append(raw)
            if len(selected) == target_count:
                if target_count == 2000:
                    make_submission(selected).to_csv(out, index=False)
                else:
                    pd.DataFrame({"smiles": selected}).to_csv(out, index=False)
                print(f"Saved {out}: {target_count:,} valid unique molecules after {start + size:,} draws", flush=True)
                return
        print(f"Generated {start + size:,}: valid unique {len(selected):,}", flush=True)
    raise RuntimeError(f"Only {len(selected)} valid unique molecules after {max_draws} draws")


def gp_phase(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(ROOT / "data/train.csv"))
    parser.add_argument("--validation-csv", help="Use a separate fixed validation set")
    parser.add_argument("--checkpoint", default="gp_molformer_best.pt")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Continue training from an existing state_dict checkpoint")
    parser.add_argument("--out", default="results/gp_molformer/submission_gp_molformer.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--save-epochs", type=int, nargs="+", default=[])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-draws", type=int, default=50_000)
    parser.add_argument("--target-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-unencodable", action="store_true",
                        help="Exclude rows unsupported by the tokenizer during synthetic fine-tuning")
    parser.add_argument("--full-train", action="store_true",
                        help="Use all training rows and save the fixed final epoch; no internal holdout")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.lr <= 0 or args.temperature <= 0 or args.target_count < 1:
        parser.error("epochs, patience, batch-size, target-count, lr, and temperature must be positive")
    if args.validation_csv and (args.full_train or args.generate_only):
        parser.error("--validation-csv requires training with validation")

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()
    cache_dir = ROOT.parent / "local_artifacts/hf-cache/hub"
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir,
        trust_remote_code=True).to(device)
    print(f"Device: {device}; bf16: {mixed_precision}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    if args.generate_only:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    else:
        smiles = pd.read_csv(args.train_csv)["smiles"].tolist()
        sequences = tokenizer(smiles, add_special_tokens=True, return_token_type_ids=False)["input_ids"]
        supported = [s for s in sequences if tokenizer.unk_token_id not in s and len(s) <= 202]
        if len(supported) != len(sequences):
            if not args.skip_unencodable or not supported:
                raise ValueError("Training data contains unknown tokens or sequences longer than 202 tokens")
            print(f"Excluded {len(sequences) - len(supported):,} unsupported training rows", flush=True)
            sequences = supported
        random.shuffle(sequences)
        if args.validation_csv:
            validation_smiles = pd.read_csv(args.validation_csv)["smiles"].tolist()
            if not validation_smiles or set(smiles).intersection(validation_smiles):
                raise ValueError("Validation must be nonempty and disjoint from training")
            valid = tokenizer(validation_smiles, add_special_tokens=True,
                              return_token_type_ids=False)["input_ids"]
            if any(tokenizer.unk_token_id in s or len(s) > 202 for s in valid):
                raise ValueError("Validation data contains unknown tokens or sequences longer than 202 tokens")
            train = sequences
        else:
            n_valid = 0 if args.full_train else max(1, round(len(sequences) * 0.1))
            valid, train = sequences[:n_valid], sequences[n_valid:]
        batch = lambda rows: collate(rows, tokenizer.pad_token_id)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=batch)
        valid_loader = None if args.full_train else DataLoader(valid, batch_size=args.batch_size, collate_fn=batch)
        if args.init_checkpoint:
            model.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu", weights_only=True))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best_loss, best_epoch, stale = float("inf"), 0, 0
        with open(os.devnull, "w") as quiet:
            if args.init_checkpoint and not args.full_train:
                best_loss = epoch_loss(model, valid_loader, device, None, mixed_precision, quiet)
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.checkpoint)
                print(f"Starting validation loss: {best_loss:.4f}", flush=True)
            for epoch in range(1, args.epochs + 1):
                train_loss = epoch_loss(model, train_loader, device, optimizer, mixed_precision, quiet)
                if epoch in args.save_epochs and epoch != args.epochs:
                    path = Path(args.checkpoint)
                    epoch_path = path.with_name(f"{path.stem}_epoch{epoch}{path.suffix}")
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, epoch_path)
                    print(f"Saved epoch {epoch}: {epoch_path}", flush=True)
                if args.full_train:
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.checkpoint)
                    print(f"epoch {epoch}: train={train_loss:.4f}; saved final-epoch checkpoint", flush=True)
                else:
                    valid_loss = epoch_loss(model, valid_loader, device, None, mixed_precision, quiet)
                    print(f"epoch {epoch}: train={train_loss:.4f}, valid={valid_loss:.4f}", flush=True)
                    if valid_loss < best_loss - 1e-4:
                        best_loss, best_epoch, stale = valid_loss, epoch, 0
                        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.checkpoint)
                    else:
                        stale += 1
                if not args.full_train and stale >= args.patience:
                    break
        if args.full_train:
            print(f"Trained on all {len(train):,} molecules; checkpoint: {args.checkpoint}", flush=True)
        else:
            print(f"Best validation loss: {best_loss:.4f} at epoch {best_epoch}; checkpoint: {args.checkpoint}", flush=True)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    if not args.skip_generation:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(os.devnull, "w") as quiet:
            generate_submission(model, tokenizer, device, args.out, args.max_draws,
                                args.batch_size, args.temperature, quiet,
                                args.target_count, args.seed)
