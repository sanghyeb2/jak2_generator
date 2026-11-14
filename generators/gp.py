"""GP-MoLFormer: full-data 8+10 epochs and three-seed 10k pool."""
import argparse
from pathlib import Path
import pandas as pd
from common import ROOT, gp_phase, standardize_smiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'generate', 'all'), default='generate')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'weights/gp/final.pt')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint
    if args.mode in ('train', 'all'):
        stage1, checkpoint = args.output_dir/'stage1.pt', args.output_dir/'checkpoint.pt'
        if stage1.exists() or checkpoint.exists():
            raise FileExistsError('Use a new output directory for training')
        base = ['--full-train', '--skip-generation', '--train-csv', str(ROOT/'data/train.csv'), '--batch-size', '32']
        gp_phase([*base, '--epochs', '8', '--lr', '1e-5', '--checkpoint', str(stage1)])
        gp_phase([*base, '--epochs', '10', '--lr', '5e-6', '--init-checkpoint', str(stage1), '--checkpoint', str(checkpoint)])
    if args.mode == 'train':
        return
    output = args.output_dir/'gp_10000.csv'
    if output.exists():
        raise FileExistsError(output)
    rows, seen = [], set()
    for seed, count, quota in zip((123, 124, 125), (10000, 4500, 4500), (3334, 3333, 3333)):
        path = args.output_dir/f'gp_seed_{seed}.csv'
        if path.exists():
            raise FileExistsError(path)
        gp_phase(['--generate-only', '--checkpoint', str(checkpoint), '--out', str(path),
                  '--target-count', str(count), '--max-draws', '100000', '--batch-size', '32',
                  '--temperature', '0.9', '--seed', str(seed)])
        accepted = 0
        for raw in pd.read_csv(path).smiles:
            key = standardize_smiles(raw)
            if key is None or key in seen:
                continue
            seen.add(key)
            rows.append((raw, 'gp', seed))
            accepted += 1
            if accepted == quota:
                break
        if accepted != quota:
            raise RuntimeError(f'Seed {seed}: insufficient candidates')
    pd.DataFrame(rows, columns=['smiles', 'source', 'seed']).to_csv(output, index=False)


if __name__ == '__main__':
    main()
