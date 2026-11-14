"""Transfer503: public GP -> external503 -> JAK2 8+8 epochs -> 10k."""
import argparse
from pathlib import Path
from common import ROOT, gp_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'generate', 'all'), default='generate')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'weights/transfer/final.pt')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint
    if args.mode in ('train', 'all'):
        external, stage1, checkpoint = [args.output_dir/name for name in ('external.pt', 'stage1.pt', 'checkpoint.pt')]
        if any(path.exists() for path in (external, stage1, checkpoint)):
            raise FileExistsError('Use a new output directory for training')
        base = ['--full-train', '--skip-generation', '--batch-size', '32']
        gp_phase([*base, '--train-csv', str(ROOT/'data/jak_transfer_candidates_503.csv'),
                  '--epochs', '3', '--lr', '5e-6', '--checkpoint', str(external)])
        gp_phase([*base, '--train-csv', str(ROOT/'data/train.csv'), '--epochs', '8', '--lr', '1e-5',
                  '--init-checkpoint', str(external), '--checkpoint', str(stage1)])
        gp_phase([*base, '--train-csv', str(ROOT/'data/train.csv'), '--epochs', '8', '--lr', '1e-5',
                  '--init-checkpoint', str(stage1), '--checkpoint', str(checkpoint)])
    if args.mode == 'train':
        return
    output = args.output_dir/'transfer_10000.csv'
    if output.exists():
        raise FileExistsError(output)
    gp_phase(['--generate-only', '--checkpoint', str(checkpoint), '--out', str(output),
              '--target-count', '10000', '--max-draws', '100000', '--batch-size', '32',
              '--temperature', '0.9', '--seed', '123'])


if __name__ == '__main__':
    main()
