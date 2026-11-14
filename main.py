"""Standalone training -> uniform-seed generation -> fullprior -> neural pipeline.

Default execution starts full training. --dry-run only prints commands.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODELS = ('gp', 'gru', 'transfer', 'molgpt', 'reinvent')


def check_inputs(args, models):
    """Check local assets before starting an expensive worker; no model loading."""
    from pipeline_utils import digest
    records = json.loads((HERE / 'results/assets_manifest.json').read_text(encoding='utf-8'))['files']
    data = ('inputs/train.csv', 'generators/data/train.csv',
            'generators/data/jak_transfer_candidates_503.csv', 'generators/data/molgpt_settings.json')
    for name in data:
        path = HERE / name
        if not path.is_file():
            raise FileNotFoundError(f'{name} is missing. See README quick start and prepare_assets.py.')
        if digest(path) != records[name]['sha256']:
            raise ValueError(f'The fixed training input changed: {name}')
    required = [HERE / 'assets/ChemNet_v0.13_pretrained.pt', HERE / 'runtime/rdkit-2025.9.6/rdkit']
    if args.stage in ('train', 'generate', 'all'):
        for model in models:
            if model in ('gp', 'transfer'):
                required += [HERE / 'assets/GP-MoLFormer-Uniq/model.safetensors',
                             HERE / 'assets/MoLFormer-XL-both-10pct/tokenizer.json']
            elif model == 'molgpt':
                required += [HERE / 'generators/weights/molgpt/pretrained/model.safetensors',
                             HERE / 'runtime/molgpt-runtime/transformers']
            elif model == 'reinvent':
                required += [HERE / 'generators/weights/reinvent/pretrained/reinvent_pubchem.prior',
                             HERE / 'generators/weights/reinvent/pretrained/source/rnn.py',
                             HERE / 'generators/weights/reinvent/pretrained/source/vocabulary.py']
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f'{path.relative_to(HERE)} is missing. Run prepare_assets.py --download first.')


def worker_env(model=None):
    env = os.environ.copy()
    paths = [HERE / 'generators', HERE]
    if model == 'molgpt':
        paths.insert(0, HERE / 'runtime/molgpt-runtime')
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONHASHSEED='123', PYTHONIOENCODING='utf-8',
        PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        HF_HOME=str(HERE / 'local_artifacts/hf-cache'),
        HF_MODULES_CACHE=str(HERE / 'local_artifacts/hf-cache/modules'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    )
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('train', 'generate', 'select', 'all'), default='all')
    parser.add_argument('--model', choices=MODELS, help='train/generate only; otherwise all five')
    parser.add_argument('--run-name', default='standalone_seed42')
    parser.add_argument('--use-trained', action='store_true', help='generate from this run training outputs')
    parser.add_argument('--use-generated', action='store_true', help='select from this run generated CSVs')
    parser.add_argument('--dry-run', action='store_true', help='print commands without running them')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('run-name must contain only ASCII letters, digits, _ or -')
    if args.model and args.stage not in ('train', 'generate'):
        parser.error('--model applies only to train/generate')
    work = HERE / 'runs' / args.run_name
    generation = work / 'generation'
    selection = work / 'selection'
    models = (args.model,) if args.model else MODELS
    tasks = []
    for action in ('train', 'generate'):
        if args.stage not in (action, 'all'):
            continue
        for model in models:
            command = [sys.executable, str(HERE / 'generators/generate.py'),
                       '--action', action, '--model', model, '--output-dir', str(generation)]
            if action == 'generate' and (args.use_trained or args.stage == 'all'):
                command.append('--use-trained')
            tasks.append((f'{action}_{model}', command, model))
    if args.stage in ('select', 'all'):
        candidates = generation / 'candidates' if args.use_generated or args.stage == 'all' else HERE / 'inputs/candidates'
        tasks.append(('selection', [sys.executable, str(HERE / 'selection_pipeline.py'), '--stage', 'all',
                     '--candidate-dir', str(candidates), '--train', str(HERE / 'inputs/train.csv'),
                     '--output-dir', str(selection), '--rdkit-path', str(HERE / 'runtime/rdkit-2025.9.6')], None))
    if args.dry_run:
        print(json.dumps({'tasks': [{'stage': name, 'command': cmd} for name, cmd, _ in tasks],
                          'no_training_started': True}, ensure_ascii=False, indent=2))
        return
    from pipeline_utils import save_json, digest
    check_inputs(args, models)
    if digest(HERE / 'inputs/train.csv') != digest(HERE / 'generators/data/train.csv'):
        raise ValueError('Generator and selector training snapshots differ')
    generation.mkdir(parents=True, exist_ok=True)
    for name, command, model in tasks:
        print('START', name, flush=True)
        env = worker_env(model)
        if model is None:
            env['PYTHONPATH'] = str(HERE)
        subprocess.run(command, cwd=HERE, env=env, check=True)
    if args.stage in ('select', 'all'):
        result = selection / 'neural_submission.csv'
        save_json(work / 'pipeline_complete.json', {
            'submission': str(result.relative_to(HERE)), 'sha256': digest(result),
            'end_to_end_repeat_verification': 'not performed',
        })
        print('FINAL CSV:', result, flush=True)


if __name__ == '__main__':
    main()
