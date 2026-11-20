"""Check fixed inputs, package imports and CUDA without training or generating."""
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def main():
    import main as pipeline
    import torch
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Python 3.11 required')
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA GPU required')
    mismatches = []
    for line in (HERE / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith(('#', '--')):
            package, expected = line.split('==')
            if version(package) != expected:
                mismatches.append(f'{package}: expected {expected}, found {version(package)}')
    if mismatches:
        raise RuntimeError('Environment differs: ' + '; '.join(mismatches))
    pipeline.check_inputs(SimpleNamespace(stage='all'), pipeline.MODELS)
    for model in pipeline.MODELS:
        command = ('import importlib; importlib.import_module(' + repr(model) + '); '
                   'import torch; assert torch.cuda.is_available(); '
                   'print(' + repr(model + ': import PASS') + ')')
        subprocess.run([sys.executable, '-X', 'utf8', '-c', command],
                       cwd=HERE / 'generators', env=pipeline.worker_env(model), check=True)
    environment = pipeline.worker_env()
    environment['PYTHONPATH'] = str(HERE)
    subprocess.run([sys.executable, '-c', 'import common,selector,neural_stage; print("selectors: import PASS")'],
                   cwd=HERE, env=environment, check=True)
    environment['PYTHONPATH'] = str(HERE / 'runtime/rdkit-2025.9.6')
    subprocess.run([sys.executable, '-c', 'from rdkit import rdBase; assert rdBase.rdkitVersion=="2025.09.6"; print("official RDKit: PASS")'],
                   cwd=HERE, env=environment, check=True)
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest', 'test_entrypoint', '-v'], cwd=HERE, check=True)
    print(json.dumps({'status': 'PASS', 'gpu': torch.cuda.get_device_name(0),
                      'training_executed': False, 'full_reproduction': 'not verified'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
