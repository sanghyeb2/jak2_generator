"""Prepare private training inputs and pinned public priors; never train models.

Private competition data are supplied by the participant, not downloaded from
GitHub. Public model files are fetched only with the explicit --download flag.
"""
import argparse
from importlib.metadata import distribution
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen
import zipfile

from pipeline_utils import digest

HERE = Path(__file__).resolve().parent
PRIVATE_FILES = ('inputs/train.csv', 'generators/data/train.csv',
                 'generators/data/jak_transfer_candidates_503.csv')


def expected():
    return json.loads((HERE / 'results/assets_manifest.json').read_text(encoding='utf-8'))['files']


def check(path, records):
    name = path.relative_to(HERE).as_posix()
    if not path.is_file():
        return False
    if digest(path) != records[name]['sha256']:
        raise ValueError(f'File differs from the preserved input: {name}')
    return True


def import_private(archive, records):
    with zipfile.ZipFile(archive) as source:
        for name in PRIVATE_FILES:
            if name not in source.namelist():
                raise ValueError(f'Private archive is missing {name}')
            path = HERE / name
            if check(path, records):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix(path.suffix + '.pending')
            with source.open(name) as incoming, pending.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if digest(pending) != records[name]['sha256']:
                raise ValueError(f'Private archive hash mismatch: {name}')
            pending.replace(path)


def fetch_file(url, target, records):
    if check(target, records):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + '.pending')
    with urlopen(url, timeout=120) as source, pending.open('wb') as output:
        shutil.copyfileobj(source, output)
    name = target.relative_to(HERE).as_posix()
    if digest(pending) != records[name]['sha256']:
        raise ValueError(f'Download hash mismatch: {name}')
    pending.replace(target)


def public_assets(records):
    from huggingface_hub import snapshot_download
    repositories = (
        ('ibm-research/GP-MoLFormer-Uniq', '6eca879581e2302b4e1ab07bb02908636bddb4a2', 'assets/GP-MoLFormer-Uniq'),
        ('ibm-research/MoLFormer-XL-both-10pct', '361063d0ad524ef77cf39b08469f6be770dc550f', 'assets/MoLFormer-XL-both-10pct'),
        ('jonghyunlee1993/MolGPT_long_context_pretrained-by-ZINC15', 'a52c3df13d9d67cf68f1977aca5dd7777cf88935', 'generators/weights/molgpt/pretrained'),
    )
    for repo, revision, folder in repositories:
        names = [name for name in records if name.startswith(folder + '/')]
        present = [check(HERE / name, records) for name in names]
        if not all(present):
            snapshot_download(repo_id=repo, revision=revision, local_dir=str(HERE / folder),
                              allow_patterns=[name[len(folder) + 1:] for name in names])
        for name in names:
            if not check(HERE / name, records):
                raise FileNotFoundError(name)
    prefix = 'generators/weights/reinvent/pretrained/'
    fetch_file('https://zenodo.org/api/records/20701824/files/reinvent_pubchem.prior/content',
               HERE / prefix / 'reinvent_pubchem.prior', records)
    commit = '0f075b349ed22015e0406350c6f426701bb0815d'
    base = f'https://raw.githubusercontent.com/MolecularAI/REINVENT4/{commit}/'
    for name in ('rnn.py', 'vocabulary.py', 'LICENSE'):
        relative = 'LICENSE' if name == 'LICENSE' else 'reinvent/models/reinvent/models/' + name
        fetch_file(base + relative, HERE / prefix / 'source' / name, records)
    chemnet = HERE / 'assets/ChemNet_v0.13_pretrained.pt'
    if not check(chemnet, records):
        source = Path(distribution('fcd-torch').locate_file('fcd_torch/ChemNet_v0.13_pretrained.pt'))
        chemnet.parent.mkdir(parents=True, exist_ok=True)
        if digest(source) != records[chemnet.relative_to(HERE).as_posix()]['sha256']:
            raise ValueError('Installed ChemNet differs from the recorded model')
        shutil.copy2(source, chemnet)
    for folder, marker, packages in (
        ('runtime/molgpt-runtime', 'transformers/__init__.py', ('transformers==4.46.3', 'tokenizers==0.20.3')),
        ('runtime/rdkit-2025.9.6', 'rdkit/__init__.py', ('rdkit==2025.9.6',)),
    ):
        if not (HERE / folder / marker).is_file():
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target',
                            str(HERE / folder), *packages], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-archive', type=Path, help='Private data ZIP supplied separately by the participant')
    parser.add_argument('--download', action='store_true', help='Download pinned public priors and install runtime overlays')
    args = parser.parse_args()
    records = expected()
    if args.data_archive:
        import_private(args.data_archive, records)
    if args.download:
        public_assets(records)
    missing = [name for name in PRIVATE_FILES if not check(HERE / name, records)]
    if missing:
        raise FileNotFoundError('Private data are required: ' + ', '.join(missing))
    print('Private input files match the recorded data. No training or generation was run.')


if __name__ == '__main__':
    main()
