"""Local generator checkpoint/RNG/manifest helpers; no parent-project imports."""
from pathlib import Path
import os,sys,json,random
import numpy as np
import torch
PACKAGE=Path(__file__).resolve().parent.parent
sys.path.insert(1,str(PACKAGE))
from pipeline_utils import digest,save_json,save_receipt,receipt_valid,configure,environment as _environment

def environment(path):
    save_json(path,_environment())

def atomic_torch_save(value,path):
    tmp=path.with_suffix(path.suffix+'.pending');torch.save(value,tmp);tmp.replace(path)

def atomic_csv(frame,path):
    tmp=path.with_suffix(path.suffix+'.pending');frame.to_csv(tmp,index=False);tmp.replace(path)

def restore_rng(saved):
    torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    if torch.cuda.is_available() and saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])

def generation_identity(work,use_trained):
    from importlib.metadata import version
    folder=PACKAGE/'generators'
    files=[PACKAGE/'main.py',PACKAGE/'pipeline_utils.py',*sorted(folder.glob('*.py')),*sorted((folder/'data').glob('*'))]
    for name in ('gp','gru','transfer','molgpt','reinvent'):
        if use_trained:
            checkpoint=work/'training'/name/('last.pt' if name=='reinvent' else 'checkpoint.pt')
        else:checkpoint=folder/'weights'/name/('checkpoint.pt' if name=='molgpt' else 'final.pt')
        files.append(checkpoint)
    for base in (PACKAGE/'assets',folder/'weights/molgpt/pretrained',folder/'weights/reinvent/pretrained'):
        files.extend(p for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts)
    for p in files:
        if not p.is_file():raise FileNotFoundError(p)
    return {'files':{str(p.relative_to(PACKAGE)):digest(p) for p in files},
        'seeds':[123,124,125],'quotas':[3334,3333,3333],'max_draws_per_seed':100000,
        'use_trained':use_trained,'packages':{n:version(n) for n in ('torch','transformers','tokenizers','numpy','pandas','rdkit')}}


def training_identity(model):
    from importlib.metadata import version
    folder=PACKAGE/'generators'
    files=[PACKAGE/'main.py',PACKAGE/'pipeline_utils.py',*sorted(folder.glob('*.py')),folder/'data/train.csv']
    if model=='transfer':files.append(folder/'data/jak_transfer_candidates_503.csv')
    if model=='molgpt':files.append(folder/'data/molgpt_settings.json')
    bases=[]
    if model in ('gp','transfer'):bases=[PACKAGE/'assets/GP-MoLFormer-Uniq',PACKAGE/'assets/MoLFormer-XL-both-10pct']
    elif model in ('molgpt','reinvent'):bases=[folder/f'weights/{model}/pretrained']
    for base in bases:
        files.extend(p for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts)
    return {'model':model,'files':{str(p.relative_to(PACKAGE)):digest(p) for p in files},
            'seed':42,'packages':{n:version(n) for n in ('torch','transformers','tokenizers','numpy','pandas','rdkit')}}
