"""Small reproducibility helpers, independent of historical experiment state."""
import hashlib
import json
import os
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)

def save_receipt(work,stage,paths):
    save_json(work/'receipts'/f'{stage}.json',{'outputs':{str(Path(p).relative_to(work)):digest(p) for p in paths}})

def receipt_valid(work,stage):
    path=work/'receipts'/f'{stage}.json'
    if not path.exists():return False
    for name,sha in json.loads(path.read_text(encoding='utf-8'))['outputs'].items():
        if not (work/name).is_file() or digest(work/name)!=sha:raise ValueError(f'Completed output changed: {stage}/{name}')
    return True

def configure(seed=42):
    import random,numpy as np,torch
    configure_determinism();torch.set_num_threads(4)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the defined training profile')

def atomic_numpy(path,writer):
    tmp=path.with_suffix(path.suffix+'.pending')
    with tmp.open('wb') as f:writer(f)
    tmp.replace(path)

def environment():
    import sys,torch
    from rdkit import rdBase
    from importlib.metadata import version
    return {'python':sys.version,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
        'rdkit_runtime':rdBase.rdkitVersion,'packages':{n:version(n) for n in ('torch','numpy','pandas','scipy','fcd-torch')},
        'deterministic':torch.are_deterministic_algorithms_enabled(),'tf32':torch.backends.cuda.matmul.allow_tf32,
        'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'pythonhashseed':os.environ.get('PYTHONHASHSEED')}


def configure_determinism():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
