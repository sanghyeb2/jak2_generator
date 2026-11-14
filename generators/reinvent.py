"""Pinned REINVENT loading, fixed-parameter fit and sampling; no historical HPO dependency."""
import os,hashlib,importlib.util,json,random,sys,time,uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent/'runs/generation/reinvent'
ASSETS=ROOT/'weights/reinvent/pretrained'
PRIOR=ASSETS/'reinvent_pubchem.prior'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pending.replace(path)


def save_checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    torch.save(value,pending);pending.replace(path)


def configure(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def vendor_module(name,filename):
    if name in sys.modules:return sys.modules[name]
    path=ASSETS/'source'/filename
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_prior(device):
    vocab_module=vendor_module('jak2_reinvent_vocab','vocabulary.py')
    rnn_module=vendor_module('jak2_reinvent_network','rnn.py')
    with torch.serialization.safe_globals([uuid.UUID,(vocab_module.SMILESTokenizer,
          'reinvent.models.reinvent.models.vocabulary.SMILESTokenizer')]):
        state=torch.load(PRIOR,map_location='cpu',weights_only=True)
    vocab=vocab_module.Vocabulary.load_from_dictionary(state['vocabulary'])
    tokenizer=vocab_module.SMILESTokenizer()
    # Token strings are authoritative: old prior metadata incorrectly says EOS2.
    if vocab['$']!=0 or vocab['^']!=1:raise ValueError('Unexpected stop/start tokens')
    model=rnn_module.RNN(len(vocab),**state['network_params'],device=torch.device(device))
    model.load_state_dict(state['network'])
    info={'network_params':state['network_params'],'max_sequence_length':state['max_sequence_length'],
          'parameter_count':sum(p.numel() for p in model.parameters()),'vocab_size':len(vocab),
          'bos_token_id':vocab['^'],'eos_and_padding_token_id':vocab['$'],
          'metadata':json.loads(json.dumps(state.get('metadata'),default=str))}
    return model,vocab,tokenizer,info


def collate(rows):
    return torch.nn.utils.rnn.pad_sequence([torch.tensor(x,dtype=torch.long) for x in rows],batch_first=True,padding_value=0)


def sequence_nll(logits,targets):
    """Official REINVENT likelihood: sum all padded-position NLLs per sequence."""
    return -logits.log_softmax(-1).gather(-1,targets[...,None]).squeeze(-1).sum(-1)


def epoch(model,loader,device,optimizer=None,limit=None):
    model.train(optimizer is not None)
    total,count=0.,0
    for step,tokens in enumerate(loader):
        if limit is not None and step>=limit:break
        tokens=tokens.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits,_=model(tokens[:,:-1])
            losses=sequence_nll(logits,tokens[:,1:]);loss=losses.mean()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                optimizer.step()
        total+=losses.detach().sum().item();count+=len(tokens)
    return total/count


def rng_state(loader_rng):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader=loader_rng.get_state())


def restore_rng(state,loader_rng):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    loader_rng.set_state(state['loader'].cpu())


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fit(lr,full=False,epochs=30,limit_batches=None,output=None):
    configure(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=torch.load(WORK/'encoded.pt',weights_only=True)
    model,vocab,tokenizer,info=load_prior(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loader_rng=torch.Generator().manual_seed(42)
    fit_data=data['full' if full else 'fit']
    train_loader=DataLoader(fit_data,batch_size=64,shuffle=True,generator=loader_rng,collate_fn=collate)
    val_loader=DataLoader(data['val'],batch_size=64,collate_fn=collate) if not full else None
    directory=Path(output) if output else WORK/('full_6029' if full else f'lr_{lr:g}')
    directory.mkdir(parents=True,exist_ok=True)
    identity={'lr':lr,'full':full,'epochs':epochs,'batch_size':64,'seed':42,'optimizer':'Adam',
        'clip':1.,'limit_batches':limit_batches,'encoded_sha256':digest(WORK/'encoded.pt'),
        'prior_sha256':digest(PRIOR),'source_sha256':digest(Path(__file__)),
        'loss':'official mean sequence-summed NLL','precision':'float32','device':str(device)}
    marker=directory/'config.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))!=identity:raise ValueError('Resume identity differs')
    write_json(marker,identity)
    last=directory/'last.pt';start,stale,best_loss,best_epoch,history=0,0,float('inf'),0,[]
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start,stale,best_loss,best_epoch,history=(saved[k] for k in ('epoch','stale','best_loss','best_epoch','history'))
        restore_rng(saved['rng'],loader_rng)
    previous=cpu_state(model)
    since=time.time()
    for current in range(start+1,epochs+1):
        if not full and stale>=4:break
        train_loss=epoch(model,train_loader,device,optimizer,limit_batches)
        val_loss=epoch(model,val_loader,device,limit=limit_batches) if not full else None
        history.append({'epoch':current,'train_nll':train_loss,'validation_nll':val_loss})
        snapshot=cpu_state(model)
        if not full:
            if val_loss<best_loss-1e-4:
                if current>1:save_checkpoint(directory/'before_best.pt',{'model':previous,'epoch':current-1})
                best_loss,best_epoch,stale=val_loss,current,0
                save_checkpoint(directory/'best.pt',{'model':snapshot,'epoch':current})
                (directory/'after_best.pt').unlink(missing_ok=True)
            else:
                stale+=1
                if current==best_epoch+1:save_checkpoint(directory/'after_best.pt',{'model':snapshot,'epoch':current})
        save_checkpoint(last,dict(model=snapshot,optimizer=optimizer.state_dict(),epoch=current,
            rng=rng_state(loader_rng),stale=stale,best_loss=best_loss,best_epoch=best_epoch,history=history))
        write_json(directory/'history.json',history)
        print(f'{directory.name} epoch={current}/{epochs} train={train_loss:.5f} val={val_loss} best={best_epoch} seconds={time.time()-since:.1f}',flush=True)
        previous=snapshot
    write_json(directory/'complete.json',{'epochs':history[-1]['epoch'],'best_epoch':best_epoch if not full else epochs,
        'best_validation_nll':best_loss if not full else None,'stopped_by_patience':stale>=4,
        'validation_boundary':not full and best_epoch==epochs,'checkpoint_sha256':digest(last)})
    return last


@torch.inference_mode()
def sample_batch(model,vocab,tokenizer,size,rng,temperature,max_length):
    device=next(model.parameters()).device
    bos,eos=vocab['^'],vocab['$']
    current=torch.full((size,1),bos,dtype=torch.long,device=device)
    hidden=None;done=torch.zeros(size,dtype=torch.bool,device=device)
    sequences=[]
    for _ in range(max_length-1):
        logits,hidden=model(current,hidden)
        logits=logits[:,-1]/temperature
        logits[:,bos]=-torch.inf
        current=torch.multinomial(logits.softmax(-1),1,generator=rng)
        current[done]=eos
        sequences.append(current[:,0].cpu())
        done|=current[:,0]==eos
        if bool(done.all()):break
    tokens=torch.stack(sequences,dim=1).numpy()
    return [tokenizer.untokenize(vocab.decode(row)) if complete else None for row,complete in zip(tokens,done.cpu().tolist())]
