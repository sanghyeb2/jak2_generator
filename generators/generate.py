"""Uniform123/124/125 generation, extracted from the observed five-model experiment."""
import argparse,os,sys,json,random
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent
sys.path.insert(0,str(ROOT))
MODELS=('gp','gru','transfer','molgpt','reinvent')
WEIGHTS={m:ROOT/f'weights/{m}/final.pt' for m in MODELS}
WEIGHTS['molgpt']=ROOT/'weights/molgpt/checkpoint.pt'
TEMPERATURE={'gru':1.,'gp':.9,'transfer':.9,'molgpt':1.,'reinvent':.9}
BATCH={'gru':256,'gp':32,'transfer':32,'molgpt':64,'reinvent':128}
from generation_utils import digest,save_json,save_receipt,configure,environment

def seed_quotas(total,seeds):
    base,extra=divmod(total,len(seeds))
    return tuple(base+(i<extra) for i in range(len(seeds)))


def standardize(raw):
    if raw is None:return None
    from common import _standardize
    return _standardize(raw)


def sampler(model_name):
    """Same saved architectures and sampling settings; caller sets RNG AFTER loading."""
    import torch
    device=torch.device('cuda');rng=torch.Generator(device=device)
    if model_name=='gru':
        import gru
        state=torch.load(WEIGHTS['gru'],map_location='cpu',weights_only=True)
        vocab=state['vocab'];model=gru.SmilesGRU(len(vocab)).to(device)
        model.load_state_dict(state['state_dict']);model.eval()
        return lambda size:gru.sample_batch(model,vocab,size,rng,1.),rng
    if model_name=='reinvent':
        import reinvent
        model,vocab,tokenizer,_=reinvent.load_prior(device)
        model.load_state_dict(torch.load(WEIGHTS['reinvent'],map_location='cpu',weights_only=False)['model']);model.eval()
        return lambda size:reinvent.sample_batch(model,vocab,tokenizer,size,rng,.9,128),rng
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if model_name=='molgpt':
        import molgpt
        tokenizer=AutoTokenizer.from_pretrained(molgpt.PRETRAINED,local_files_only=True)
        model=molgpt.load_model(tokenizer,device)
        model.load_state_dict(torch.load(WEIGHTS['molgpt'],map_location='cpu',weights_only=False)['model']);model.eval()
        maximum=256;forbidden=[[tokenizer.pad_token_id],[tokenizer.bos_token_id],[3]]
    else:
        import common
        tokenizer=AutoTokenizer.from_pretrained(common.TOKENIZER_ID,revision=common.TOKENIZER_REVISION,
            cache_dir=PACKAGE/'local_artifacts/hf-cache/hub',local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(common.MODEL_ID,revision=common.MODEL_REVISION,
            cache_dir=PACKAGE/'local_artifacts/hf-cache/hub',trust_remote_code=True,local_files_only=True).to(device)
        model.load_state_dict(torch.load(WEIGHTS[model_name],map_location='cpu',weights_only=True));model.eval()
        maximum=202;forbidden=[[t] for t in (tokenizer.bos_token_id,tokenizer.pad_token_id,tokenizer.mask_token_id,tokenizer.unk_token_id)]
    def batch(size):
        prompt=torch.full((size,1),tokenizer.bos_token_id,device=device,dtype=torch.long)
        kwargs={'attention_mask':torch.ones_like(prompt),'use_cache':True} if model_name=='molgpt' else {
            'eos_token_id':tokenizer.eos_token_id,'pad_token_id':tokenizer.pad_token_id}
        output=model.generate(prompt,do_sample=True,top_k=0,temperature=TEMPERATURE[model_name],
            max_length=maximum,bad_words_ids=forbidden,**kwargs)
        if model_name=='molgpt':return [tokenizer.decode(seq,skip_special_tokens=True).strip()
            if tokenizer.eos_token_id in seq[1:] else None for seq in output.tolist()]
        return tokenizer.batch_decode(output,skip_special_tokens=True)
    return batch,rng


def accept_candidates(rows,seen,raw,values,model,seed,quota):
    accepted=0
    for text,value in zip(raw,values):
        if value is None or value[1] in seen:continue
        seen.add(value[1]);rows.append({'smiles':value[0],'raw_smiles':text,'key':value[1],'source':model,'seed':seed})
        accepted+=1
        if accepted==quota:break
    return accepted


def generate(a,name):
    import random,numpy as np,pandas as pd,torch
    from concurrent.futures import ProcessPoolExecutor
    import generation_utils as s
    sample,rng=sampler(name)
    quotas=seed_quotas(a.count_per_model,a.seeds)
    progress=a.output_dir/f'{name}.progress.pt'
    identity={'manifest':digest(a.generation_config_path),'model':name,'seeds':a.seeds,
        'quotas':quotas,'batch':BATCH[name],'max_draws_per_seed':a.max_draws}
    rows=[];accepted=[0]*3;draws=[0]*3;index=0;seen=set();resuming=False
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Generation resume settings differ')
        rows,accepted,draws,index=[saved[k] for k in ('rows','accepted','draws','index')]
        seen={r['key'] for r in rows};rng.set_state(saved['generator_rng']);s.restore_rng(saved);resuming=True
    with ProcessPoolExecutor(max_workers=max(1,4//a.parallel)) as executor,torch.inference_mode():
        for idx in range(index,3):
            seed=a.seeds[idx]
            if not (resuming and idx==index):configure(seed);rng.manual_seed(seed)
            while accepted[idx]<quotas[idx] and draws[idx]<a.max_draws:
                size=min(BATCH[name],a.max_draws-draws[idx]);raw=sample(size)
                values=list(executor.map(standardize,raw,chunksize=8));draws[idx]+=size
                accepted[idx]+=accept_candidates(rows,seen,raw,values,name,seed,quotas[idx]-accepted[idx])
                s.atomic_torch_save({'identity':identity,'rows':rows,'accepted':accepted,'draws':draws,'index':idx,
                    'generator_rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                    'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all()},progress)
                print(f'{name} seed={seed}: unique={accepted[idx]}/{quotas[idx]}, model_total={len(rows)}, draws={draws[idx]}',flush=True)
            if accepted[idx]!=quotas[idx]:raise RuntimeError(f'{name} seed{seed}: draw budget exhausted')
    folder=a.output_dir/'candidates';folder.mkdir(exist_ok=True)
    out=folder/f'{name}.csv';frame=pd.DataFrame(rows);s.atomic_csv(frame,out)
    seed_files=[]
    for seed,quota in zip(a.seeds,quotas):
        path=folder/f'{name}_seed{seed}.csv';part=frame[frame.seed==seed]
        if len(part)!=quota:raise ValueError('Quota count differs')
        s.atomic_csv(part,path);seed_files.append(path)
    report=out.with_suffix('.generation.json')
    save_json(report,{'identity':identity,'accepted':accepted,'draws':draws,'csv_sha256':digest(out)})
    save_receipt(a.output_dir,name,[out,report,*seed_files]);progress.unlink(missing_ok=True)



def train(name,work):
    import runpy
    output=work/'training'/name
    output.mkdir(parents=True,exist_ok=True)
    if name=='reinvent':
        import reinvent as r
        r.WORK=output
        marker=output/'encoded.pt'
        if not marker.exists():
            import pandas as pd
            model,vocab,tokenizer,info=r.load_prior('cpu')
            rows=pd.read_csv(ROOT/'data/train.csv').smiles.tolist()
            if len(rows)!=6029:raise ValueError('Expected6029 training rows')
            tokens=[tokenizer.tokenize(s) for s in rows]
            if any(t not in vocab for row in tokens for t in row):raise ValueError('Unknown REINVENT token')
            if max(map(len,tokens))>info['max_sequence_length']:raise ValueError('REINVENT sequence too long')
            r.save_checkpoint(marker,{'full':[vocab.encode(row).astype(int).tolist() for row in tokens]})
            del model
        r.fit(1e-4,full=True,epochs=13)
    else:
        sys.argv=[str(ROOT/f'{name}.py'),'--mode','train','--output-dir',str(output)]
        runpy.run_path(str(ROOT/f'{name}.py'),run_name='__main__')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('train','generate'),required=True)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--use-trained',action='store_true')
    a=parser.parse_args();a.output_dir=a.output_dir.resolve()
    if not a.output_dir.is_relative_to(PACKAGE/'runs'):raise ValueError('Output must stay in package/runs')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    configure(42)
    environment(a.output_dir/f'environment_{a.action}_{a.model}.json')
    if a.action=='train':
        from generation_utils import training_identity,digest
        manifest=a.output_dir/'train_manifests'/f'{a.model}.json'
        identity=training_identity(a.model)
        if manifest.exists():
            saved=json.loads(manifest.read_text(encoding='utf-8'))
            if saved['identity']!=identity:raise ValueError('Training inputs/code changed; use new run')
            path=a.output_dir/saved['checkpoint']
            if digest(path)!=saved['sha256']:raise ValueError('Completed training checkpoint changed')
            print('Completed training:',a.model);return
        snapshot=a.output_dir/'training'/a.model/'source_snapshot'
        snapshot.mkdir(parents=True,exist_ok=True)
        import shutil
        for path in ROOT.glob('*.py'):shutil.copy2(path,snapshot/path.name)
        save_json(snapshot/'input_identity.json',identity)
        train(a.model,a.output_dir)
        filename='last.pt' if a.model=='reinvent' else 'checkpoint.pt'
        checkpoint=a.output_dir/'training'/a.model/filename
        save_json(manifest,{'identity':identity,'checkpoint':str(checkpoint.relative_to(a.output_dir)),
                            'sha256':digest(checkpoint),'fresh_training_repeat':'not performed'})
        return
    if a.use_trained:
        filename='last.pt' if a.model=='reinvent' else 'checkpoint.pt'
        WEIGHTS[a.model]=a.output_dir/'training'/a.model/filename
    if not WEIGHTS[a.model].is_file():raise FileNotFoundError(WEIGHTS[a.model])
    a.seeds=[123,124,125];a.count_per_model=10000;a.max_draws=100000;a.parallel=1
    # Identity includes code, assets, trained weights, packages and runtime before any generation.
    from generation_utils import generation_identity
    config=a.output_dir/f'config_{a.model}.json';identity=generation_identity(a.output_dir,a.use_trained)
    a.generation_config_path=config
    if config.exists():
        if json.loads(config.read_text(encoding='utf-8'))!=identity:raise ValueError('Generation inputs/code changed; use new run')
    else:save_json(config,identity)
    from generation_utils import receipt_valid
    if receipt_valid(a.output_dir,a.model):
        print('Completed generator:',a.model);return
    generate(a,a.model)

if __name__=='__main__':main()
