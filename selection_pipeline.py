"""Fresh fullprior -> optional neural experiment from six input CSVs only.

Default stops after validated fullprior submission. No generators, inherited
logit tables, winning submissions or old feature caches are read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
PROJECT=HERE  # standalone package root
sys.path.insert(0,str(HERE))
from pipeline_utils import digest,save_json,save_receipt,receipt_valid,atomic_numpy
MODELS=('gp','gru','transfer','molgpt','reinvent')
ALL_STEPS=('prepare','features','fullprior','validate_fullprior','neural','validate_neural')
CODE=('selection_pipeline.py','pipeline_utils.py','neural_stage.py','common.py','selector.py')


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('fullprior','neural','all'),default='fullprior')
    parser.add_argument('--candidate-dir',type=Path,default=HERE/'inputs/candidates')
    parser.add_argument('--train',type=Path,default=HERE/'inputs/train.csv')
    parser.add_argument('--output-dir',type=Path,default=HERE/'runs/fresh_seed42')
    parser.add_argument('--rdkit-path',type=Path,default=HERE/'runtime/rdkit-2025.9.6')
    parser.add_argument('--fullprior-updates',type=int,default=300)
    parser.add_argument('--neural-updates',type=int,default=300)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',choices=ALL_STEPS,help=argparse.SUPPRESS)
    a=parser.parse_args(argv)
    for key in ('candidate_dir','train','output_dir','rdkit_path'):setattr(a,key,getattr(a,key).resolve())
    if not a.output_dir.is_relative_to(HERE/'runs') or a.output_dir==HERE/'runs':raise ValueError('Output must stay inside this experiment/runs')
    if min(a.fullprior_updates,a.neural_updates,a.workers)<1:raise ValueError('Updates/workers must be positive')
    return a


def steps_for(stage):
    return ALL_STEPS[:4] if stage=='fullprior' else ALL_STEPS[4:] if stage=='neural' else ALL_STEPS


def data_inputs(a):return [a.candidate_dir/f'{m}.csv' for m in MODELS]+[a.train]


def require_fullprior(work):
    for step in ALL_STEPS[:4]:
        if not receipt_valid(work,step):raise ValueError('Run fullprior stage and inspect its submission before starting neural')


def candidate(text,key):
    """Check supplied CANONICAL molecule/key without repeating tautomer search.

Inputs are snapshots of our already-standardized generator output. This is
not a raw-SMILES standardization API. Official final validation still applies.
"""
    from rdkit import Chem
    from common import ALLOWED
    if not isinstance(text,str) or not text or len(text)>349 or any(c.isspace() for c in text) or '|' in text:
        raise ValueError('Expected nonempty standardized SMILES<=349')
    mol=Chem.MolFromSmiles(text)
    if mol is None or not 0<mol.GetNumHeavyAtoms()<=128:raise ValueError('Invalid candidate molecule')
    if not any(a.GetSymbol()=='C' for a in mol.GetAtoms()):raise ValueError('Carbon required')
    if any(a.GetSymbol() not in ALLOWED or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):raise ValueError('Unsupported element/radical')
    canonical=Chem.MolToSmiles(mol)
    Chem.RemoveStereochemistry(mol);identity=Chem.MolToSmiles(mol,isomericSmiles=False)
    if not isinstance(key,str) or identity!=key:raise ValueError('Supplied standardized identity key differs')
    return canonical,identity


def merge_pools(pools):
    result={}
    for model,rows in pools.items():
        seen=set()
        for row in rows:
            text,key=candidate(row['smiles'],row['key'])
            if key in seen:raise ValueError(f'Within-model duplicate: {model}')
            seen.add(key)
            if key not in result:result[key]={'smiles':text,'key':key,'source':model,'generated_by':model}
            else:result[key]['generated_by']+='|'+model
    return list(result.values())


def signature(a):
    from importlib.metadata import version
    import selector as s
    files=data_inputs(a)
    for file in files:
        if not file.is_file():raise FileNotFoundError(file)
    if not (a.rdkit_path/'rdkit').is_dir():raise FileNotFoundError('Official RDKit2025.9.6 runtime folder missing')
    return {'input_csv_sha256':{str(f):digest(f) for f in files},
        'local_code_sha256':{n:digest(HERE/n) for n in CODE},'model_order':list(MODELS),
        'training_seed':42,'reference_sample_seed':9001,'landmark_seed':31415,
        'initialization':'fresh Morgan transport2000, then baseline2.2/others-4.5; no400-proposal pre-search',
        'fullprior_updates':a.fullprior_updates,'neural_updates':a.neural_updates,'workers':a.workers,
        'rdkit_path':str(a.rdkit_path),'packages':{n:version(n) for n in ('torch','numpy','pandas','scipy','rdkit','fcd-torch')},
        'feature_dependencies':s.feature_dependencies(),'historical_learned_selection_dependencies':[]}


def prepare(a):
    import pandas as pd,numpy as np
    from common import fingerprints,select_by_transport
    import selector as s
    work=a.output_dir;folder=work/'cache';folder.mkdir(exist_ok=True)
    pools={}
    for model in MODELS:
        data=pd.read_csv(a.candidate_dir/f'{model}.csv')
        if len(data)!=10000 or not {'smiles','key','seed'}<=set(data.columns):raise ValueError(f'{model}: expected10000 standardized rows with seed/key')
        if data.seed.value_counts().to_dict()!={123:3334,124:3333,125:3333}:raise ValueError(f'{model}: seed quota differs')
        pools[model]=data.to_dict('records')
        print(f'Input {model}:10000 rows, seed quotas checked',flush=True)
    frame=pd.DataFrame(merge_pools(pools))
    if len(frame)<=2000:raise ValueError('Need more than2000 unique candidates')
    train=pd.read_csv(a.train)
    if len(train)!=6029:raise ValueError('Expected6029 training rows')
    refs=s.canonical_rows(train.smiles.tolist(),folder/'standardized_training.csv',a.workers)
    novel=~frame.key.isin(set(refs.key)).to_numpy()
    if novel.sum()<1500:raise ValueError('Not enough novel candidates')
    print(f'Fresh initial2000: pool={len(frame)}, training={len(refs)}; Morgan matching starts',flush=True)
    selected=select_by_transport(fingerprints(frame.smiles.tolist()),fingerprints(refs.smiles.tolist()),novel,
        target=2000,min_novel=1500)
    frame['novel']=novel;frame['baseline']=False;frame.loc[selected,'baseline']=True
    s.atomic_csv(frame,folder/'candidates.csv');s.atomic_csv(refs[['smiles']],folder/'reference.csv')
    s.write_submission(frame,selected,work/'initial.csv')
    save_json(work/'prepare.json',{'raw_count':50000,'unique_count':len(frame),'reference_count':len(refs),
        'initial_novel':int(novel[selected].sum()),'initial_method':'fresh training-reference Morgan transport',
        'learned_prior_used':False,'initial_csv_sha256':digest(work/'initial.csv')})
    save_receipt(work,'prepare',[folder/'candidates.csv',folder/'reference.csv',folder/'standardized_training.csv',work/'initial.csv',work/'prepare.json'])


def embedding_parts(fcd,rows,folder,prefix,identity):
    import numpy as np
    parts=[]
    for start in range(0,len(rows),2048):
        path=folder/f'{prefix}_{start:06d}.npy';meta=path.with_suffix('.json')
        count=min(2048,len(rows)-start)
        if path.exists() and meta.exists():
            state=json.loads(meta.read_text(encoding='utf-8'))
            if state!={'identity':identity,'start':start,'count':count,'sha256':digest(path)}:raise ValueError('Embedding chunk changed')
            value=np.load(path)
        else:
            value=fcd.get_predictions(rows[start:start+count])
            atomic_numpy(path,lambda f:np.save(f,value))
            save_json(meta,{'identity':identity,'start':start,'count':count,'sha256':digest(path)})
        if value.shape!=(count,512) or not np.isfinite(value).all():raise ValueError('Invalid ChemNet output')
        parts.append(value)
        print(f'ChemNet {prefix}: {start+count}/{len(rows)}',flush=True)
    return np.concatenate(parts)


def features(a):
    import numpy as np,pandas as pd
    from fcd_torch import FCD
    from rdkit import DataStructs
    from common import fingerprints
    work=a.output_dir;folder=work/'cache';parts=folder/'embedding_parts';parts.mkdir(exist_ok=True)
    identity=digest(work/'config.json')
    frame=pd.read_csv(folder/'candidates.csv');refs=pd.read_csv(folder/'reference.csv')
    fcd=FCD(device='cuda',n_jobs=1,batch_size=256,model_path=str(HERE/'assets/ChemNet_v0.13_pretrained.pt'))
    act=embedding_parts(fcd,frame.smiles.tolist(),parts,'pool',identity)
    ref_act=embedding_parts(fcd,refs.smiles.tolist(),parts,'training',identity)
    center=ref_act.astype(np.float64).mean(0)
    _,vec=np.linalg.eigh(np.cov(ref_act.astype(np.float64).T));basis=vec[:,-64:]
    x=(act-center)@basis;ref_x=(ref_act-center)@basis
    fps=fingerprints(frame.smiles.tolist());ref_fps=fingerprints(refs.smiles.tolist())
    cost_file=folder/'cost.npy';progress=folder/'cost_progress.json';shape=(len(frame),len(refs))
    state=json.loads(progress.read_text(encoding='utf-8')) if progress.exists() else {'identity':identity,'next':0,'blocks':[]}
    if state['identity']!=identity:raise ValueError('Cost checkpoint inputs differ')
    mode='r+' if state['next'] else 'w+'
    cost=np.lib.format.open_memmap(cost_file,mode=mode,dtype=np.float32,shape=shape)
    if cost.shape!=shape or cost.dtype!=np.float32:raise ValueError('Cost checkpoint shape/dtype differs')
    for block in state['blocks']:
        if hashlib.sha256(np.asarray(cost[block['start']:block['stop']]).tobytes()).hexdigest()!=block['sha256']:raise ValueError('Completed cost block changed')
    for start in range(state['next'],len(frame),1000):
        stop=min(start+1000,len(frame))
        for idx in range(start,stop):
            sim=np.asarray(DataStructs.BulkTanimotoSimilarity(fps[idx],ref_fps),dtype=np.float32)
            cost[idx]=np.expm1(np.log(10.)*(1-sim))/9
        cost.flush();state['next']=stop
        state['blocks'].append({'start':start,'stop':stop,
            'sha256':hashlib.sha256(np.asarray(cost[start:stop]).tobytes()).hexdigest()})
        save_json(progress,state);print(f'Morgan cost: {stop}/{len(frame)}',flush=True)
    land=np.random.default_rng(31415).choice(len(refs),256,replace=False);land_fps=[ref_fps[i] for i in land]
    w=np.array([DataStructs.BulkTanimotoSimilarity(fp,land_fps) for fp in land_fps])
    eig,v=np.linalg.eigh(w);keep=eig>1e-7
    sim=1-np.log10(9*np.asarray(cost[:,land],dtype=np.float64)+1)
    low=sim@(v[:,keep]/np.sqrt(eig[keep]))
    similarity=sum(sum(DataStructs.BulkTanimotoSimilarity(ref_fps[i],ref_fps[:i])) for i in range(1,len(ref_fps)))
    ref_div=1-2*similarity/(len(ref_fps)*(len(ref_fps)-1))
    atomic_numpy(folder/'features.npz',lambda f:np.savez(f,pool_act=act,ref_act=ref_act,pool_x=x,ref_x=ref_x,
        center=center,basis=basis,low_rank=low,landmarks=land,ref_div=ref_div))
    files=[folder/n for n in ('candidates.csv','reference.csv','features.npz','cost.npy')]
    save_json(folder/'manifest.json',{'input_identity':identity,'candidate_count':len(frame),'reference_count':len(refs),
        'reference_fit':'training only','fresh_embeddings':True,'cached_historical_selector_used':False,
        'cache_hashes':{f.name:digest(f) for f in files}})
    save_receipt(work,'features',[*files,folder/'manifest.json'])


def fullprior(a):
    import torch,pandas as pd
    import selector as s
    cache=a.output_dir/'cache'
    args=SimpleNamespace(pool=cache/'candidates.csv',train=a.train,baseline=a.output_dir/'initial.csv',
        work=a.output_dir/'fullprior',device='cuda',dimensions=64,landmarks=256,
        updates=a.fullprior_updates,eval_every=25,patience=4,ot_sample=512,ot_iterations=35,min_novel=1500,
        lrs=[.03],seeds=[42],caps=[-1],workers=a.workers)
    evaluator=s.Evaluator(cache,'cuda')
    result=s.train_run(args,cache,evaluator,.03,42,None)
    run=args.work/result['run'];out=a.output_dir/'fullprior_submission.csv'
    shutil.copy2(run/'selected.csv',out)
    state=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    tau=max(.25,1-.75*(state['step']-1)/max(a.fullprior_updates-1,1))
    q=s.selection_mass(state['logits']/tau,2000)
    frame=evaluator.frame
    lookup={v:i for i,v in enumerate(frame.smiles)}
    frame.attrs['baseline_order']=[lookup[v] for v in pd.read_csv(a.output_dir/'initial.csv').sort_values('id').smiles]
    selected=s.round_selection(q.numpy(),frame.novel.to_numpy(),frame.baseline.to_numpy(),min_novel=1500)
    if set(selected)!=set(state['selected']):raise ValueError('Saved fullprior logits did not reproduce selection')
    replay=a.output_dir/'fullprior_replay.csv';s.write_submission(frame,selected,replay)
    if digest(out)!=digest(replay):raise ValueError('Saved fullprior CSV replay differs')
    save_json(a.output_dir/'fullprior_result.json',{'result':result,'saved_logit_replay':'PASS','public_score':None,
        'full_independent_training_repeat':'not performed'})
    save_receipt(a.output_dir,'fullprior',[out,replay,run/'best.pt',run/'config.json',run/'result.json',a.output_dir/'fullprior_result.json'])


def neural(a):
    require_fullprior(a.output_dir)
    from neural_stage import neural as train_neural
    train_neural(a)


def validate(a,kind):
    import selector as s
    from rdkit import rdBase
    if rdBase.rdkitVersion!='2025.09.6':raise ValueError('Official RDKit2025.9.6 required')
    baseline=a.output_dir/('initial.csv' if kind=='fullprior' else 'fullprior_submission.csv')
    out=a.output_dir/f'{kind}_submission.csv';report=a.output_dir/f'validation_{kind}.json'
    s.validate(['--train',str(a.train),'--baseline',str(baseline),'--workers',str(a.workers),
        '--report',str(report),str(out)])
    save_receipt(a.output_dir,f'validate_{kind}',[out,report])


def worker_env(a,step):
    env=os.environ.copy();prefix=[HERE]
    if step.startswith('validate_'):prefix.insert(0,a.rdkit_path)
    env.update(PYTHONPATH=os.pathsep.join(map(str,prefix)),PYTHONHASHSEED='42',PYTHONIOENCODING='utf-8',
        PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env


def main(argv=None):
    a=parse_args(argv)
    if a.worker:
        from pipeline_utils import configure,environment
        configure(42);save_json(a.output_dir/f'environment_{a.worker}.json',environment())
        if a.worker.startswith('validate_'):validate(a,a.worker.removeprefix('validate_'))
        else:globals()[a.worker](a)
        return
    config=signature(a)
    print(json.dumps({'steps':steps_for(a.stage),'no_generation':True,'no_historical_prior':True,
        'fullprior_output':str(a.output_dir/'fullprior_submission.csv'),
        'neural_output':str(a.output_dir/'neural_submission.csv')},ensure_ascii=False,indent=2),flush=True)
    if a.dry_run:return
    a.output_dir.mkdir(parents=True,exist_ok=True);path=a.output_dir/'config.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=config:raise ValueError('Inputs/code/settings differ; use a new output directory')
    else:
        save_json(path,config);snapshot=a.output_dir/'code_snapshot';snapshot.mkdir(exist_ok=True)
        for name in (*CODE,'README.md','PLAN.md','run.cmd'):
            if (HERE/name).is_file():shutil.copy2(HERE/name,snapshot/name)
    if a.stage=='neural':require_fullprior(a.output_dir)
    for step in steps_for(a.stage):
        if receipt_valid(a.output_dir,step):print('Verified completed stage:',step,flush=True);continue
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',step,'--candidate-dir',str(a.candidate_dir),
            '--train',str(a.train),'--output-dir',str(a.output_dir),'--rdkit-path',str(a.rdkit_path),
            '--fullprior-updates',str(a.fullprior_updates),'--neural-updates',str(a.neural_updates),'--workers',str(a.workers)]
        print('START',step,flush=True)
        with (a.output_dir/f'{step}.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(cmd,cwd=HERE,env=worker_env(a,step),stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            try:
                for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                if process.wait():raise subprocess.CalledProcessError(process.returncode,cmd)
            except BaseException:
                if process.poll() is None:
                    if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    else:process.terminate()
                process.wait();raise
        if not receipt_valid(a.output_dir,step):raise RuntimeError('Worker did not produce verified outputs')
    if a.stage=='fullprior':
        save_json(a.output_dir/'fullprior_complete.json',{'official_validation':'PASS','saved_logit_replay':'PASS',
            'next':'Submit fullprior_submission.csv to Kaggle; then explicitly run --stage neural. No aggregate score is a molecule label.'})
        print('FULLPRIOR READY. Submit fullprior_submission.csv; then run --stage neural.',flush=True)
    else:
        save_json(a.output_dir/'complete.json',{'fullprior_csv_sha256':digest(a.output_dir/'fullprior_submission.csv'),
            'neural_csv_sha256':digest(a.output_dir/'neural_submission.csv'),'official_validation':'PASS',
            'saved_model_replay':'PASS','public_scores':None,'independent_full_training_repeat':'not performed'})
        print('NEURAL READY:',a.output_dir/'neural_submission.csv',flush=True)

if __name__=='__main__':main()
