"""Fresh residual neural selector; imports code only, no historical selection assets."""
from pathlib import Path
import json
from pipeline_utils import configure,digest,save_json,save_receipt

def verify_control(work,path,selected,expected):
    if set(map(int,selected))!=set(map(int,expected)):raise ValueError('Neural zero residual differs from fresh fullprior selected rows')
    if digest(path)!=digest(work/'fullprior_submission.csv'):raise ValueError('Neural control differs from fresh fullprior CSV')

def neural_model():
    import torch
    model=torch.nn.Sequential(torch.nn.Linear(387,128),torch.nn.GELU(),
        torch.nn.Linear(128,64),torch.nn.GELU(),torch.nn.Linear(64,1))
    torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
    return model

def neural_logits(model,features,anchor):
    import torch
    return anchor+(2*torch.tanh(model(features).squeeze(-1)/2)).double()

def neural(a):
    import random,numpy as np,pandas as pd,torch
    from rdkit import DataStructs
    from common import fingerprints
    import selector as s
    work=a.output_dir/'neural';work.mkdir(exist_ok=True)
    cache=a.output_dir/'cache';anchor_run=a.output_dir/'fullprior/lr0.03_seed42_capnone'
    evaluator=s.Evaluator(cache,'cuda');frame=evaluator.frame;feat=evaluator.features
    lookup={text:i for i,text in enumerate(frame.smiles)}
    frame.attrs['baseline_order']=[lookup[text] for text in pd.read_csv(a.output_dir/'initial.csv').sort_values('id').smiles]
    if not all((work/n).exists() for n in ('features.npy','normalization.npz')):
        ref=feat['ref_x'];mu=ref.mean(0);sd=np.maximum(ref.std(0),1e-5)
        z=np.clip((feat['pool_x']-mu)/sd,-6,6);land=feat['landmarks'].astype(np.int64)
        fps=fingerprints(pd.read_csv(cache/'reference.csv').smiles.tolist());land_fps=[fps[i] for i in land]
        similarities=np.array([DataStructs.BulkTanimotoSimilarity(fp,land_fps) for fp in fps],dtype=np.float32)
        sm,ss=similarities.mean(0),np.maximum(similarities.std(0),.05)
        sim=1-np.log10(9*np.array(evaluator.cost[:,land])+1)
        stats=np.empty((len(frame),2),np.float32)
        for start in range(0,len(frame),512):
            block=np.asarray(evaluator.cost[start:start+512]);stats[start:start+512,0]=block.min(1);stats[start:start+512,1]=block.mean(1)
        features=np.column_stack([z,(z*z-1)/np.sqrt(2),np.clip((sim-sm)/ss,-6,6),stats,frame.novel.to_numpy()]).astype(np.float32)
        # Atomic npy write: an interrupted partial feature file is never reused.
        pending=work/'features.pending.npy';np.save(pending,features)
        np.savez(work/'normalization.pending.npz',pca_mean=mu,pca_std=sd,landmarks=land,similarity_mean=sm,similarity_std=ss)
        (work/'normalization.pending.npz').replace(work/'normalization.npz');pending.replace(work/'features.npy')
    features=torch.tensor(np.load(work/'features.npy'),device='cuda')
    configure(42);model=neural_model().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    old=torch.load(anchor_run/'best.pt',map_location='cpu',weights_only=False)
    anchor=old['logits'].double().cuda()
    scale=json.loads((anchor_run/'result.json').read_text(encoding='utf-8'))['fcd_scale']
    config={'seed':42,'lr':.001,'updates':a.neural_updates,'eval_every':25,'patience':6,'clip':1,'temperature':.25,
        'architecture':[387,128,64,1],'residual':'anchor + double(2*tanh(float32_net/2))',
        'loss':'35*(1-exp(-FCD/5))+35*normalized_OT+15*(1-N)+15*abs(IntDiv-RefIntDiv)',
        'fcd_scale':scale,'ot_sample':512,'ot_epsilon':.08,'ot_iterations':35,'min_novel':1500,
        'anchor_sha256':digest(anchor_run/'best.pt'),'cache_manifest_sha256':digest(cache/'manifest.json'),
        'features_sha256':digest(work/'features.npy'),'code_sha256':digest(Path(__file__))}
    cp=work/'config.json'
    if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=config:raise ValueError('Neural resume settings differ')
    save_json(cp,config)
    def chosen(logits):
        q=s.selection_mass(logits/.25,2000)
        return s.round_selection(q.detach().cpu().numpy(),frame.novel.to_numpy(),frame.baseline.to_numpy(),min_novel=1500)
    with torch.no_grad():control=chosen(neural_logits(model,features,anchor))
    s.write_submission(frame,control,work/'control.csv')
    verify_control(a.output_dir,work/'control.csv',control,old['selected'])
    best,metrics,best_step,start,stale,history,finished=control,evaluator(control),0,0,0,[],False
    ref=torch.tensor(feat['ref_x'],device='cuda',dtype=torch.float64)
    x=torch.tensor(feat['pool_x'],device='cuda',dtype=torch.float64)
    low=torch.tensor(feat['low_rank'],device='cuda',dtype=torch.float64)
    mu,cov=ref.mean(0),torch.cov(ref.T);root=s.psd_root(cov+1e-8*torch.eye(ref.shape[1],device='cuda',dtype=ref.dtype))
    novel=torch.tensor(frame.novel.to_numpy(),device='cuda',dtype=torch.float64)
    checkpoint=work/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);s.restore_rng(state)
        best,metrics,best_step,start,stale,history,finished=[state[k] for k in ('best','metrics','best_step','step','stale','history','finished')]
    else:s.atomic_torch_save({'model':model.state_dict(),'anchor':anchor.cpu(),'selected':best,'step':0,'metrics':metrics},work/'best.pt')
    for step in (() if finished else range(start+1,a.neural_updates+1)):
        q=s.selection_mass(neural_logits(model,features,anchor)/.25,2000)
        ids=np.random.default_rng(42*100000+step).choice(len(ref),512,replace=False)
        cost=torch.tensor(np.array(evaluator.cost[:,ids]),device='cuda')
        fcd=s.weighted_fcd(x,q,mu,cov,root)*scale;ot=s.sinkhorn_cost(cost,q.float(),iterations=35)
        loss=35*(1-torch.exp(-fcd/5))+35*ot+15*(1-(q*novel).sum()/2000)+15*abs(s.nystrom_diversity(low,q)-evaluator.ref_div)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite neural loss/gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%25==0 or step==a.neural_updates:
            with torch.no_grad():selected=chosen(neural_logits(model,features,anchor))
            current=evaluator(selected)
            if current['score_proxy']>metrics['score_proxy']+1e-7:
                best,metrics,best_step,stale=selected,current,step,0
                s.atomic_torch_save({'model':model.state_dict(),'anchor':anchor.cpu(),'selected':best,'step':step,'metrics':metrics},work/'best.pt')
            else:stale+=1
            history.append({'step':step,'loss':float(loss.detach()),'metrics':current,'best_step':best_step})
            finished=step==a.neural_updates or stale>=6
            s.atomic_torch_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'best':best,'metrics':metrics,
                'best_step':best_step,'step':step,'stale':stale,'history':history,'finished':finished,
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all()},checkpoint)
            save_json(work/'history.json',history)
            print(f'Neural step={step} best={metrics["score_proxy"]:.6f}',flush=True)
            if finished:break
    saved=torch.load(work/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'])
    with torch.no_grad():replayed=chosen(neural_logits(model,features,saved['anchor'].cuda()))
    if set(replayed)!=set(saved['selected']):raise ValueError('Saved neural forward did not reproduce selection')
    out=a.output_dir/'neural_submission.csv';s.write_submission(frame,best,out)
    s.write_submission(frame,replayed,work/'replay.csv')
    if digest(out)!=digest(work/'replay.csv'):raise ValueError('Replayed CSV hash differs')
    save_json(work/'result.json',{'best_step':best_step,'local_proxy':metrics,'submission_sha256':digest(out),
        'actual_model_forward_replay':'PASS','kaggle_score':None,'full_independent_training_repeat':'not performed'})
    save_receipt(a.output_dir,'neural',[out,work/'best.pt',work/'config.json',work/'result.json',
        work/'features.npy',work/'normalization.npz',work/'replay.csv'])
