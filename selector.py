"""Local legacy fullprior numerical engine. No historical candidate/model state is read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent

class _SelectionMass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, total):
        if not 0 < total < logits.numel():
            raise ValueError('Selection mass must be strictly between zero and pool size')
        lo, hi = logits.min() - 60, logits.max() + 60
        for _ in range(70):
            mid = (lo + hi) / 2
            if torch.sigmoid(logits - mid).sum() > total:
                lo = mid
            else:
                hi = mid
        q = torch.sigmoid(logits - (lo + hi) / 2)
        ctx.save_for_backward(q)
        return q

    @staticmethod
    def backward(ctx, grad):
        q, = ctx.saved_tensors
        derivative = q * (1 - q)
        correction = (grad * derivative).sum() / derivative.sum().clamp_min(1e-30)
        return derivative * (grad - correction), None

def selection_mass(logits, total):
    """Implicitly differentiated sigmoid projection, sum(q)=total, 0<=q<=1."""
    return _SelectionMass.apply(logits, float(total))

def round_selection(scores, novel, baseline, target=2000, min_novel=1500,
                    max_replacements=None):
    """Maximum summed score subject to size, novelty, and baseline retention.

    Four disjoint categories allow exact count enumeration. No heuristic repair
    can silently break a different constraint. Ties use stable input ordering.
    """
    scores, novel, baseline = np.asarray(scores), np.asarray(novel, bool), np.asarray(baseline, bool)
    if not (len(scores) == len(novel) == len(baseline)) or not np.isfinite(scores).all():
        raise ValueError('Invalid candidate scores or masks')
    if not 0 <= min_novel <= target <= len(scores):
        raise ValueError('Invalid size or novelty requirement')
    cap = target if max_replacements is None else max_replacements
    if not 0 <= cap <= target:
        raise ValueError('Invalid replacement cap')
    groups = {}
    prefixes = {}
    for b in (False, True):
        for n in (False, True):
            ids = np.flatnonzero((baseline == b) & (novel == n))
            ids = ids[np.argsort(-scores[ids], kind='stable')]
            groups[b, n] = ids
            prefixes[b, n] = np.r_[0., np.cumsum(scores[ids])]
    base = np.flatnonzero(baseline)
    base = base[np.argsort(-scores[base], kind='stable')]
    best_base_novel = np.r_[0, np.cumsum(novel[base])]
    best_value, best_counts = -np.inf, None
    # For each outsider count, vectorize its split into known/novel molecules.
    for outside in range(cap + 1):
        remaining = target - outside
        if remaining > len(base):
            continue
        out_n = np.arange(max(0, outside - len(groups[False, False])),
                          min(outside, len(groups[False, True])) + 1)
        out_k = outside - out_n
        base_n = np.maximum.reduce([
            np.full_like(out_n, best_base_novel[remaining]),
            min_novel - out_n,
            np.full_like(out_n, remaining - len(groups[True, False]))])
        feasible = (base_n >= 0) & (base_n <= min(remaining, len(groups[True, True])))
        out_n, out_k, base_n = out_n[feasible], out_k[feasible], base_n[feasible]
        if not len(out_n):
            continue
        base_k = remaining - base_n
        values = (prefixes[False, True][out_n] + prefixes[False, False][out_k]
                  + prefixes[True, True][base_n] + prefixes[True, False][base_k])
        j = int(values.argmax())
        if values[j] > best_value:
            best_value = float(values[j])
            best_counts = {(False, True): out_n[j], (False, False): out_k[j],
                           (True, True): base_n[j], (True, False): base_k[j]}
    if best_counts is None:
        raise ValueError('No subset satisfies the joint constraints')
    chosen = np.concatenate([groups[g][:int(k)] for g, k in best_counts.items()])
    return np.sort(chosen)

def psd_root(matrix, epsilon=1e-9):
    values, vectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * values.clamp_min(epsilon).sqrt()) @ vectors.T

def weighted_fcd(features, mass, ref_mean, ref_cov, ref_root=None):
    n = mass.sum()
    mean = (features * mass[:, None]).sum(0) / n
    centered = features - mean
    cov = (centered.T * mass) @ centered / (n - 1)
    eye = torch.eye(features.shape[1], dtype=features.dtype, device=features.device)
    cov, ref_cov = cov + 1e-8 * eye, ref_cov + 1e-8 * eye
    root = psd_root(ref_cov) if ref_root is None else ref_root
    cross = root @ cov @ root
    eigenvalues = torch.linalg.eigvalsh((cross + cross.T) / 2).clamp_min(1e-12)
    return ((mean - ref_mean).square().sum() + torch.trace(cov)
            + torch.trace(ref_cov) - 2 * eigenvalues.sqrt().sum()).clamp_min(0)

def nystrom_diversity(features, mass):
    """Weighted off-diagonal diversity of a low-rank Tanimoto kernel."""
    diagonal = features.square().sum(1)
    numerator = (features.T @ mass).square().sum() - (mass.square() * diagonal).sum()
    denominator = mass.sum().square() - mass.square().sum()
    return 1 - numerator / denominator.clamp_min(1e-12)

def sinkhorn_cost(cost, mass, epsilon=.08, iterations=35):
    """Balanced entropic OT cost; differentiate through the scaling iterations."""
    kernel = torch.exp(-cost / epsilon)
    return _sinkhorn_kernel(kernel, kernel * cost, mass, iterations)[0]

def _sinkhorn_kernel(kernel, cost_kernel, mass, iterations):
    p = mass / mass.sum()
    v = torch.ones(kernel.shape[1], dtype=kernel.dtype, device=kernel.device)
    for _ in range(iterations):
        u = p / (kernel @ v).clamp_min(1e-30)
        v = (1 / len(v)) / (kernel.T @ u).clamp_min(1e-30)
    # Last row update improves row feasibility for training and hard evaluation.
    u = p / (kernel @ v).clamp_min(1e-30)
    value = (u * (cost_kernel @ v)).sum()
    residual = ((v * (kernel.T @ u)) - 1 / len(v)).abs().sum()
    return value, residual

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_write(path, writer):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_torch_save(obj, path):
    atomic_write(path, lambda temporary: torch.save(obj, temporary))

def save_json(path, obj):
    contents = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    atomic_write(path, lambda temporary: temporary.write_text(contents, encoding='utf-8'))

def atomic_csv(frame, path):
    atomic_write(path, lambda temporary: frame.to_csv(temporary, index=False))

def feature_dependencies():
    from importlib.metadata import version
    import fcd_torch
    chemnet = ROOT/'assets/ChemNet_v0.13_pretrained.pt'
    return {'standardizer_sha256': digest(ROOT/'common.py'),
            'fingerprint_code_sha256': digest(ROOT/'common.py'),
            'chemnet_sha256': digest(chemnet),
            'fcd_torch': version('fcd-torch'), 'torch': version('torch'),
            'numpy': version('numpy')}

def _standardize_worker(smiles):
    from common import _standardize
    return _standardize(smiles)

def canonical_rows(rows, path, workers):
    """Restartable, ordered CPU standardization; invalid rows are never skipped."""
    from concurrent.futures import ProcessPoolExecutor
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['smiles', 'key'])
    if len(done) > len(rows) or done.isna().any().any():
        raise ValueError('Invalid standardization checkpoint')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(len(done), len(rows), 1000):
            values = list(executor.map(_standardize_worker, rows[start:start+1000], chunksize=32))
            if any(value is None for value in values):
                raise ValueError(f'Invalid molecule in {path.name} rows {start}..{start+len(values)}')
            part = pd.DataFrame(values, columns=['smiles', 'key'])
            done = pd.concat([done, part], ignore_index=True)
            atomic_csv(done, path)
            print(f'Standardized {path.stem} {len(done)}/{len(rows)}', flush=True)
    return done

class Evaluator:
    """Hard-set rescore: full 512-D FCD, exact diversity, full-prior balanced OT."""
    def __init__(self, cache, device):
        from common import fingerprints
        self.device = device
        self.frame = pd.read_csv(cache / 'candidates.csv')
        self.features = np.load(cache / 'features.npz')
        self.cost = np.load(cache / 'cost.npy', mmap_mode='r')
        self.fps = fingerprints(self.frame.smiles.tolist())
        self.ref_div = float(self.features['ref_div'])
        self.x = torch.tensor(self.features['pool_act'], dtype=torch.float64, device=device)
        ref = torch.tensor(self.features['ref_act'], dtype=torch.float64, device=device)
        self.ref_mean, self.ref_cov = ref.mean(0), torch.cov(ref.T)
        self.root = psd_root(self.ref_cov + 1e-8 * torch.eye(ref.shape[1], device=device, dtype=ref.dtype))
        self.memo = {}

    @torch.no_grad()
    def __call__(self, selected):
        from rdkit import DataStructs
        from common import weighted_score
        selected = np.sort(np.asarray(selected))
        key = tuple(selected)
        if key in self.memo:
            return self.memo[key].copy()
        mass = torch.ones(len(selected), device=self.device, dtype=torch.float64)
        fcd = float(weighted_fcd(self.x[selected], mass, self.ref_mean, self.ref_cov, self.root))
        c = torch.tensor(np.array(self.cost[selected]), device=self.device)
        kernel = torch.exp(-c / .08)
        kc = kernel * c
        ot, residual = _sinkhorn_kernel(kernel, kc, mass.float(), 120)
        if residual > 1e-4:
            ot, residual = _sinkhorn_kernel(kernel.double(), kc.double(), mass, 500)
        if residual > 1e-3:
            raise RuntimeError(f'Unconverged full-prior transport: L1 residual {residual}')
        fps = [self.fps[i] for i in selected]
        similarity = sum(sum(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
                         for i in range(1, len(fps)))
        div = 1 - 2 * similarity / (len(fps) * (len(fps)-1))
        novelty = float(self.frame.novel.to_numpy()[selected].mean())
        score = weighted_score(fcd, 9 * float(ot), novelty, div, self.ref_div)
        row = {'score_proxy': score, 'fcd_512': fcd, 'otd_sinkhorn_proxy': 9*float(ot),
               'ot_column_L1_residual': float(residual), 'novelty': novelty,
               'diversity_exact': div, 'reference_diversity': self.ref_div,
               'replacements': int((~self.frame.baseline.to_numpy()[selected]).sum()),
               'selected_by_source': self.frame.source.iloc[selected].value_counts().to_dict()}
        self.memo[key] = row
        return row.copy()

def ordered_subset(selected, baseline_order):
    """Keep retained baseline molecules in place; fill only the removed slots."""
    selected = set(map(int, selected))
    baseline_set = set(map(int, baseline_order))
    additions = iter(sorted(selected - baseline_set))
    if len(selected) != len(baseline_order):
        raise ValueError('Subset and baseline sizes differ')
    return [int(i) if int(i) in selected else next(additions) for i in baseline_order]

def write_submission(frame, selected, path):
    if 'baseline_order' in frame.attrs:
        selected = ordered_subset(selected, frame.attrs['baseline_order'])
    chosen = frame.iloc[selected]
    # Canonical structures were validated during feature preparation. Repeating
    # tautomer enumeration here adds minutes to each fresh process, not safety.
    # The independent competition-version validator remains the final gate.
    if len(chosen) != 2000 or chosen.key.nunique() != 2000:
        raise ValueError('Output must contain exactly 2,000 unique cached structures')
    atomic_csv(pd.DataFrame({'id': range(2000), 'smiles': chosen.smiles.tolist()}), path)

def restore_rng(saved):
    # map_location='cuda' also maps byte RNG tensors; the RNG APIs need CPU bytes.
    torch.set_rng_state(saved['torch_rng'].cpu())
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])
    if torch.cuda.is_available() and saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])

def train_run(args, cache, evaluator, lr, seed, cap):
    """Learn per-candidate logits; keep best hard-set score including baseline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    name = f'lr{lr:g}_seed{seed}_cap{cap if cap is not None else "none"}'
    run = args.work / name
    run.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config.update(lr=lr, seed=seed, cap=cap,
                  cache_manifest_sha256=digest(cache / 'manifest.json'),
                  source_sha256=digest(Path(__file__)))
    marker = run / 'config.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != config:
        raise ValueError('Run configuration differs; use a new work directory')
    save_json(marker, config)
    frame, feature = evaluator.frame, evaluator.features
    base, novel = frame.baseline.to_numpy(), frame.novel.to_numpy()
    from common import _standardize
    lookup = {s: i for i, s in enumerate(frame.smiles)}
    initial = np.array([lookup[s] if s in lookup else lookup[_standardize(s)[0]]
                        for s in pd.read_csv(args.baseline).sort_values('id').smiles])
    frame.attrs['baseline_order'] = initial.tolist()
    if novel[initial].sum() < args.min_novel:
        raise ValueError('Baseline violates the requested novelty requirement')
    device = args.device
    x = torch.tensor(feature['pool_x'], device=device, dtype=torch.float64)
    ref = torch.tensor(feature['ref_x'], device=device, dtype=torch.float64)
    low_rank = torch.tensor(feature['low_rank'], device=device, dtype=torch.float64)
    mu, cov = ref.mean(0), torch.cov(ref.T)
    ref_root = psd_root(cov + 1e-8 * torch.eye(x.shape[1], device=device, dtype=x.dtype))
    novel_tensor = torch.tensor(novel, dtype=torch.float64, device=device)
    base_tensor = torch.tensor(base, device=device)
    initial_mass = base_tensor.double()
    baseline_metrics = evaluator(initial)
    projected_baseline = float(weighted_fcd(x, initial_mass, mu, cov, ref_root).detach())
    scale = baseline_metrics['fcd_512'] / max(projected_baseline, 1e-8)
    # Baseline-biased initialization, not a hard retention constraint.
    logit = torch.nn.Parameter(torch.tensor(np.where(base, 2.2, -4.5), device=device, dtype=torch.float64))
    initial_logits = logit.detach().cpu().clone()
    optimizer = torch.optim.Adam([logit], lr=lr)
    best_selected, best_metrics, best_step, start = initial, baseline_metrics, 0, 0
    history, stale = [], 0
    checkpoint = run / 'checkpoint.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        logit.data.copy_(saved['logits'])
        optimizer.load_state_dict(saved['optimizer'])
        best_selected, best_metrics = saved['best_selected'], saved['best_metrics']
        best_step, start = saved['best_step'], saved['step']
        history, stale = saved['history'], saved['stale']
        restore_rng(saved)
        print(f'Resuming {name} at update {start}', flush=True)
        if saved['finished']:
            result_path = run / 'result.json'
            if result_path.exists():
                return json.loads(result_path.read_text(encoding='utf-8'))
            # A stop immediately after the last checkpoint may precede final CSV.
            write_submission(frame, best_selected, run / 'selected.csv')
            result = {'run': name, 'best_step': best_step, 'baseline': baseline_metrics,
                      'fcd_scale': scale,
                      'improvement': best_metrics['score_proxy']-baseline_metrics['score_proxy'],
                      'selected_sha256': digest(run / 'selected.csv'), **best_metrics}
            if best_step == 0:
                atomic_torch_save({'logits': initial_logits, 'selected': best_selected,
                                  'step': 0, 'metrics': best_metrics,
                                  'note': 'Baseline retained.'}, run/'best.pt')
            save_json(result_path, result)
            return result
    last_loss = None
    since = time.time()
    for step in range(start + 1, args.updates + 1):
        temperature = max(.25, 1 - .75 * (step - 1) / max(args.updates-1, 1))
        q = selection_mass(logit / temperature, 2000)
        # Deterministic per-update reference subsets cover the full empirical prior.
        ids = np.random.default_rng(seed * 100000 + step).choice(
            len(ref), min(args.ot_sample, len(ref)), replace=False)
        cost = torch.tensor(np.array(evaluator.cost[:, ids]), device=device)
        fcd = weighted_fcd(x, q, mu, cov, ref_root) * scale
        ot = sinkhorn_cost(cost, q.float(), iterations=args.ot_iterations)
        novelty = (q * novel_tensor).sum() / 2000
        div = nystrom_diversity(low_rank, q)
        score_loss = 35 * (1-torch.exp(-fcd/5)) + 35*ot + 15*(1-novelty) + 15*abs(div-evaluator.ref_div)
        # Optional protection is explicitly separate from the published score.
        cap_penalty = q[~base_tensor].sum() * 0
        if cap is not None:
            cap_penalty = .1 * torch.relu(q[~base_tensor].sum() - cap)
        loss = score_loss + cap_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(logit.grad).all():
            raise RuntimeError(f'Nonfinite loss/gradient at update {step}')
        torch.nn.utils.clip_grad_norm_([logit], 10)
        optimizer.step()
        last_loss = float(loss.detach())
        if step % args.eval_every == 0 or step == args.updates:
            with torch.no_grad():
                weights = selection_mass(logit / temperature, 2000).cpu().numpy()
            selected = round_selection(weights, novel, base, min_novel=args.min_novel,
                                       max_replacements=cap)
            metrics = evaluator(selected)
            improved = metrics['score_proxy'] > best_metrics['score_proxy'] + 1e-7
            if improved:
                best_selected, best_metrics, best_step, stale = selected, metrics, step, 0
                write_submission(frame, selected, run / 'selected.csv')
                atomic_torch_save({'logits': logit.detach().cpu(), 'selected': selected,
                            'step': step, 'metrics': metrics}, run / 'best.pt')
            else:
                stale += 1
            history.append({'update': step, 'temperature': temperature, 'loss': last_loss,
                            'fcd_projected_scaled': float(fcd.detach()),
                            'otd_sample_proxy': 9*float(ot.detach()),
                            'diversity_nystrom': float(div.detach()),
                            'novelty_soft': float(novelty.detach()),
                            'cap_penalty': float(cap_penalty.detach()),
                            'hard_metrics': metrics, 'best_step': best_step})
            finished = step == args.updates or (stale >= args.patience and metrics['replacements'] > 0)
            atomic_torch_save({'logits': logit.detach().cpu(), 'optimizer': optimizer.state_dict(),
                        'step': step, 'best_step': best_step, 'best_selected': best_selected,
                        'best_metrics': best_metrics, 'history': history, 'stale': stale,
                        'torch_rng': torch.get_rng_state(),
                        'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        'finished': finished}, checkpoint)
            save_json(run / 'history.json', history)
            print(f'{name} update={step} soft_loss={last_loss:.4f} hard={metrics["score_proxy"]:.5f}'
                  f' best={best_metrics["score_proxy"]:.5f} replaced={metrics["replacements"]}'
                  f' seconds={time.time()-since:.1f}', flush=True)
            if finished:
                break
    write_submission(frame, best_selected, run / 'selected.csv')
    # Always save a model artifact, even when the baseline remains best.
    if best_step == 0:
        atomic_torch_save({'logits': initial_logits, 'selected': best_selected, 'step': 0, 'metrics': best_metrics,
                    'note': 'Baseline retained; no learned checkpoint improved the hard criterion.'}, run / 'best.pt')
    result = {'run': name, 'best_step': best_step, 'fcd_scale': scale,
              'baseline': baseline_metrics, 'improvement': best_metrics['score_proxy']-baseline_metrics['score_proxy'],
              'selected_sha256': digest(run / 'selected.csv'), **best_metrics}
    save_json(run / 'result.json', result)
    return result

def validate(argv=None):
    from rdkit import rdBase
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('paths', type=Path, nargs='+')
    parser.add_argument('--train', type=Path, default=ROOT/'inputs/train.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT/'runs/fresh_seed42/initial.csv')
    parser.add_argument('--min-novel', type=int, default=1500)
    parser.add_argument('--caps', type=int, nargs='+', help='Per-file replacement caps; -1 means unrestricted')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if args.caps is not None and len(args.caps) != len(args.paths):
        parser.error('Supply exactly one replacement cap per file')
    cache = args.report.parent / f'validation_cache_{rdBase.rdkitVersion}'
    cache.mkdir(parents=True, exist_ok=True)
    def keys(path):
        return canonical_rows(pd.read_csv(path).smiles.tolist(), cache/f'{digest(path)}.csv', args.workers).key.tolist()
    train_keys, baseline_keys = set(keys(args.train)), set(keys(args.baseline))
    rows = []
    for index, path in enumerate(args.paths):
        frame = pd.read_csv(path)
        schema_ok = list(frame.columns) == ['id', 'smiles'] and sorted(frame.id.tolist()) == list(range(2000))
        identities = keys(path)
        novelty = sum(k not in train_keys for k in identities)
        replacements = sum(k not in baseline_keys for k in identities)
        cap = None if args.caps is None or args.caps[index] == -1 else args.caps[index]
        passed = (schema_ok and len(identities) == 2000 and None not in identities and len(set(identities)) == 2000
                  and novelty >= args.min_novel and (cap is None or replacements <= cap))
        rows.append({'path': str(path), 'sha256': digest(path), 'passed': passed,
                     'valid_count': sum(k is not None for k in identities), 'unique_count': len(set(identities)-{None}),
                     'novel_count': novelty, 'replacements': replacements, 'max_replacements': cap})
        print(path.name, 'PASS' if passed else 'FAIL', flush=True)
    save_json(args.report, {'rdkit': rdBase.rdkitVersion, 'files': rows,
                            'train_sha256': digest(args.train), 'baseline_sha256': digest(args.baseline)})
    if not all(r['passed'] for r in rows):
        raise SystemExit(1)
