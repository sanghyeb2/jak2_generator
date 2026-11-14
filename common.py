"""Local standardization/Morgan/score/bootstrap utilities; no generator or historical artifacts."""
from pathlib import Path
from functools import lru_cache
import math
import random
import numpy as np
import pandas as pd
from rdkit import Chem,rdBase,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize as MS
ROOT=Path(__file__).resolve().parent
ALLOWED=set("H B C N O F Si P S Cl Br I".split())
UNCHARGER=MS.Uncharger()
TAUTOMER=MS.TautomerEnumerator()
TAUTOMER.SetMaxTautomers(1000)
TAUTOMER.SetMaxTransforms(2000)
FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

@lru_cache(maxsize=50000)
def _standardize(smiles):
    """문자열 하나 → (제출용 SMILES, 중복 판정용 SMILES). 채점 기준과 동일."""
    if len(smiles) > 1000:
        return None
    smiles = smiles.strip()
    if not smiles or any(c.isspace() for c in smiles) or "|" in smiles:
        return None
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not 0 < mol.GetNumHeavyAtoms() <= 128:
                return None
            # 염·전하·동위원소·atom map을 정리합니다.
            mol = UNCHARGER.uncharge(MS.FragmentParent(MS.Cleanup(mol)))
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
                atom.SetAtomMapNum(0)
            mol = Chem.RemoveHs(mol)
            # 동일한 tautomer 계열을 하나의 대표 구조로 정리합니다.
            forms = TAUTOMER.Enumerate(mol)
            if forms.status != MS.TautomerEnumeratorStatus.Completed:
                return None
            mol = TAUTOMER.PickCanonical(forms)
            Chem.SanitizeMol(mol)
            if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
                return None
            if any(a.GetSymbol() not in ALLOWED or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms()):
                return None
            canonical = Chem.MolToSmiles(mol)
            if len(canonical) > 349:
                return None
            # 제출용 입체정보는 보존하되, 중복 판정에서는 입체 차이를 무시합니다.
            Chem.RemoveStereochemistry(mol)
            identity = Chem.MolToSmiles(mol, isomericSmiles=False)
            return canonical, identity
    except (ValueError, RuntimeError, OverflowError):
        return None

def standardize_smiles(smiles):
    """None이면 제외, 반환 문자열이 같으면 중복입니다."""
    result = _standardize(smiles) if isinstance(smiles, str) else None
    return None if result is None else result[1]

def make_submission(generated_smiles):
    """유효·고유 분자 2,000개를 입력 순서대로 선택합니다. 부족하면 오류."""
    if isinstance(generated_smiles, (str, bytes)):
        raise TypeError("SMILES 하나가 아니라 SMILES 목록을 넣어 주세요.")
    unique = {}  # 중복 판정용 SMILES → 제출용 SMILES
    for raw in generated_smiles:
        result = _standardize(raw) if isinstance(raw, str) else None
        if result is not None:
            canonical, identity = result
            unique.setdefault(identity, canonical)
        if len(unique) == 2000:
            return pd.DataFrame({"id": range(2000), "smiles": list(unique.values())})
    raise ValueError(f"유효·고유 분자가 {len(unique)}개입니다. 2,000개까지 추가 생성하세요.")

def weighted_score(fcd, otd, novelty, diversity_generated, diversity_reference):
    """Return a score from 0 to 100; higher is better.

    FCD >= 0; OTD in [0, 9]; novelty and both IntDiv values in [0, 1].
    IntDiv = mean(1 - Tanimoto) over distinct molecule pairs (no self-pairs).
    Inputs are already computed metrics; this does not evaluate a SMILES CSV.
    """
    values = (fcd, otd, novelty, diversity_generated, diversity_reference)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("All metrics must be finite.")
    if fcd < 0 or not 0 <= otd <= 9 or not all(0 <= x <= 1 for x in values[2:]):
        raise ValueError("Metric values are outside their allowed ranges.")
    diversity_score = 1 - abs(diversity_generated - diversity_reference)
    return 100 * math.fsum((
        0.35 * math.exp(-fcd / 5),
        0.35 * (1 - otd / 9),
        0.15 * novelty,
        0.15 * diversity_score,
    ))

def fingerprints(smiles):
    return [FP.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]

def select_by_transport(pool_fps, fit_fps, novel, target=2000, min_novel=1500):
    references = random.Random(9001).sample(range(len(fit_fps)), target)
    similarities = np.empty((target, len(pool_fps)), dtype=np.float32)
    for row, idx in enumerate(references):
        similarities[row] = DataStructs.BulkTanimotoSimilarity(fit_fps[idx], pool_fps)
    # Treat references with few close matches first, then greedily take their
    # closest unused pool molecule. This approximates a one-to-one OT match.
    order = np.argsort(np.max(similarities, axis=1))
    available = np.ones(len(pool_fps), dtype=bool)
    chosen = []
    known_count = 0
    for row in order:
        allowed = available & (novel | (known_count < target - min_novel))
        scores = np.where(allowed, similarities[row], -1)
        idx = int(np.argmax(scores))
        if scores[idx] < 0:
            raise RuntimeError("Not enough candidates for novelty constraint")
        chosen.append(idx)
        available[idx] = False
        known_count += not bool(novel[idx])
    return chosen
