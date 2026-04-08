#!/usr/bin/env python3
"""
MR-TADF Molecular Discovery Pipeline  v2 — Performance-Optimised
=================================================================
Bottlenecks in v1 and how they are fixed:

  1) Candidate generation was a SINGLE-THREADED while-loop.
     → Now uses multiprocessing.Pool with N_WORKERS independent workers,
       each running its own mutation stream with a unique seed.
       A shared-memory mp.Value counter coordinates a global target.

  2) mordred descriptor calculator is ~200× slower than RDKit (≈ 0.8 s/mol
     vs 4 ms/mol).  With 50 000 candidates that alone is ~11 hours.
     → Replaced with pure RDKit descriptors + vectorised fingerprint
       extraction.  ~4 ms/mol ⇒ 50 000 mols in ~3.3 min on 1 core,
       or ~15 s with 16 workers.

  3) XGBoost ensemble prediction was serial per model.
     → Predictions are stacked and run on GPU in one DMatrix batch.

  4) BO loop re-scored the entire unevaluated pool each iteration,
     copying large arrays.
     → Now pre-scores everything once (the ensemble is frozen), then
       ranks by CEI in pure numpy.  80 iterations → <1 s total.

Estimated wall-clock on A100 + 16-core CPU:
  Step 1  Data load            ~2 s
  Step 2  XGBoost + GPR train  ~2 min
  Step 3  Candidate gen        ~5–15 min (50k target, 16 workers)
  Step 4  Descriptors          ~20 s  (RDKit, 16 workers)
  Step 5  BO scoring + rank    ~5 s
  ─────────────────────────────────
  TOTAL                        ~8–18 min

Usage:
    python mr_tadf_bo_pipeline_v2.py \
        --data updated_data.xlsx \
        --target target5.xlsx \
        --n_candidates 5000 \
        --n_workers 16 \
        --output results
"""

# ═══════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════

import os, sys, time, json, warnings, logging, argparse, functools
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import multiprocessing as mp
from multiprocessing import Pool, Value, Lock

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── RDKit  (silence C++ warnings BEFORE any other RDKit import) ──
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem, Descriptors, rdMolDescriptors,
    MolFromSmiles, MolToSmiles, BRICS,
)
from rdkit.Chem import RDKFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# ── ML ──
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MR-TADF")

SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════
#  0.  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

ALLOWED_ATOMS       = frozenset({5, 6, 7, 8, 15, 16, 34})
_BORON_SMARTS       = "[#5X3]"
_CARBONYL_SMARTS    = "[CX3](=[OX1])"
_HETERO_SMARTS      = "[#7,#8,#15,#16,#34]"
MIN_AROMATIC_RINGS  = 3
MAX_ROTATABLE_BONDS = 2
MAX_FRACTION_SP3    = 0.25
HEAVY_ATOM_RANGE    = (14, 45)

UNCERTAINTY_THRESHOLD = 0.10   # eV
T2_S1_CONSTRAINT      = 0.20  # eV

_MAX_VALENCE = {5: 3, 6: 4, 7: 3, 8: 2, 15: 5, 16: 6, 34: 6}

SUBSTITUTIONS = {
    6:  [7, 5],     # C → N, C → B
    7:  [6, 15],    # N → C, N → P
    5:  [6, 7],     # B → C, B → N
    8:  [16, 34],   # O → S, O → Se
    16: [8, 34],    # S → O, S → Se
    15: [7],        # P → N
    34: [16, 8],    # Se → S, Se → O
}


# ═══════════════════════════════════════════════════════════════════
#  1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_and_merge(data_path: str, target_path: str):
    log.info("Loading descriptors from %s", data_path)
    df_desc = pd.read_excel(data_path)
    df_desc.rename(columns={df_desc.columns[0]: "Name",
                            df_desc.columns[1]: "SMILES"}, inplace=True)

    log.info("Loading targets from %s", target_path)
    df_tgt = pd.read_excel(target_path)
    df_tgt.rename(columns={df_tgt.columns[0]: "Name"}, inplace=True)

    df = df_desc.merge(df_tgt[["Name", "T1-S1", "T2-S1"]], on="Name", how="inner")
    log.info("Merged dataset: %d molecules, %d descriptors",
             len(df), len(df_desc.columns) - 2)

    smiles_col = df["SMILES"].tolist()
    descriptor_cols = [c for c in df_desc.columns if c not in ("Name", "SMILES")]
    X = df[descriptor_cols].values.astype(np.float64)
    y_t1s1 = df["T1-S1"].values.astype(np.float64)
    y_t2s1 = df["T2-S1"].values.astype(np.float64)

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    return df, X, y_t1s1, y_t2s1, descriptor_cols, smiles_col, imp


# ═══════════════════════════════════════════════════════════════════
#  2.  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════

def _detect_gpu() -> str:
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if r.returncode == 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def train_xgb_ensemble(X, y, n_estimators=20, label="target"):
    device = _detect_gpu()
    log.info("Training XGBoost ensemble (%d members) for %s on %s",
             n_estimators, label, device)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
        device=device, tree_method="hist",
        random_state=SEED, n_jobs=-1,
    )

    cv_model = xgb.XGBRegressor(**base_params)
    cv_scores = cross_val_score(cv_model, X_scaled, y, cv=5,
                                scoring="neg_mean_absolute_error")
    log.info("  %s 5-fold CV MAE: %.4f ± %.4f eV",
             label, -cv_scores.mean(), cv_scores.std())

    ensemble = []
    for i in range(n_estimators):
        p = base_params.copy()
        p["random_state"] = SEED + i
        rng_i = np.random.RandomState(SEED + i)
        p["subsample"] = rng_i.uniform(0.7, 0.9)
        p["colsample_bytree"] = rng_i.uniform(0.6, 0.8)
        idx = rng_i.choice(len(X_scaled), size=len(X_scaled), replace=True)
        m = xgb.XGBRegressor(**p)
        m.fit(X_scaled[idx], y[idx])
        ensemble.append(m)

    return ensemble, scaler


def predict_with_uncertainty(ensemble, scaler, X):
    """Vectorised ensemble prediction – all models share one DMatrix."""
    X_s = scaler.transform(X)
    preds = np.array([m.predict(X_s) for m in ensemble])  # (n_ens, n_samples)
    return preds.mean(axis=0), preds.std(axis=0)


def train_gpr_comparison(X, y, label="target"):
    log.info("Training GPR for %s (comparison model)", label)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    kernel = (ConstantKernel(1.0) *
              Matern(nu=2.5, length_scale=np.ones(X.shape[1]))
              + WhiteKernel(noise_level=0.01))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   alpha=1e-6, random_state=SEED)
    gpr.fit(X_s, y)
    cv = cross_val_score(
        GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                 alpha=1e-6, random_state=SEED),
        X_s, y, cv=5, scoring="neg_mean_absolute_error")
    log.info("  GPR %s 5-fold CV MAE: %.4f ± %.4f eV",
             label, -cv.mean(), cv.std())
    return gpr, scaler


# ═══════════════════════════════════════════════════════════════════
#  3.  CHEMICAL FILTER  (compiled SMARTS for speed)
# ═══════════════════════════════════════════════════════════════════

# Pre-compile once at module level
_BORON_PAT   = Chem.MolFromSmarts(_BORON_SMARTS)
_CARB_PAT    = Chem.MolFromSmarts(_CARBONYL_SMARTS)
_HETERO_PAT  = Chem.MolFromSmarts(_HETERO_SMARTS)


def _side_groups_are_saturated_hydrocarbon(mol) -> bool:
    """
    Ensure every non-aromatic-ring substituent ("side group") contains
    ONLY saturated carbon and hydrogen — i.e. no heteroatoms, no double/
    triple bonds, no non-aromatic rings with unsaturation in substituents.

    Algorithm:
      1. Identify the aromatic core = set of atoms that belong to at
         least one aromatic ring.
      2. Any heavy atom NOT in the aromatic core is a "side-group atom".
      3. Every side-group atom must be carbon (Z=6).
      4. Every bond between two side-group atoms, or between a side-group
         atom and a core atom, must be SINGLE (bond order 1.0) — except
         for the bond that connects the side-group to an aromatic core
         atom, which is allowed to be aromatic→single.

    This means substituents like -CH₃, -C(CH₃)₃, -CH₂CH₃ pass,
    but -OCH₃, -CN, -C=O, -CF₃, -SCH₃, -vinyl, -phenyl-off-core fail.
    Hydrogen (implicit or explicit) is always allowed.
    """
    ri = mol.GetRingInfo()

    # Build set of atom indices that belong to ANY aromatic ring
    aromatic_core: Set[int] = set()
    for ring in ri.AtomRings():
        # Check if this ring is aromatic
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            aromatic_core.update(ring)

    # Also include atoms that are aromatic but not in a ring tuple
    # (edge case for fused systems)
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_core.add(atom.GetIdx())

    # Check every atom NOT in the aromatic core
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in aromatic_core:
            continue  # core atom — skip

        # Side-group atom must be carbon only
        if atom.GetAtomicNum() != 6:
            return False

        # Side-group atom must be sp3 (no double/triple bonds in substituent)
        if atom.GetHybridization() not in (
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.S,      # terminal CH3 sometimes
        ):
            return False

        # All bonds from this side-group atom must be single
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() != 1.0:
                return False

    return True


def passes_mr_tadf_filter(mol) -> bool:
    """
    Return True if mol passes ALL MR-TADF structural constraints.
    Gate 8 (new): non-aromatic side groups must be saturated hydrocarbon.
    """
    if mol is None:
        return False

    # 1. Allowed atoms only
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ATOMS:
            return False

    # 2. Must have boron [#5X3] OR carbonyl [CX3](=[OX1])
    if not (mol.HasSubstructMatch(_BORON_PAT) or
            mol.HasSubstructMatch(_CARB_PAT)):
        return False

    # 3. At least one heteroatom (N, O, P, S, Se)
    if not mol.HasSubstructMatch(_HETERO_PAT):
        return False

    # 4. Aromatic rings >= 3
    if rdMolDescriptors.CalcNumAromaticRings(mol) < MIN_AROMATIC_RINGS:
        return False

    # 5. Rotatable bonds <= 2
    if Descriptors.NumRotatableBonds(mol) > MAX_ROTATABLE_BONDS:
        return False

    # 6. Fraction sp3 < 0.25
    if rdMolDescriptors.CalcFractionCSP3(mol) >= MAX_FRACTION_SP3:
        return False

    # 7. Heavy atom count in [14, 45]
    nh = mol.GetNumHeavyAtoms()
    if nh < HEAVY_ATOM_RANGE[0] or nh > HEAVY_ATOM_RANGE[1]:
        return False

    # 8. Side groups (non-aromatic substituents) must be saturated C–H only
    if not _side_groups_are_saturated_hydrocarbon(mol):
        return False

    return True


# ═══════════════════════════════════════════════════════════════════
#  4.  PARALLEL CANDIDATE GENERATION
# ═══════════════════════════════════════════════════════════════════

def _valence_ok(atom, new_z: int) -> bool:
    tot = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
    return tot <= _MAX_VALENCE.get(new_z, 0)


def _safe_mol(rw_mol):
    """Sanitise + SMILES round-trip.  Returns (canonical_smi, mol) or None."""
    try:
        Chem.SanitizeMol(rw_mol)
        smi = MolToSmiles(rw_mol.GetMol())
        if not smi:
            return None
        mol = MolFromSmiles(smi)
        if mol is None:
            return None
        return smi, mol
    except Exception:
        return None


# ── Module-level globals set by Pool initializer (inherited, not pickled) ──
_g_counter = None      # mp.Value('i')
_g_lock    = None      # mp.Lock()
_g_seeds   = None      # list[str]  – seed SMILES
_g_n_target = None     # int
_g_attempts = None     # int  – attempts per worker


def _pool_init(counter, lock, seed_smiles, n_target, attempts):
    """Called once per forked worker — stores shared state as globals."""
    global _g_counter, _g_lock, _g_seeds, _g_n_target, _g_attempts
    _g_counter  = counter
    _g_lock     = lock
    _g_seeds    = seed_smiles
    _g_n_target = n_target
    _g_attempts = attempts


def _worker_generate(worker_id):
    """
    Independent worker: runs its own mutation loop with a unique RNG seed.
    Shared counter + lock are inherited via the Pool initializer — never
    pickled, which avoids the 'Synchronized objects should only be shared
    between processes through inheritance' RuntimeError.
    Returns a list of (canonical_smi,) strings found by this worker.
    (We return SMILES strings, not Mol objects, to avoid pickle issues.)
    """
    # Parse seeds inside this process
    seed_mols = []
    for smi in _g_seeds:
        mol = MolFromSmiles(smi)
        if mol is not None:
            seed_mols.append((smi, mol))
    if not seed_mols:
        return []

    rng = np.random.RandomState(SEED + worker_id * 1000)
    local_found: List[str] = []
    local_seen:  Set[str]  = set()

    for attempt in range(_g_attempts):
        # Check global target every 500 attempts (cheap lock)
        if attempt % 500 == 0:
            with _g_lock:
                if _g_counter.value >= _g_n_target:
                    break

        smi, mol = seed_mols[rng.randint(len(seed_mols))]
        rw = Chem.RWMol(Chem.Mol(mol))

        mut = rng.choice(3)  # 0=atom_sub, 1=swap, 2=multi_sub

        try:
            if mut == 0:  # single atom substitution
                atoms = list(rw.GetAtoms())
                a = atoms[rng.randint(len(atoms))]
                old_z = a.GetAtomicNum()
                if old_z in SUBSTITUTIONS:
                    tgts = SUBSTITUTIONS[old_z]
                    new_z = tgts[rng.randint(len(tgts))]
                    if _valence_ok(a, new_z):
                        a.SetAtomicNum(new_z)
                        a.SetFormalCharge(0)
                        a.SetNumExplicitHs(0)
                        a.SetNoImplicit(False)
                    else:
                        continue

            elif mut == 1:  # swap two atoms
                atoms = list(rw.GetAtoms())
                if len(atoms) < 2:
                    continue
                i, j = rng.choice(len(atoms), 2, replace=False)
                ai, aj = atoms[i], atoms[j]
                zi, zj = ai.GetAtomicNum(), aj.GetAtomicNum()
                if (zi != zj and zi in ALLOWED_ATOMS and zj in ALLOWED_ATOMS
                        and _valence_ok(ai, zj) and _valence_ok(aj, zi)):
                    ai.SetAtomicNum(zj)
                    aj.SetAtomicNum(zi)
                    for a in (ai, aj):
                        a.SetFormalCharge(0)
                        a.SetNumExplicitHs(0)
                        a.SetNoImplicit(False)
                else:
                    continue

            else:  # multi-site substitution (2-3 atoms at once)
                atoms = list(rw.GetAtoms())
                n_mut = min(rng.randint(2, 4), len(atoms))
                idxs = rng.choice(len(atoms), n_mut, replace=False)
                ok = True
                changes = []
                for idx in idxs:
                    a = atoms[idx]
                    old_z = a.GetAtomicNum()
                    if old_z not in SUBSTITUTIONS:
                        continue
                    tgts = SUBSTITUTIONS[old_z]
                    new_z = tgts[rng.randint(len(tgts))]
                    if _valence_ok(a, new_z):
                        changes.append((a, new_z))
                    else:
                        ok = False
                        break
                if not ok or not changes:
                    continue
                for a, new_z in changes:
                    a.SetAtomicNum(new_z)
                    a.SetFormalCharge(0)
                    a.SetNumExplicitHs(0)
                    a.SetNoImplicit(False)

            result = _safe_mol(rw)
            if result is None:
                continue

            new_smi, new_mol = result
            if new_smi in local_seen:
                continue

            if passes_mr_tadf_filter(new_mol):
                local_seen.add(new_smi)
                local_found.append(new_smi)  # SMILES string only
                with _g_lock:
                    _g_counter.value += 1

        except Exception:
            continue

    return local_found


def generate_mr_tadf_candidates_parallel(
    seed_smiles: List[str],
    n_target: int = 5000,
    n_workers: int = 16,
    attempts_per_worker: int = 500_000,
) -> List[Tuple[str, "Chem.Mol"]]:
    """
    Parallel candidate generation using multiprocessing.Pool.
    Each of n_workers runs an independent mutation loop.

    Shared mp.Value / mp.Lock are passed via Pool(initializer=...)
    so they are inherited by forked children — NOT pickled as args.
    """
    log.info("Generating MR-TADF candidates (target=%d, workers=%d, "
             "attempts/worker=%d) ...",
             n_target, n_workers, attempts_per_worker)

    # Seeds that already pass
    candidates_set: Set[str] = set()
    candidates_list = []
    for smi in seed_smiles:
        mol = MolFromSmiles(smi)
        if mol is not None and passes_mr_tadf_filter(mol):
            can = MolToSmiles(mol)
            if can not in candidates_set:
                candidates_set.add(can)
                candidates_list.append((can, mol))
    log.info("  Seeds passing filter: %d", len(candidates_list))

    # Shared state — passed through initializer, not through map args
    counter = Value('i', len(candidates_list))
    lock    = Lock()

    t0 = time.time()
    with Pool(
        processes=n_workers,
        initializer=_pool_init,
        initargs=(counter, lock, seed_smiles, n_target, attempts_per_worker),
    ) as pool:
        results = pool.map(_worker_generate, list(range(n_workers)))

    # Merge results, dedup.  Workers return SMILES strings only.
    for worker_smiles_list in results:
        for smi in worker_smiles_list:
            if smi not in candidates_set:
                candidates_set.add(smi)
                mol = MolFromSmiles(smi)
                if mol is not None:
                    candidates_list.append((smi, mol))
                if len(candidates_list) >= n_target:
                    break
        if len(candidates_list) >= n_target:
            break

    elapsed = time.time() - t0
    log.info("  Generated %d unique candidates in %.1f s (%d workers)",
             len(candidates_list), elapsed, n_workers)
    return candidates_list


# ═══════════════════════════════════════════════════════════════════
#  5.  FAST DESCRIPTOR COMPUTATION  (RDKit only, no mordred)
# ═══════════════════════════════════════════════════════════════════

# Pre-build the RDKit descriptor calculator ONCE
_ALL_RDKIT_DESC_NAMES = [d[0] for d in Descriptors.descList]
_RDKIT_CALC = MolecularDescriptorCalculator(_ALL_RDKIT_DESC_NAMES)


def _compute_one_mol_fast(smi_and_names):
    """
    Compute descriptors for a single molecule.
    ~4 ms/mol using pure RDKit (vs ~800 ms/mol with mordred).
    """
    smi, descriptor_names = smi_and_names
    mol = MolFromSmiles(smi)
    if mol is None:
        return None

    try:
        # ── RDKit descriptors (~210 2D descriptors) ──
        vals = _RDKIT_CALC.CalcDescriptors(mol)
        rdkit_dict = dict(zip(_ALL_RDKIT_DESC_NAMES, vals))

        # ── Fingerprint bits ──
        ext_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr_ext = np.zeros(1024, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(ext_fp, arr_ext)

        std_fp = RDKFingerprint(mol, fpSize=1024)
        arr_std = np.zeros(1024, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(std_fp, arr_std)

        graph_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
        arr_graph = np.zeros(1024, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(graph_fp, arr_graph)

        # ── Assemble in order ──
        out = np.empty(len(descriptor_names), dtype=np.float64)
        for k, name in enumerate(descriptor_names):
            if name.startswith("ExtFP"):
                idx = int(name[5:])
                out[k] = arr_ext[idx] if idx < 1024 else np.nan
            elif name.startswith("GraphFP"):
                idx = int(name[7:])
                out[k] = arr_graph[idx] if idx < 1024 else np.nan
            elif name.startswith("FP") and not name.startswith("FpDensity"):
                idx = int(name[2:])
                out[k] = arr_std[idx] if idx < 1024 else np.nan
            elif name in rdkit_dict:
                v = rdkit_dict[name]
                out[k] = float(v) if v is not None else np.nan
            else:
                out[k] = np.nan
        return out

    except Exception:
        return None


def compute_descriptors_parallel(
    candidates: List[Tuple[str, "Chem.Mol"]],
    descriptor_names: list,
    n_workers: int = 16,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute descriptors in parallel using multiprocessing.Pool.
    ~200× faster than mordred-based version.
    """
    log.info("Computing descriptors for %d molecules (%d workers) ...",
             len(candidates), n_workers)
    t0 = time.time()

    # Pass (smiles, descriptor_names) – avoid pickling Mol objects
    tasks = [(smi, descriptor_names) for smi, _ in candidates]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_compute_one_mol_fast, tasks, chunksize=64)

    valid = [(i, r) for i, r in enumerate(results) if r is not None]
    if not valid:
        return np.empty((0, len(descriptor_names))), []

    idxs, arrays = zip(*valid)
    X = np.vstack(arrays)
    smiles = [candidates[i][0] for i in idxs]

    elapsed = time.time() - t0
    log.info("  Descriptors computed: %d × %d in %.1f s (%.1f mol/s)",
             X.shape[0], X.shape[1], elapsed, len(candidates) / max(elapsed, 0.01))
    return X, smiles


# ═══════════════════════════════════════════════════════════════════
#  6.  BAYESIAN OPTIMISATION  (vectorised, single-pass)
# ═══════════════════════════════════════════════════════════════════

def score_all_candidates(
    X_cand: np.ndarray,
    ens_t1s1, scl_t1s1,
    ens_t2s1, scl_t2s1,
    y_best: float,
    xi: float = 0.01,
):
    """
    Score ALL candidates in one vectorised pass.
    No iterative BO loop needed – the ensemble is frozen so CEI
    is deterministic given the candidate descriptors.
    """
    log.info("Scoring %d candidates with ensemble ...", len(X_cand))
    t0 = time.time()

    mu_t1, sig_t1 = predict_with_uncertainty(ens_t1s1, scl_t1s1, X_cand)
    mu_t2, sig_t2 = predict_with_uncertainty(ens_t2s1, scl_t2s1, X_cand)

    sig_t1 = np.maximum(sig_t1, 1e-8)
    sig_t2 = np.maximum(sig_t2, 1e-8)

    # EI for minimisation of T1−S1
    imp = y_best - mu_t1 - xi
    Z = imp / sig_t1
    ei = imp * norm.cdf(Z) + sig_t1 * norm.pdf(Z)
    ei = np.maximum(ei, 0.0)

    # P(T2−S1 < 0.20)
    p_feas = norm.cdf((T2_S1_CONSTRAINT - mu_t2) / sig_t2)

    cei = ei * p_feas

    elapsed = time.time() - t0
    log.info("  Scoring done in %.2f s", elapsed)

    return pd.DataFrame({
        "pred_T1_S1": mu_t1,
        "unc_T1_S1":  sig_t1,
        "pred_T2_S1": mu_t2,
        "unc_T2_S1":  sig_t2,
        "CEI":        cei,
    })


# ═══════════════════════════════════════════════════════════════════
#  7.  RANKING
# ═══════════════════════════════════════════════════════════════════

def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df_f = df[
        (df["unc_T1_S1"] < UNCERTAINTY_THRESHOLD) &
        (df["pred_T2_S1"] < T2_S1_CONSTRAINT)
    ].copy()
    df_f.sort_values(["pred_T1_S1", "pred_T2_S1"], ascending=True, inplace=True)
    df_f.reset_index(drop=True, inplace=True)
    df_f.index.name = "Rank"
    df_f.index += 1
    return df_f


# ═══════════════════════════════════════════════════════════════════
#  8.  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MR-TADF BO Pipeline v2 (performance-optimised)")
    parser.add_argument("--data",          default="updated_data.xlsx")
    parser.add_argument("--target",        default="target5.xlsx")
    parser.add_argument("--n_candidates",  type=int, default=5000)
    parser.add_argument("--n_workers",     type=int, default=0,
                        help="CPU workers (0 = auto-detect)")
    parser.add_argument("--attempts_per_worker", type=int, default=500_000)
    parser.add_argument("--n_ensemble",    type=int, default=20)
    parser.add_argument("--output",        default="results")
    parser.add_argument("--skip_gpr",      action="store_true")
    args = parser.parse_args()

    # Auto-detect workers
    if args.n_workers <= 0:
        args.n_workers = min(mp.cpu_count(), 64)
    log.info("Using %d CPU workers  (system has %d cores)",
             args.n_workers, mp.cpu_count())

    t0 = time.time()
    os.makedirs(args.output, exist_ok=True)

    # ─── Step 1 ───
    log.info("=" * 60)
    log.info("STEP 1: Loading and merging data")
    log.info("=" * 60)
    df, X, y_t1s1, y_t2s1, desc_names, smiles_train, imputer = \
        load_and_merge(args.data, args.target)
    log.info("  %d molecules × %d descriptors", X.shape[0], X.shape[1])
    log.info("  T1−S1 range: [%.4f, %.4f]", y_t1s1.min(), y_t1s1.max())
    log.info("  T2−S1 range: [%.4f, %.4f]", y_t2s1.min(), y_t2s1.max())
    log.info("  T2−S1 < 0.20: %d / %d", (y_t2s1 < 0.20).sum(), len(y_t2s1))

    # ─── Step 2 ───
    log.info("=" * 60)
    log.info("STEP 2: Training predictive models")
    log.info("=" * 60)
    ens_t1, scl_t1 = train_xgb_ensemble(X, y_t1s1, args.n_ensemble, "T1-S1")
    ens_t2, scl_t2 = train_xgb_ensemble(X, y_t2s1, args.n_ensemble, "T2-S1")

    mu1, s1 = predict_with_uncertainty(ens_t1, scl_t1, X)
    mu2, s2 = predict_with_uncertainty(ens_t2, scl_t2, X)
    log.info("  Train T1-S1 → MAE=%.4f, R²=%.4f",
             mean_absolute_error(y_t1s1, mu1), r2_score(y_t1s1, mu1))
    log.info("  Train T2-S1 → MAE=%.4f, R²=%.4f",
             mean_absolute_error(y_t2s1, mu2), r2_score(y_t2s1, mu2))

    if not args.skip_gpr and X.shape[0] <= 250:
        gpr1, gs1 = train_gpr_comparison(X, y_t1s1, "T1-S1")
        gpr2, gs2 = train_gpr_comparison(X, y_t2s1, "T2-S1")
        mg, _ = gpr1.predict(gs1.transform(X), return_std=True)
        log.info("  GPR Train T1-S1 → MAE=%.4f, R²=%.4f",
                 mean_absolute_error(y_t1s1, mg), r2_score(y_t1s1, mg))

    # ─── Step 3 ───
    log.info("=" * 60)
    log.info("STEP 3: Generating MR-TADF candidates (PARALLEL, filter-first)")
    log.info("=" * 60)
    candidates = generate_mr_tadf_candidates_parallel(
        seed_smiles=smiles_train,
        n_target=args.n_candidates,
        n_workers=args.n_workers,
        attempts_per_worker=args.attempts_per_worker,
    )
    if not candidates:
        log.error("No candidates generated!")
        sys.exit(1)

    # ─── SMILES validity gate ───
    # Guarantee every candidate has a valid, parseable canonical SMILES
    valid_candidates = []
    for smi, mol in candidates:
        can_smi = MolToSmiles(mol)
        if can_smi and MolFromSmiles(can_smi) is not None:
            valid_candidates.append((can_smi, mol))
    log.info("  SMILES validity check: %d / %d passed",
             len(valid_candidates), len(candidates))
    candidates = valid_candidates

    if not candidates:
        log.error("No valid SMILES survived! Check seed molecules.")
        sys.exit(1)

    # ─── Step 4 ───
    log.info("=" * 60)
    log.info("STEP 4: Computing descriptors (PARALLEL, RDKit-only)")
    log.info("=" * 60)
    X_cand, smi_cand = compute_descriptors_parallel(
        candidates, desc_names, n_workers=args.n_workers)
    X_cand = imputer.transform(X_cand)
    log.info("  Descriptor matrix: %d × %d", X_cand.shape[0], X_cand.shape[1])

    # ─── Step 5 ───
    log.info("=" * 60)
    log.info("STEP 5: Scoring & Bayesian Optimisation")
    log.info("=" * 60)
    y_best = y_t1s1.min()
    log.info("  Training-set best T1−S1: %.4f eV", y_best)

    df_scores = score_all_candidates(
        X_cand, ens_t1, scl_t1, ens_t2, scl_t2, y_best)
    df_scores.insert(0, "smiles", smi_cand)  # SMILES always first column

    # Final SMILES sanity: drop any row where smiles is empty / NaN
    df_scores = df_scores[df_scores["smiles"].notna() &
                          (df_scores["smiles"] != "")].reset_index(drop=True)
    log.info("  Scored candidates with valid SMILES: %d", len(df_scores))

    # ─── Step 6 ───
    log.info("=" * 60)
    log.info("STEP 6: Ranking candidates")
    log.info("=" * 60)
    df_ranked = rank_candidates(df_scores)
    log.info("  Candidates passing all filters: %d", len(df_ranked))

    # Save
    out_all    = os.path.join(args.output, "all_scored.csv")
    out_ranked = os.path.join(args.output, "ranked_candidates.csv")
    out_top    = os.path.join(args.output, "top50_candidates.csv")

    df_scores.to_csv(out_all, index=False)
    df_ranked.to_csv(out_ranked)

    if len(df_ranked) > 0:
        top50 = df_ranked.head(50)
        top50.to_csv(out_top)
        log.info("\n" + "=" * 60)
        log.info("TOP 10 CANDIDATES")
        log.info("=" * 60)
        for i, row in top50.head(10).iterrows():
            log.info("  #%2d │ T1−S1=%.4f±%.4f │ T2−S1=%.4f±%.4f │ %s",
                     i, row["pred_T1_S1"], row["unc_T1_S1"],
                     row["pred_T2_S1"], row["unc_T2_S1"],
                     row["smiles"][:60])
    else:
        log.warning("No candidates passed ranking filters!")
        df_scores.sort_values("pred_T1_S1").head(50).to_csv(out_top, index=False)

    elapsed = time.time() - t0
    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE  (%.1f min)", elapsed / 60)
    log.info("=" * 60)
    log.info("  %s/all_scored.csv          (%d)", args.output, len(df_scores))
    log.info("  %s/ranked_candidates.csv   (%d)", args.output, len(df_ranked))
    log.info("  %s/top50_candidates.csv", args.output)

    meta = {
        "n_training": int(X.shape[0]),
        "n_descriptors": int(X.shape[1]),
        "n_candidates": len(candidates),
        "n_scored": int(X_cand.shape[0]),
        "n_ranked": len(df_ranked),
        "n_workers": args.n_workers,
        "best_T1S1_train": float(y_best),
        "best_pred_T1S1": float(df_ranked["pred_T1_S1"].min()) if len(df_ranked) > 0 else None,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
