#!/usr/bin/env python3
"""
MR-TADF Pipeline v18 — v17 with the deep generative block removed
==========================================================================
v18 = v17 minus the deep-generative augmentation module (VAE / Flow /
Diffusion / RL). Everything else from v17 is preserved verbatim:
operator-based candidate generation, chemistry filters, surrogate
ensembles, scoring/ranking/diversity, queues, AD gating, and the
v17 correctness fixes (47-50 plus the seven listed below).

  ── what v18 removes ──────────────────────────────────────────────────
    • The PyTorch generative stack (SMILESVAE, RealNVPFlow,
      LatentDiffusion, RLPolicy, GenerativeOrchestrator,
      `_train_and_sample_generative`).
    • The associated `--enable_deep_gen` / `--deep_gen_*` CLI flags.
    • The deep-gen merge step in `main()` and its metadata entries
      in the run-summary JSON.
    The torch and selfies optional imports remain (try/except-guarded,
    harmless if absent) so the rest of the file is unchanged.

  ── v17 fixes (carried into v18) ──────────────────────────────────────
    1. SMILES VAE tokenizer recognises real multi-character atom symbols
       (Br, Cl, Se, Te, Si). v16's vocabulary flattened these to single
       chars, so the greedy 2-char branch never matched. The 'Cl' atom
       was tokenised as ['C','l'], indistinguishable from a C-l(...)
       fragment by the model. Vocab is now split into single-char and
       multi-char lists; greedy 2-char lookup actually works.
       (Now dead code in v18 since the VAE itself is removed; the
       helpers `_build_smiles_vocab` / `_smiles_tokenize` /
       `_smiles_detokenize` are gone with the rest of the block.)
    2. _ann_BN now produces a topologically correct 1,2-azaborine in the
       canonical Kekulé form RDKit returns for N-aryl, B-aryl azaborines
       (`C1=CB(Ph)N(Ph)C=C1`). v16 emitted aromatic bonds plus
       NoImplicit-fixed N/B, which RDKit could not kekulise → the new
       6-ring stayed non-aromatic and the docstring's "Hückel-aromatic"
       claim was false. v17 emits the canonical Kekulé form, demotes
       the shared edge to single, and lets the sanitiser re-perceive
       the parent ring's aromaticity. Note: with N-aryl + B-aryl
       pendants, the BN 6-ring is NOT aromatic per RDKit's perception
       model; the docstring inside _ann_BN now reflects this.
    3. validation-history log uses the actual run count. v16 logged
       `len(json.load(open(...)))` which returns the dict key count
       (always 1 for `{"runs":[...]}`). v17 reads `["runs"]` first.
    4. SAScorer fragment frequency counts molecule-occurrences (per
       Ertl & Schuffenhauer 2009) instead of summing Morgan-bit
       multiplicities. v16's accumulator inflated weights for
       symmetric/repeated cores common in MR-TADF chemistry.
    5. _pinit forwards a SAScorer reference corpus (sa_ref) so workers
       under multiprocessing 'spawn' (macOS default since Py3.8,
       Windows) can re-fit _GLOBAL_SASCORER. Linux fork remains a
       no-op. v16 silently disabled the SA gate in workers under spawn.
    6. predict_benchmark canonicalises BOTH sides of its label-merge
       map via MolToSmiles(MolFromSmiles(...)). v16 used raw SMILES
       strings as join keys, so non-canonical user input
       (e.g. 'C1=CC=CC=C1' vs 'c1ccccc1') silently produced n=0
       benchmark metrics.
    7. _retro_unfeasible matches all positively-charged nitrogens via
       `[#7+]`. v16's `[NX3+]` SMARTS missed the most common ammonium
       form (quaternary [NX4+]); when the charge gate is disabled, the
       retrosynthesis filter let those through.
   47. Operator cumulative-weight CW[-1] clamped to 1.0 so float drift
       from `W=W/W.sum()` cannot leave a rounding gap that wastes a
       generation attempt.
   48. aza / diaza edit tags now carry a directional `<src>><dst>`
       suffix (e.g. 'aza:S>Se') so the novelty signature distinguishes
       substitution direction instead of collapsing all atom swaps
       into a single 'aza' label. `edits_ok` strips the suffix via
       `_edit_cat` for category-level pairwise checks.
   49. _DBHET_PAT matches aromatic Se via both `se` (aromatic symbol)
       and `#34` (atomic number) so DBSe pendant detection survives
       RDKit-version differences in aromaticity perception.
   50. The all_scored.csv / ranked_candidates.csv / top_diverse_*.csv /
       queue_*.csv / validation_merged_results.csv writes route
       through `_stable_csv_columns` against `_SCORED_CSV_SCHEMA` so
       downstream tools see a deterministic header (missing columns
       padded with NaN, post-hoc extras such as qc_* appended at end).

──────────────────────────────────────────────────────────────────────
v16 feature set (carried through):

Builds on v15. v16 implemented four additions; (1) the deep generative
stack has been removed in v18 — see "what v18 removes" above. The
remaining three (correct ν-DABNA topology, SA gate, charge/radical
gate) are unchanged.

  1. [REMOVED in v18] Deep generative models (VAE / Flow / Diffusion
     / RL). See header above.

  2. Correct ν-DABNA topology in `_ann_BN`. v15's atom-substitution
     produced a non-aromatic borinine. v16 fuses a 6-membered
     1,2-azaborine ring across a perimeter aryl edge of the parent
     BN-MR core: pyrrolic N (donates 2 π e) + trivalent B (empty p,
     accepts 0 π e) + 4 ring carbons → 6 π e Hückel-aromatic. Both
     N and B carry phenyl pendants matching ν-DABNA's substitution
     pattern. RDKit's aromaticity perception passes after sanitize.

  3. Synthetic-accessibility gate at `_hard_filter`:
       (a) `SAScorer` — Ertl-Schuffenhauer SAscore, fragment frequency
                        from the training corpus + complexity penalty
                        (rings, bridgeheads, spiro, stereocenters,
                        macrocycles). Rejects SAscore > SASCORE_MAX.
       (b) `_retro_unfeasible` — heuristic retrosynthesis gate that
                                  rejects patterns not constructible
                                  by routine cross-coupling /
                                  Buchwald-Hartwig / electrophilic
                                  borylation routes.

  4. Charge / radical / open-shell rejection at `_hard_filter`:
       - any atom formal charge ≠ 0 → reject
       - any atom radical electrons > 0 → reject
       - molecule total formal charge ≠ 0 → reject
       Forbids zwitterions, radicals, anionic boronates, ammonium
       salts, and open-shell species that violate the closed-shell
       ground-state assumption used implicitly by the surrogate.

  All v15 features preserved.

  ── v16.1 additive enhancements (carried through) ──────────────────
  All additions are modular, opt-in (or default-on with documented
  fallbacks), and preserve v16's chemistry filters and output schema.

    1. Mandatory scaffold-family AND Murcko-core leave-cluster-out
       cross-validation. Both are written each run; out-of-fold MAE
       per cluster is reported.
    2. Residual cross-target covariance for joint TADF FoM uncertainty:
       Monte-Carlo sampling in score_cands draws CORRELATED noise via
       a Cholesky factor of the training residual correlation matrix
       instead of independent normals.
    3. Scaffold-aware conformal calibration: per-family κ with global
       fallback for novel/under-represented families.
    4. Active-learning validation history: a persistent JSON appended
       on every run (top-N predictions ± matched QC labels) with a
       per-iteration MAE / bias summary.
    5. Label-source weighting: optional per-row sample weights derived
       from a configurable label-source column (e.g. experimental vs
       DFT) propagated to XGB training.
    6. Stacked ensembles: opt-in XGB-bag → ridge meta-learner stack
       (StackedEnsemble); legacy bagged-XGB pred_unc still default.
    7. Interpretability reports: per-target top-K feature importances
       averaged over the bag (or stack base models).
    8. Three candidate queues for next-round QC: exploitation (top
       final_score, AD-clean), exploration (top novelty, AD-clean),
       control (closest to seeds, AD-clean). Tag column 'queue' added.
    9. Stricter applicability-domain gating: hard-reject candidates
       below AD_HARD_THRESHOLD; trust penalty in the soft band
       [AD_HARD, AD_SOFT].
"""

import os, sys, time, json, warnings, logging, argparse, hashlib
from typing import List, Tuple, Dict, Set, FrozenSet
import multiprocessing as mp
from multiprocessing import Pool, Value, Lock
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem, DataStructs
from rdkit.Chem import (AllChem, Descriptors, rdMolDescriptors,
                        MolFromSmiles, MolToSmiles, rdmolops)
from rdkit.Chem import RDKFingerprint
from rdkit.Chem.inchi import MolToInchi
from rdkit.Chem.Scaffolds import MurckoScaffold  # v16.1: Murcko-core grouping
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# v16: optional deep-learning stack for generative augmentation.
# All deep generators (VAE / Flow / Diffusion / RL) gracefully no-op if
# torch is unavailable; the pipeline falls back to v15's operator-only
# generator with a logged warning.
try:
    import torch
    import torch.nn as _tnn
    import torch.nn.functional as _tF
    from torch.utils.data import DataLoader as _TDataLoader, Dataset as _TDataset
    _TORCH_OK=True
except Exception:
    torch=None; _tnn=None; _tF=None; _TDataLoader=None; _TDataset=None
    _TORCH_OK=False
# selfies is optional; if available we use it for VAE input (guarantees
# decoded validity). Fall back to character-level SMILES otherwise.
try:
    import selfies as _selfies
    _SELFIES_OK=True
except Exception:
    _selfies=None; _SELFIES_OK=False
# math used by VAE / diffusion noise schedule
import math as _math
import gzip, pickle

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("MR-TADF"); SEED=42; np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════
# Atomic-number sets (v15):
#   default      = {B,C,N,O,F,P,S,Se,Te}  — P admitted for PO/PS/PSE
#                  family generated in default mode by `_pinsert`. Si/Ge
#                  remain exploratory-only.
#   exploratory  = default ∪ {Si, Ge} for spiro-silole/germole bridges.
ALLOWED_ATOMS=frozenset({5,6,7,8,9,15,16,34,52})
ALLOWED_ATOMS_EXPLR=frozenset({5,6,7,8,9,14,15,16,32,34,52})
_BORON_SMARTS="[#5X3]"; _CARBONYL_SMARTS="[#6X3](=[OX1])"
_HETERO_SMARTS="[#7,#8,#16,#34,#52]"  # N,O,S,Se,Te count as MR heteroatoms
_SELENO_SMARTS="[#34]"; _TELLURO_SMARTS="[#52]"
_PCHALC_SMARTS="[PX4](=[OX1,SX1,#34])"  # P=O, P=S, P=Se
MIN_AROMATIC_RINGS=3; MAX_ROTATABLE_BONDS=3
# v14: bumped from (12,55) to (12,80) so large helicenic / triple-MR
# scaffolds (e.g. multi-resonance double-helicenes, ν-DABNA dimers) are
# admitted. The hard filter still enforces an MR motif and ≥3 aromatic rings.
HEAVY_ATOM_RANGE=(12,80)
# Constraint thresholds. Units assumed eV (energy gaps) and dimensionless (fOSC).
# SOC values assumed in cm⁻¹ (ORCA/Gaussian convention) but unit-aware via CLI.
UNCERTAINTY_THRESHOLD=0.10; T2_T1_CONSTRAINT=0.40; DEST_CONSTRAINT=0.20
SOC1_MIN=0.01; SOC2_MIN=0.05; FOSC_MIN=0.01
# Maximum bond-order sum (≈ valence) accepted for substitution targets.
# Te (group 16) sits at ~3 in tellurophene; Si/Ge tetravalent in spiro-bridges.
_MAX_VALENCE={5:3,6:4,7:3,8:2,9:1,14:4,15:5,16:6,32:4,34:6,52:6}
# v17 (#48): atomic-number → element symbol map used by the aza / diaza
# edit tags so the novelty signature distinguishes substitution direction
# (C→N vs N→C, S→Se vs Se→S, etc.). Without this, all atom swaps collapse
# into a single 'aza' tag and `R{rank}:aza` cannot tell apart e.g. an
# S→Se edit from an Se→Te edit at the same canonical rank.
_ELEM={5:'B',6:'C',7:'N',8:'O',9:'F',14:'Si',15:'P',16:'S',32:'Ge',34:'Se',52:'Te'}
def _aza_tag(prefix,src_z,dst_z):
    """Build a directional aza/diaza tag: '<prefix>:<src>>'<dst>'."""
    s=_ELEM.get(int(src_z),f"Z{int(src_z)}")
    d=_ELEM.get(int(dst_z),f"Z{int(dst_z)}")
    return f"{prefix}:{s}>{d}"
def _edit_cat(t):
    """Strip directional suffix from an edit tag to recover the category
    used by `edits_ok` (e.g. 'aza:C>N' → 'aza', 'diaza:S>N' → 'diaza',
    plain 'F'/'CH3'/'tBu'/... pass through unchanged)."""
    if not isinstance(t,str): return t
    return t.split(":",1)[0]
# Aromatic substitutions: default C↔N, S↔Se↔Te bidirectional.
# v15: bidirectional within each chalcogen group (S↔Se↔Te) and reverse
# O→S/Se/Te swap admitted in exploratory mode. The forward (heavier)
# direction biases generation toward higher SOC; the reverse direction
# trades SOC for smaller ΔE_ST and is occasionally desirable for very-
# blue MR-TADFs where ΔE_ST dominates the FoM.
SUBSTITUTIONS={6:[7],7:[6],16:[34,52],34:[16,52],52:[16,34]}
SUBSTITUTIONS_EXPLR={
    6:  [7,15],            # C → N or P
    7:  [6],               # N → C
    8:  [16,34,52],        # O → S / Se / Te (heavy-atom enhancement)
    16: [8,34,52],         # S ↔ {O, Se, Te} bidirectional
    34: [8,16,52],         # Se ↔ {O, S, Te} bidirectional
    52: [8,16,34],         # Te ↔ {O, S, Se} bidirectional (v15: full reverse)
}
_BP=Chem.MolFromSmarts(_BORON_SMARTS); _CP=Chem.MolFromSmarts(_CARBONYL_SMARTS)
_HP=Chem.MolFromSmarts(_HETERO_SMARTS); _SeP=Chem.MolFromSmarts(_SELENO_SMARTS)
_TeP=Chem.MolFromSmarts(_TELLURO_SMARTS); _PXP=Chem.MolFromSmarts(_PCHALC_SMARTS)
# Donor-pendant fingerprint SMARTS — used by core_topo_desc to count
# how many auxiliary donor units are attached. Conservative patterns
# avoid matching the MR core itself.
_CZ_PAT=Chem.MolFromSmarts("[#6]-[#7;X3;R](-[#6])(-[#6])")  # tertiary aryl-N (Cz/DPA-like)
# v17 (#49): match aromatic Se via both `se` (lowercase aromatic symbol)
# AND `#34` (atomic number). Older RDKit aromaticity perception sometimes
# emits aromatic Se but `[se]` only matches when the perception flag is
# set, while `[#34]` matches by Z regardless; newer RDKit versions have
# tightened the symbol/atomic-number distinction so neither form is a
# strict superset of the other across versions. Listing both inside the
# bracket OR (`,`) keeps DBSe pendant detection version-stable.
_DBHET_PAT=Chem.MolFromSmarts("c1ccc2c(c1)[o,s,se,#34]c1ccccc12")  # DBF / DBT / DBSe
FP_RADIUS=2; FP_NBITS=2048
MAX_WHOLE_SIM_TRAIN=0.95; MIN_CORE_SIM_TRAIN=0.60
MIN_CORE_SIM_PARENT=0.70; MAX_WHOLE_SIM_PARENT=0.999
BAND_TRAIN_WHOLE_CTR=0.65; BAND_TRAIN_WHOLE_SIG=0.15
BAND_PARENT_CORE_CTR=0.82; BAND_PARENT_CORE_SIG=0.08
AD_ALPHA=0.3; AD_BETA=1.0; AD_GAMMA=1.5; AD_DELTA=1.0
# v16.1: stricter applicability-domain (AD) gating.
#   Candidates with AD_score < AD_HARD_THRESHOLD are hard-rejected before
#   ranking (legacy ranking only checked trust gates). Candidates whose
#   AD_score lies in [AD_HARD_THRESHOLD, AD_SOFT_PENALTY] receive a
#   multiplicative trust penalty (down-weight without elimination).
AD_HARD_THRESHOLD=0.15
AD_SOFT_PENALTY=0.40

# Unit registry — declared once, propagated everywhere.
# DeltaEST and T2-T1 share energy units; SOC has its own; fOSC dimensionless.
ENERGY_UNITS="eV"; SOC_UNITS="cm-1"
# Conversion factors to the *internal* unit (eV for energies, cm-1 for SOC).
_EV_FROM={"eV":1.0,"meV":1e-3,"cm-1":1.239841984e-4,"kcal/mol":4.336410390e-2,
          "kJ/mol":1.036426965e-2,"hartree":27.211386246}
_SOC_FROM={"cm-1":1.0,"meV":8.0655439,"eV":8065.5439,"hartree":2.1947463137e5}

# Concordance threshold (in *internal* units) used when comparing predicted
# vs QC values. Used by QC reconciliation and benchmark.
_CONCORD_DEFAULT={"DeltaEST":0.10,"T2-T1":0.15,"OscStr":0.05,
                  "T1-S1(SOC)":0.05,"T2-S1(SOC)":0.05,"Singlets":0.10}

# TADF figure-of-merit numerical floors — prevent log(0) divergences.
# v17 splits the floor per quantity (was a single 1e-4 in v16) so
# inverted-singlet candidates (|ΔE_ST| ≪ 1 meV) retain rank
# discrimination at the design target. All values are in internal units
# (eV / cm⁻¹ / dimensionless).
#
#   _FOM_FLOOR_DEST  = 1e-6 eV  (= 1 µeV = 0.001 meV).
#       v16's 1e-4 eV floor (0.1 meV) collapsed every candidate with
#       |ΔE_ST| < 0.1 meV to the same FoM, killing resolution exactly
#       where the inverted-singlet target lives. 1 µeV is well below
#       any realistic surrogate σ but preserves ranking.
#   _FOM_FLOOR_SOC   = 1e-4 cm⁻¹.  SOC ≈ 0 means no TADF rate;
#       finer resolution gives no physical signal.
#   _FOM_FLOOR_FOSC  = 1e-6.  Very dark (≈ 1 ppm of allowed). A dark
#       state still ranks below any radiating one.
#   _FOM_FLOOR       = 1e-4 (legacy alias; preserved for any external
#       callers that imported the constant directly).
_FOM_FLOOR_DEST = 1e-6
_FOM_FLOOR_SOC  = 1e-4
_FOM_FLOOR_FOSC = 1e-6
_FOM_FLOOR      = 1e-4   # legacy alias — preserved for backwards-compat

# ═══════════════════════════════════════════════════════════════════
#  SCAFFOLD + SITE ROLES
# ═══════════════════════════════════════════════════════════════════
class SF:
    BN="BN-MR"          # boron-nitrogen (DABNA, ν-DABNA, BCz-BN, …)
    CO="CARBONYL-MR"    # carbonyl-MR (DiKTa, QAO, CzBN-CO, …)
    BO="BO-MR"          # boron-oxygen (BOBO/POBO; Hatakeyama 2022)
    NO="NO-MR"          # nitrogen-oxygen (planar N/O alternation)
    SE="SE-MR"          # selenium MR (heavy-atom-enhanced SOC; SeBN, BSe-MR)
    TE="TE-MR"          # tellurium MR — Z=52 boosts SOC by Z⁴ scaling
    PO="PO-MR"          # phosphine oxide as σ*-acceptor MR
    PS="PS-MR"          # phosphine sulfide
    PSE="PSE-MR"        # phosphine selenide
    OT="OTHER"
class SR:
    FROZEN=0; NODE_SAFE=1; HOMO=2; LUMO=3; OVERLAP=4
    STERIC=5; PERIM=6; MODERATE=7; FORBIDDEN=8

# v14: corpus-mass priority. BN dominates the empirical MR-TADF corpus;
# heavy-atom enhancers (Se, Te) and phosphine chalcogenides are minority
# but distinguishable. Order: BN > BO > CO > SE > TE > PO > PS > PSE > NO > OT.
_FAMILY_ORDER=[SF.BN,SF.BO,SF.CO,SF.SE,SF.TE,SF.PO,SF.PS,SF.PSE,SF.NO,SF.OT]

# ═══════════════════════════════════════════════════════════════════
#  v16: SYNTHETIC ACCESSIBILITY (SAscore, Ertl & Schuffenhauer 2009)
# ═══════════════════════════════════════════════════════════════════
# Default cap; overrideable via CLI.
SASCORE_MAX=6.0
# Patterns that are very hard to access by routine cross-coupling /
# Buchwald-Hartwig / Yamamoto / electrophilic borylation. Hard reject.
_RETRO_UNFEASIBLE_SMARTS=[
    "[#6]=[#6]=[#6]",                # cumulene/allene in conjugated π
    "[#6]#[#6]#[#6]",                # poly-yne
    "[#7+]",                         # any N+ (ammonium NX3+/NX4+ etc.)
    "[OX1-]",                        # alkoxide / phenoxide (charge gate)
    "[#5-]",                         # boronate anion
    "[#7][#7][#7]",                  # azide / triazide
    "[#52][#6,#7,#8,#16,#15][#52]",  # two Te bridged through any heteroatom — exotic
    "[#34][#34]",                    # Se-Se bond — disulfide-like instability
    "[#52][#52]",                    # Te-Te bond — unstable
    # >12-membered macrocycle (handled by ring-size scan below, not SMARTS)
]
_RETRO_PATS=[Chem.MolFromSmarts(s) for s in _RETRO_UNFEASIBLE_SMARTS]
_RETRO_PATS=[p for p in _RETRO_PATS if p is not None]

class SAScorer:
    """Ertl & Schuffenhauer (2009) synthetic-accessibility score.

    Definition (https://doi.org/10.1186/1758-2946-1-8):
       SAscore = -fragmentScore + complexityPenalty
       fragmentScore  = average over Morgan-radius-2 fragment-frequency
                        scores (frequent → low SAscore).
       complexityPenalty = log(NumRing) + log(NumStereoCenters)
                          + log(NumSpiro+1) + log(NumBridge+1)
                          + log(NumMacrocycle+1) + size penalty.
       Mapped to [1, 10]. Lower = easier to synthesize.

    The reference distribution of fragment frequencies is normally derived
    from a large drug-like corpus (PubChem ~1M molecules); v16 instead
    uses the supplied training corpus (the existing MR-TADF set + seeds)
    so that fragments common in published MR-TADF emitters score as
    "easy". This makes the gate self-consistent with the design space.

    `score(mol)` returns a float ≥ 1.0. Larger ⇒ harder to make.
    """
    def __init__(self, ref_smiles=None):
        self.fp_freq=Counter()
        self.n_ref=0
        self._fitted=False
        if ref_smiles:
            self.fit(ref_smiles)
    def fit(self, smiles_list):
        # Per Ertl & Schuffenhauer (2009), fp_freq counts the number of
        # MOLECULES that contain each Morgan-radius-2 fragment bit, NOT
        # the sum of multiplicities. Counting multiplicities inflates
        # weights for repetitive substructures (common in MR-TADF cores
        # with multiple equivalent rings) and shifts SAscores downward.
        self.fp_freq=Counter()
        self.n_ref=0
        for s in smiles_list:
            m=MolFromSmiles(s) if isinstance(s,str) else None
            if m is None: continue
            try:
                fp=AllChem.GetMorganFingerprint(m, 2)
                for k in fp.GetNonzeroElements().keys():
                    self.fp_freq[k]+=1   # count molecule, not multiplicity
                self.n_ref+=1
            except Exception:
                continue
        self._fitted=(self.n_ref>=1)
        if self._fitted:
            log.info("  SAScorer fit: %d ref mols, %d unique fragment bits",
                     self.n_ref, len(self.fp_freq))
    def _frag_score(self, mol):
        if not self._fitted: return 0.0
        try:
            fp=AllChem.GetMorganFingerprint(mol, 2)
            els=fp.GetNonzeroElements()
            if not els: return 0.0
            tot=0.0; n=0
            for k,v in els.items():
                f=self.fp_freq.get(k,0)
                # log(freq+1)/log(n_ref+1): bounded in [0,1] then scaled
                s=_math.log(f+1)/_math.log(self.n_ref+1)
                tot+=s*v; n+=v
            return tot/max(n,1)
        except Exception:
            return 0.0
    def _complexity_penalty(self, mol):
        nr=rdMolDescriptors.CalcNumRings(mol)
        nstereo=len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        nspiro=rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nbridge=rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nmacro=sum(1 for r in mol.GetRingInfo().AtomRings() if len(r)>8)
        size_pen=max(0.0, mol.GetNumHeavyAtoms()-50)*0.005
        return (_math.log(nr+1)+_math.log(nstereo+1)+_math.log(nspiro+1)
                +_math.log(nbridge+1)+1.5*_math.log(nmacro+1)+size_pen)
    def score(self, mol):
        if mol is None: return 10.0
        try:
            frag=self._frag_score(mol)        # ∈[0,1] roughly (fitted)
            cplx=self._complexity_penalty(mol)
            # Map: easy frags push down, complexity pushes up.
            raw=-2.5*frag + cplx
            # Re-scale to roughly [1,10] (Ertl's original scaling).
            sa=1.0+raw
            sa=max(1.0, min(10.0, sa))
            return float(sa)
        except Exception:
            return 10.0

# Module-level scorer; populated at run time from training corpus.
_GLOBAL_SASCORER=SAScorer()

def _retro_unfeasible(mol):
    """Heuristic retrosynthesis gate: True ⇔ molecule is rejected.

    Patterns flagged: cumulene/allene/polyyne, charged forms, exotic
    heteroatom dimers (Se-Se, Te-Te), Te-bridged double-heteroatom
    architectures, macrocycles >12 atoms. None of these are reachable
    by the routine cross-coupling / Buchwald-Hartwig / Yamamoto /
    electrophilic borylation sequences typical of MR-TADF synthesis.
    """
    if mol is None: return True
    for p in _RETRO_PATS:
        try:
            if mol.HasSubstructMatch(p): return True
        except Exception:
            continue
    # Macrocycle check (>12-atom ring).
    for r in mol.GetRingInfo().AtomRings():
        if len(r)>12: return True
    # Excessive heavy heteroatom diversity in a single fused-ring system.
    core=_get_core(mol) if mol.GetNumHeavyAtoms()>0 else frozenset()
    if core:
        zs={mol.GetAtomWithIdx(i).GetAtomicNum() for i in core}
        # Heteroatoms in the core (excluding C,H).
        het=zs-{6}
        # ≥4 distinct heteroatom *types* in one fused core is essentially
        # never made in published MR-TADF chemistry.
        if len(het)>=4: return True
        # ≥2 Te or ≥2 Se in core is unusual.
        n_te=sum(1 for i in core if mol.GetAtomWithIdx(i).GetAtomicNum()==52)
        n_se=sum(1 for i in core if mol.GetAtomWithIdx(i).GetAtomicNum()==34)
        if n_te>=2 or n_se>=3: return True
    return False

# ═══════════════════════════════════════════════════════════════════
#  v16: CHARGE / RADICAL / OPEN-SHELL GATE
# ═══════════════════════════════════════════════════════════════════
def _has_charge_or_radical(mol):
    """True ⇔ molecule has any non-zero formal charge or unpaired electron.

    Rejected at the hard filter:
      - any atom with formal charge ≠ 0 (zwitterion, ammonium salt,
        boronate anion, alkoxide, etc.);
      - any atom with radical electrons > 0 (open-shell radical);
      - molecule total formal charge ≠ 0 (anion, cation).

    Justification: the v15 surrogate is trained on closed-shell neutral
    ground-state TDDFT outputs; charged/radical species fall outside
    its applicability domain by construction. Allowing them produces
    silently meaningless predictions.
    """
    if mol is None: return True
    try:
        if Chem.GetFormalCharge(mol)!=0: return True
        for a in mol.GetAtoms():
            if a.GetFormalCharge()!=0: return True
            if a.GetNumRadicalElectrons()>0: return True
        return False
    except Exception:
        return True

def _get_core(mol):
    """Robust MR-TADF core detector (v15).

    Two-stage detection:

      Stage A (legacy, RDKit-aromatic): build the largest fully-aromatic
        fused-ring system via union-find on rings sharing an edge.

      Stage B (v15 fallback, ring-system): for parents whose Stage-A
        result lacks any MR motif (the typical failure mode for B-
        containing central rings that RDKit refuses to aromatise),
        union-find on ALL rings (regardless of aromaticity) and pick the
        largest cluster that contains ≥1 MR motif (B / aromatic-C=O /
        aromatic Se / aromatic Te / P-chalcogenide-neighbour).

    Stage A is always preferred when it covers an MR motif; Stage B is
    triggered specifically when Stage A is "MR-motif-free" — a *single
    pendant aromatic ring* that wins union-find because its 6 atoms
    happen to all be aromatic while the larger B-containing system is
    not aromaticity-perceived.

    Returns a frozenset of atom indices defining the MR core.
    """
    ri = mol.GetRingInfo()
    rings_all = ri.AtomRings()
    if not rings_all: return frozenset()

    # Pre-compute MR-motif atoms once (used by both stages).
    cp = Chem.MolFromSmarts("[#6X3](=[OX1])")
    co_carbons = set()
    if mol.HasSubstructMatch(cp):
        for m in mol.GetSubstructMatches(cp):
            co_carbons.add(m[0])
    p_chalc_neighbors = set()
    for pat_smt in ("[PX4]=[OX1]", "[PX4]=[SX1]", "[PX4]=[#34]"):
        pat = Chem.MolFromSmarts(pat_smt)
        if mol.HasSubstructMatch(pat):
            for m in mol.GetSubstructMatches(pat):
                p_idx = m[0]
                for nb in mol.GetAtomWithIdx(p_idx).GetNeighbors():
                    if nb.GetIdx() != m[1]:
                        p_chalc_neighbors.add(nb.GetIdx())
    def _has_mr_motif(atom_set):
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() in (5,34,52) for i in atom_set):
            return True
        if any(i in co_carbons for i in atom_set): return True
        if any(i in p_chalc_neighbors for i in atom_set): return True
        return False

    def _union_find_clusters(ring_idx_groups):
        n = len(ring_idx_groups)
        if n == 0: return []
        par = list(range(n))
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        for i in range(n):
            for j in range(i+1, n):
                if len(ring_idx_groups[i] & ring_idx_groups[j]) >= 2:
                    a, b = find(i), find(j)
                    if a != b: par[a] = b
        clusters = defaultdict(set)
        for i in range(n): clusters[find(i)] |= ring_idx_groups[i]
        return list(clusters.values())

    # Stage A
    aro_rings = [set(r) for r in rings_all
                 if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    aro_clusters = _union_find_clusters(aro_rings)
    aro_with_mr = [c for c in aro_clusters if _has_mr_motif(c)]
    if aro_with_mr:
        return frozenset(max(aro_with_mr, key=len))

    # Stage B — fallback to all-rings ring-system detection.
    all_rings = [set(r) for r in rings_all]
    all_clusters = _union_find_clusters(all_rings)
    all_with_mr = [c for c in all_clusters if _has_mr_motif(c)]
    if all_with_mr:
        return frozenset(max(all_with_mr, key=len))

    # No MR motif anywhere — return Stage A's largest cluster (or empty).
    if aro_clusters:
        return frozenset(max(aro_clusters, key=len))
    return frozenset()

def _classify(mol, core=None):
    """Multi-family MR-TADF scaffold classifier (v14).

    Returns the highest-priority family for which evidence is found in or
    adjacent to the fused-aromatic core. Priority order (corpus-derived):

        BN > BO > CO > SE > TE > PO > PS > PSE > NO > OT

    BN dominates by mass; the heavy-atom enhancers (SE, TE) and phosphine
    chalcogenides (PO/PS/PSE) are minority MR motifs and only claim the
    label when no BN/BO/CO motif is present.

    Definitions:
      - BN  : aromatic B AND aromatic N in core.
      - BO  : aromatic B AND aromatic O in core (no N rules out BN).
      - CO  : aromatic C=O whose carbonyl-C lies in core.
      - SE  : aromatic Se in core (no BN/BO/CO overrides).
      - TE  : aromatic Te in core.
      - PO  : tetracoordinate P=O attached to / inside core.
      - PS  : tetracoordinate P=S.
      - PSE : tetracoordinate P=Se.
      - NO  : aromatic N AND aromatic O in core (no B, no carbonyl).
      - OT  : everything else.
    """
    if core is None: core=_get_core(mol)
    if not core: return SF.OT
    has_B=any(mol.GetAtomWithIdx(i).GetAtomicNum()==5 for i in core)
    has_N=any(mol.GetAtomWithIdx(i).GetAtomicNum()==7 for i in core)
    has_O=any(mol.GetAtomWithIdx(i).GetAtomicNum()==8 for i in core)
    has_Se=any(mol.GetAtomWithIdx(i).GetAtomicNum()==34 for i in core)
    has_Te=any(mol.GetAtomWithIdx(i).GetAtomicNum()==52 for i in core)
    cp=Chem.MolFromSmarts("[#6X3](=[OX1])")
    has_carbonyl=False
    if mol.HasSubstructMatch(cp):
        for m in mol.GetSubstructMatches(cp):
            if m[0] in core: has_carbonyl=True; break
    # Phosphine chalcogenides: P=O / P=S / P=Se where P is bonded to ≥1 core atom
    has_PO=has_PS=has_PSE=False
    pat_PO=Chem.MolFromSmarts("[PX4]=[OX1]")
    pat_PS=Chem.MolFromSmarts("[PX4]=[SX1]")
    pat_PSe=Chem.MolFromSmarts("[PX4]=[#34]")
    def _p_attached_to_core(pat):
        for m in mol.GetSubstructMatches(pat):
            p_idx=m[0]
            for nb in mol.GetAtomWithIdx(p_idx).GetNeighbors():
                if nb.GetIdx() in core: return True
        return False
    if mol.HasSubstructMatch(pat_PO):  has_PO  = _p_attached_to_core(pat_PO)
    if mol.HasSubstructMatch(pat_PS):  has_PS  = _p_attached_to_core(pat_PS)
    if mol.HasSubstructMatch(pat_PSe): has_PSE = _p_attached_to_core(pat_PSe)
    # Priority cascade.
    if has_B and has_N:   return SF.BN
    if has_B and has_O:   return SF.BO
    if has_carbonyl:      return SF.CO
    if has_Se:            return SF.SE
    if has_Te:            return SF.TE
    if has_PO:            return SF.PO
    if has_PS:            return SF.PS
    if has_PSE:           return SF.PSE
    if has_N and has_O:   return SF.NO
    return SF.OT

def _gdist(mol,idx,tgt):
    if idx in tgt: return 0
    vis={idx}; fr=[idx]; d=0
    while fr:
        d+=1; nf=[]
        for a in fr:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                ni=nb.GetIdx()
                if ni in tgt: return d
                if ni not in vis: vis.add(ni); nf.append(ni)
        fr=nf
    return 999

def _label_sites(mol, core=None, scaff=None):
    """Assign HOMO/LUMO/OVERLAP/PERIM/NODE site roles per atom.

    Donor pool (HOMO contributors): aromatic N, O, S, Se, Te in core —
        their np lone pair contributes to π-HOMO (Te included from v14;
        same orbital symmetry as S/Se, just heavier).
    Acceptor pool (LUMO contributors): aromatic B, carbonyl-C in core,
        and P-chalcogenide P (P=O/S/Se) when attached to core.
    """
    if core is None: core=_get_core(mol)
    if scaff is None: scaff=_classify(mol,core)
    donors=set(); acceptors=set()
    for i in core:
        z=mol.GetAtomWithIdx(i).GetAtomicNum()
        # N, O, S, Se, Te → HOMO contributors
        if z in (7,8,16,34,52): donors.add(i)
        if z==5: acceptors.add(i)
    cp=Chem.MolFromSmarts("[#6X3](=[OX1])")
    if mol.HasSubstructMatch(cp):
        for m in mol.GetSubstructMatches(cp):
            if m[0] in core: acceptors.add(m[0])
    # P-chalcogenide acceptors: P atom adjacent to a core atom counts as
    # acceptor surrogate (LUMO contribution via σ*-PO).
    for pat in (Chem.MolFromSmarts("[PX4]=[OX1]"),
                Chem.MolFromSmarts("[PX4]=[SX1]"),
                Chem.MolFromSmarts("[PX4]=[#34]")):
        if mol.HasSubstructMatch(pat):
            for m in mol.GetSubstructMatches(pat):
                p_idx=m[0]
                for nb in mol.GetAtomWithIdx(p_idx).GetNeighbors():
                    if nb.GetIdx() in core:
                        acceptors.add(nb.GetIdx()); break
    perim=set()
    for i in core:
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() not in core: perim.add(i); break
    L={}
    for atom in mol.GetAtoms():
        idx=atom.GetIdx(); z=atom.GetAtomicNum()
        if idx not in core:
            L[idx]=SR.NODE_SAFE if z==6 else SR.FORBIDDEN; continue
        if z==5: L[idx]=SR.FROZEN; continue
        # Aromatic O, S, Se, Te in core: their lone pair defines the MR HOMO
        # pattern — modifying them destroys the MR character. Frozen.
        if z in (8,16,34,52) and idx in core: L[idx]=SR.FROZEN; continue
        if z==7 and any(mol.GetAtomWithIdx(n.GetIdx()).GetAtomicNum()==5 for n in atom.GetNeighbors()):
            L[idx]=SR.FROZEN; continue
        if z==6:
            is_co=any(b.GetBondTypeAsDouble()==2.0 and mol.GetAtomWithIdx(b.GetOtherAtomIdx(idx)).GetAtomicNum()==8 for b in atom.GetBonds())
            if is_co: L[idx]=SR.FROZEN; continue
        dd=_gdist(mol,idx,donors) if donors else 999
        da=_gdist(mol,idx,acceptors) if acceptors else 999
        if dd<=1 and da<=1: L[idx]=SR.OVERLAP
        elif dd<=2: L[idx]=SR.HOMO
        elif da<=2: L[idx]=SR.LUMO
        elif atom.GetTotalNumHs()>0 and idx in perim:
            L[idx]=SR.STERIC if atom.GetDegree()<=2 else SR.NODE_SAFE
        elif atom.GetTotalNumHs()>0: L[idx]=SR.NODE_SAFE
        else: L[idx]=SR.MODERATE
    for idx in perim:
        a=mol.GetAtomWithIdx(idx)
        if L.get(idx) in (SR.NODE_SAFE,SR.STERIC) and a.GetIsAromatic() and a.GetDegree()==2 and a.GetTotalNumHs()>0:
            for nb in a.GetNeighbors():
                ni=nb.GetIdx()
                if ni in perim and ni in core and nb.GetIsAromatic():
                    L[idx]=SR.PERIM; break
    return L

# ═══════════════════════════════════════════════════════════════════
#  PART 5: TIGHTENED PERMISSION MATRIX
# ═══════════════════════════════════════════════════════════════════
PERM={
    # BN-MR: strictest. F-only at frontier, no aza near it.
    SF.BN:{
        SR.FROZEN:set(), SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3'},
        SR.HOMO:{'F'}, SR.LUMO:{'F'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F','annul'}, SR.MODERATE:{'F'}, SR.FORBIDDEN:set(),
    },
    # CO-MR: similar to BN but tolerates CH3 at MODERATE.
    SF.CO:{
        SR.FROZEN:set(), SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3'},
        SR.HOMO:{'F'}, SR.LUMO:{'F'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F','annul'}, SR.MODERATE:{'F','CH3'}, SR.FORBIDDEN:set(),
    },
    # BO-MR (boron-oxygen): O lone pair on aromatic ring acts as donor;
    # tighter on PERIM/MODERATE because B-O alternation is fragile.
    SF.BO:{
        SR.FROZEN:set(), SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3'},
        SR.HOMO:{'F'}, SR.LUMO:{'F'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F','annul'}, SR.MODERATE:{'F'}, SR.FORBIDDEN:set(),
    },
    # NO-MR: planar N/O alternation. Conservative.
    SF.NO:{
        SR.FROZEN:set(), SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3'},
        SR.HOMO:{'F'}, SR.LUMO:{'F'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3'},
        SR.PERIM:{'F','annul'}, SR.MODERATE:{'F','CH3'}, SR.FORBIDDEN:set(),
    },
    # SE-MR: Se already provides heavy-atom SOC enhancement, so we keep
    # frontier conservative (F-only) and allow paired_aza at NODE_SAFE.
    # Donor grafts permitted at PERIM (Cz/DPA appended to perimeter aryl).
    SF.SE:{
        SR.FROZEN:set(),
        SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3','paired_aza','graft_D'},
        SR.HOMO:{'F'}, SR.LUMO:{'F'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F','CH3','annul','graft_D','CN'}, SR.MODERATE:{'F','CH3'},
        SR.FORBIDDEN:set(),
    },
    # TE-MR: Te is even heavier (Z=52) — Z⁴-scaled SOC ≈ 4–5× Se. We treat
    # the perimeter as more permissive because the SOC budget is huge,
    # so even moderate frontier perturbation still leaves k_RISC large.
    SF.TE:{
        SR.FROZEN:set(),
        SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3','paired_aza','graft_D'},
        SR.HOMO:{'F','CN'}, SR.LUMO:{'F','CN'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F','CH3','annul','graft_D','CN'},
        SR.MODERATE:{'F','CH3','CN'}, SR.FORBIDDEN:set(),
    },
    # PO-MR: phosphine oxide as σ*-acceptor. Frontier behaviour parallels
    # CO-MR. Donor grafts at PERIM are common in published P=O-MR-TADFs.
    SF.PO:{
        SR.FROZEN:set(),
        SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3','graft_D'},
        SR.HOMO:{'F'}, SR.LUMO:{'F'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F','annul','graft_D','CN'},
        SR.MODERATE:{'F','CH3'}, SR.FORBIDDEN:set(),
    },
    # PS-MR: P=S analogue. Same skeleton as PO; SOC slightly higher
    # (S vs O), so a touch more permissive at PERIM.
    SF.PS:{
        SR.FROZEN:set(),
        SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3','graft_D'},
        SR.HOMO:{'F'}, SR.LUMO:{'F'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F','CH3','annul','graft_D','CN'},
        SR.MODERATE:{'F','CH3'}, SR.FORBIDDEN:set(),
    },
    # PSE-MR: P=Se. Highest-SOC member of the P-chalcogenide trio.
    SF.PSE:{
        SR.FROZEN:set(),
        SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3','graft_D'},
        SR.HOMO:{'F','CN'}, SR.LUMO:{'F','CN'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F','CH3','annul','graft_D','CN'},
        SR.MODERATE:{'F','CH3','CN'}, SR.FORBIDDEN:set(),
    },
    # OTHER: most conservative.
    SF.OT:{
        SR.FROZEN:set(), SR.NODE_SAFE:{'F','CH3','paired_F','paired_CH3'},
        SR.HOMO:{'F'}, SR.LUMO:{'F'},
        SR.OVERLAP:set(), SR.STERIC:{'F','CH3','tBu'},
        SR.PERIM:{'F'}, SR.MODERATE:{'F'}, SR.FORBIDDEN:set(),
    },
}
# Donor-graft and CN are universal across all MR families per corpus
# audit. Granted at PERIM and NODE_SAFE (NODE_SAFE is the fall-through
# label for perimeter aryl-CH atoms when RDKit fails to aromatise the
# MR central ring — common with B-containing ones).
for _fam in (SF.BN, SF.CO, SF.BO, SF.NO, SF.OT):
    PERM[_fam][SR.PERIM]    = PERM[_fam][SR.PERIM]    | {'graft_D','CN'}
    PERM[_fam][SR.NODE_SAFE]= PERM[_fam][SR.NODE_SAFE]| {'graft_D','CN'}

# v17: aza substitution permissions. Without these, the aza/F+aza/diaza
# operator branches in `_wgen` are dead code (verified in v17 review):
# the W weight (~25% of generation budget) was being silently absorbed
# by the next operator branch via fall-through. The explicit corpus
# logic in SUBSTITUTIONS (C↔N, S↔Se↔Te) is now actually reachable.
#
# Conservative families (BN, CO, BO) intentionally do NOT receive aza
# permission — those scaffolds are designed around the rigid B/N/O
# pattern and aza substitution would dilute the MR character. NO-MR is
# explicitly aza-friendly (designed around N/O alternation). SE/TE/PO/
# PS/PSE/OT receive aza permission at the role labels actually
# assigned in fully-fused MR cores: NODE_SAFE (the dominant label for
# in-core CH atoms when the molecule has no out-of-core pendants),
# plus PERIM and MODERATE for cores that DO have pendants.
for _fam in (SF.NO, SF.SE, SF.TE, SF.PO, SF.PS, SF.PSE, SF.OT):
    PERM[_fam][SR.NODE_SAFE]= PERM[_fam][SR.NODE_SAFE]| {'aza'}
    PERM[_fam][SR.PERIM]    = PERM[_fam][SR.PERIM]    | {'aza'}
    PERM[_fam][SR.MODERATE] = PERM[_fam][SR.MODERATE] | {'aza'}

def _allowed(scaff,role,et): return et in PERM.get(scaff,{}).get(role,set())

# ═══════════════════════════════════════════════════════════════════
#  FP HELPERS
# ═══════════════════════════════════════════════════════════════════
def _mfp(mol): return AllChem.GetMorganFingerprintAsBitVect(mol,FP_RADIUS,nBits=FP_NBITS)
def _cfp(mol,core):
    """Core-focused Morgan FP using fromAtoms on the INTACT molecule.
    This avoids kekulization failures from subgraph extraction."""
    if not core or len(core)<4:
        return AllChem.GetMorganFingerprintAsBitVect(mol,FP_RADIUS,nBits=FP_NBITS), False
    # fromAtoms computes FP rooted at core atoms only — no molecule editing needed
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, FP_RADIUS, nBits=FP_NBITS, fromAtoms=list(core))
    return fp, True
def _maxtc(fp,fps): return max((DataStructs.TanimotoSimilarity(fp,r) for r in fps),default=0.0)

class FPCache:
    def __init__(self,smiles,label=""):
        self.wfps=[]; self.cfps=[]; self.cval=[]; self.inchis=set()
        self.topo_sigs=[]  # real seed-family topology signatures
        self.core_topos=[]  # core-topology descriptors per seed
        for s in smiles:
            m=MolFromSmiles(s)
            if m is None: continue
            self.wfps.append(_mfp(m)); c=_get_core(m)
            cf,v=_cfp(m,c); self.cfps.append(cf); self.cval.append(v)
            try:
                i=MolToInchi(m)
                if i: self.inchis.add(i)
            except Exception: pass
            # Build real topology signature for this seed
            sig=_build_seed_topo_sig(m,c)
            self.topo_sigs.append(sig)
            rl=_label_sites(m,c)
            self.core_topos.append(core_topo_desc(m,c,rl))
        self.inchis=frozenset(self.inchis)
        # Seed-family topology clusters: group by scaffold family + D/A state
        self.topo_clusters=defaultdict(list)  # cluster_id → list of indices
        for idx,sig in enumerate(self.topo_sigs):
            # Cluster key = first two fields (family|DA counts)
            parts=sig.split("|")
            ckey="|".join(parts[:2]) if len(parts)>=2 else sig
            self.topo_clusters[ckey].append(idx)
        n_real=sum(1 for s in self.topo_sigs if s!="none")
        log.info("  FPCache(%s): %d mol, %d valid core, %d topo sigs, %d clusters",
                 label,len(self.wfps),sum(self.cval),n_real,len(self.topo_clusters))

def _build_seed_topo_sig(mol, core):
    """
    Build a real topology signature for a seed molecule encoding:
    - scaffold family
    - heteroatom types in core (sorted)
    - symmetry index bucket
    - D/A count state
    """
    if not core or len(core)<4: return "none"
    scaff=_classify(mol,core)
    # Heteroatom placement: sorted list of (atomic_num, symmetry_rank) for core heteroatoms
    ranks=list(Chem.CanonicalRankAtoms(mol,breakTies=False))
    hetero_places=[]
    for idx in sorted(core):
        z=mol.GetAtomWithIdx(idx).GetAtomicNum()
        if z!=6:  # non-carbon core atom
            r=ranks[idx] if idx<len(ranks) else 0
            hetero_places.append(f"{z}@R{r}")
    hetero_str=",".join(sorted(hetero_places)) if hetero_places else "C_only"
    # D/A counts
    nd=sum(1 for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() in (7,8))
    na=sum(1 for i in core if mol.GetAtomWithIdx(i).GetAtomicNum()==5)
    # Symmetry index bucket
    core_ranks=[ranks[i] for i in core if i<len(ranks)]
    sym=1.0-len(set(core_ranks))/max(len(core_ranks),1) if core_ranks else 0.0
    sym_bucket="high" if sym>0.3 else ("med" if sym>0.1 else "low")
    return f"{scaff}|D{nd}A{na}|sym_{sym_bucket}|{hetero_str}"

def core_topo_distance(ctd_a, ctd_b):
    """Normalised distance between two core-topology descriptors."""
    d=0.0
    d+=abs(ctd_a.get("n_donor",0)-ctd_b.get("n_donor",0))/max(ctd_a.get("n_donor",1),ctd_b.get("n_donor",1),1)
    d+=abs(ctd_a.get("n_acceptor",0)-ctd_b.get("n_acceptor",0))/max(ctd_a.get("n_acceptor",1),ctd_b.get("n_acceptor",1),1)
    d+=abs(ctd_a.get("core_size",0)-ctd_b.get("core_size",0))/max(ctd_a.get("core_size",10),ctd_b.get("core_size",10),1)
    d+=abs(ctd_a.get("sym_index",0)-ctd_b.get("sym_index",0))
    return min(d/4.0, 1.0)

def _nearest_cluster(edit_sig, sc):
    """Find nearest seed-family topology cluster and distance."""
    if not sc.topo_clusters: return "none", 1.0
    best_cid="none"; best_dist=1.0
    for cid, idxs in sc.topo_clusters.items():
        for idx in idxs:
            d=subst_topo_novelty(edit_sig, sc.topo_sigs[idx])
            if d<best_dist: best_dist=d; best_cid=cid
    return best_cid, best_dist

def predict_benchmark(smiles_list, ensembles_dict, desc_names, imputer,
                      n_workers=4, labels_df=None):
    """
    Predict all targets for benchmark molecules.
    If labels_df is provided (with target columns), also compute metrics.
    Returns (predictions_df, metrics_dict).
    """
    log.info("  Predicting %d benchmark molecules ...", len(smiles_list))
    cands=[{"smiles":s} for s in smiles_list]
    X,sm,vc=comp_desc(cands, desc_names, n_workers)
    if len(sm)==0:
        log.warning("  No valid benchmark molecules!")
        return pd.DataFrame(), {}
    X=imputer.transform(X)
    # Wide-format predictions
    pred_rows=[]
    for i in range(len(sm)):
        row={"smiles":sm[i]}
        for lb,(ens,sc,_) in ensembles_dict.items():
            mu,sig=pred_unc(ens,sc,X[i:i+1])
            row[f"pred_{lb}"]=mu[0]; row[f"unc_{lb}"]=sig[0]
        pred_rows.append(row)
    pred_df=pd.DataFrame(pred_rows)
    # Compute metrics if labeled
    metrics={}
    if labels_df is not None and len(labels_df)>0:
        # Map labels_df column name → ensembles_dict key. v17 fix: the
        # prediction columns are keyed by ensembles_dict keys (e.g.
        # "OscStr"), but the labels file uses the user-facing column
        # name (e.g. "Oscillator Strengths"). v16 used the same string
        # for both sides → KeyError on `pred_Oscillator Strengths`.
        target_map={"DeltaEST":"DeltaEST","T2-T1":"T2-T1",
                     "T1-S1(SOC)":"T1-S1(SOC)","T2-S1(SOC)":"T2-S1(SOC)",
                     "Oscillator Strengths":"OscStr","Singlets":"Singlets"}
        # Canonicalise both sides before merging so user-supplied
        # SMILES (e.g. C1=CC=CC=C1) reliably matches our internal
        # canonical form (c1ccccc1).
        def _canon(s):
            try:
                m=MolFromSmiles(str(s))
                return MolToSmiles(m) if m else None
            except Exception:
                return None
        smi_col=labels_df.iloc[:,1 if labels_df.shape[1]>1 else 0].astype(str)
        lbl_keys=[_canon(s) for s in smi_col]
        merged=pred_df.copy()
        merged_can=merged["smiles"].map(_canon)
        for tcol, ekey in target_map.items():
            pcol=f"pred_{ekey}"; ucol=f"unc_{ekey}"
            if tcol in labels_df.columns and pcol in merged.columns:
                vals=pd.to_numeric(labels_df[tcol], errors="coerce")
                lbl_map={k:v for k,v in zip(lbl_keys, vals) if k}
                truth_col=f"true_{ekey}"
                merged[truth_col]=merged_can.map(lbl_map)
                valid=merged.dropna(subset=[pcol,truth_col])
                if len(valid)>=3:
                    yt=valid[truth_col].values; yp=valid[pcol].values
                    from sklearn.metrics import mean_squared_error
                    metrics[tcol]={"n":int(len(valid)),
                        "MAE":float(mean_absolute_error(yt,yp)),
                        "RMSE":float(mean_squared_error(yt,yp)**0.5),
                        "R2":float(r2_score(yt,yp)),
                        "mean_unc":float(valid[ucol].mean())}
                    log.info("    %s: MAE=%.4f RMSE=%.4f R²=%.4f (n=%d)",
                             tcol,metrics[tcol]["MAE"],metrics[tcol]["RMSE"],
                             metrics[tcol]["R2"],len(valid))
    return pred_df, metrics

# ═══════════════════════════════════════════════════════════════════
#  PART 3: TRUE SUBSTITUTION-TOPOLOGY NOVELTY
# ═══════════════════════════════════════════════════════════════════
def _subst_topo_vec(edit_sig):
    """Parse edit_sig 'R3:F|R7:aza' into occupancy vector dict {rank: edit_type}."""
    if not edit_sig or edit_sig=="none": return {}
    d={}
    for part in edit_sig.split("|"):
        if ":" in part:
            r,t=part.split(":",1); d[r]=t
    return d

def subst_topo_novelty(sig_a, sig_b):
    """Hamming-like distance between two substitution-topology vectors."""
    va=_subst_topo_vec(sig_a); vb=_subst_topo_vec(sig_b)
    all_keys=set(va)|set(vb)
    if not all_keys: return 0.0
    diff=sum(1 for k in all_keys if va.get(k,"none")!=vb.get(k,"none"))
    return diff/max(len(all_keys),1)

def sym_pattern_sig(edit_sig):
    """Whether edits are symmetry-preserving: count distinct vs total edit ranks."""
    v=_subst_topo_vec(edit_sig)
    if not v: return "no_edit"
    types=list(v.values())
    if len(set(types))==1 and len(types)>1: return "sym_preserving"
    if len(set(types))==len(types): return "sym_breaking"
    return "mixed"

# ═══════════════════════════════════════════════════════════════════
#  PART 8: CORE-TOPOLOGY DESCRIPTOR
# ═══════════════════════════════════════════════════════════════════
def core_topo_desc(mol, core, roles):
    """Lightweight MR-specific core-topology descriptor (v14).

    Beyond the v13 fields it now reports:
      n_boron_in_core    : count of B atoms in core (BNB / multi-B detect)
      is_rim_extended    : True iff n_boron_in_core ≥ 3 (ν-DABNA-like)
      n_Cz_pendants      : aryl-N(C)(C) tertiary nitrogens — Cz/DPA-like
      n_DBhet_pendants   : dibenzofuran/thiophene/selenophene pendant count
      heavy_atom_zsum    : sum of (Z⁴) over heavy heteroatoms (Z>14) — proxy
                           for total heavy-atom-effect SOC budget.
    """
    n_donor=sum(1 for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() in (7,8,16,34,52))
    n_acc=sum(1 for i in core if mol.GetAtomWithIdx(i).GetAtomicNum()==5)
    cp=Chem.MolFromSmarts("[#6X3](=[OX1])")
    if mol.HasSubstructMatch(cp):
        for m in mol.GetSubstructMatches(cp):
            if m[0] in core: n_acc+=1
    n_boron=sum(1 for i in core if mol.GetAtomWithIdx(i).GetAtomicNum()==5)
    rh=Counter(roles.get(i,SR.FORBIDDEN) for i in core)
    ranks=list(Chem.CanonicalRankAtoms(mol,breakTies=False))
    core_ranks=[ranks[i] for i in core if i<len(ranks)]
    sym_idx=1.0-len(set(core_ranks))/max(len(core_ranks),1) if core_ranks else 0.0
    # v14: aux-donor and DB-heterocycle pendant counts. Use module-level
    # SMARTS handles for speed.
    n_cz=len(mol.GetSubstructMatches(_CZ_PAT)) if _CZ_PAT is not None else 0
    n_dbhet=len(mol.GetSubstructMatches(_DBHET_PAT)) if _DBHET_PAT is not None else 0
    # v14: heavy-atom Z⁴ sum (proxy for SOC budget). Uses Z⁴ scaling per
    # Pyykkö–Atsumi heavy-atom-effect approximation, restricted to Z>14
    # (P, S, Se, Te, etc.) and divided by 1e6 to keep magnitudes reasonable.
    heavy_z4=0.0
    for a in mol.GetAtoms():
        z=a.GetAtomicNum()
        if z>14: heavy_z4 += (z**4)/1e6
    return {"n_donor":n_donor,"n_acceptor":n_acc,"core_size":len(core),
            "sym_index":sym_idx,"n_perim_expand":rh.get(SR.PERIM,0),
            "n_overlap":rh.get(SR.OVERLAP,0),
            "n_boron_in_core":n_boron,
            "is_rim_extended":bool(n_boron>=3),
            "n_Cz_pendants":n_cz,
            "n_DBhet_pendants":n_dbhet,
            "heavy_atom_z4sum":float(heavy_z4)}

# ═══════════════════════════════════════════════════════════════════
#  BAND-PASS NOVELTY (Part 4: family-conditioned)
# ═══════════════════════════════════════════════════════════════════
def _gband(x,c,s): return float(np.exp(-0.5*((x-c)/max(s,1e-6))**2))

def compute_novelty(mol_fp,core_fp,cfv,inchi,tc,sc,pfp,pcfp,edit_sig,scaff,
                    seen_sigs=None, cand_ctd=None, parent_ctd=None):
    """
    Band-pass novelty with:
    - family-conditioned FP bands
    - parent-relative substitution-topology (Hamming)
    - REAL seed-family substitution-topology distance (from topo_sigs)
    - core-topology novelty (separate channel)
    - seen-set diversity
    """
    nid=0.0 if (inchi and inchi in tc.inchis) else 1.0
    mtw=_maxtc(mol_fp,tc.wfps); mtc=_maxtc(core_fp,tc.cfps) if cfv else mtw
    msw=_maxtc(mol_fp,sc.wfps) if sc.wfps else mtw
    msc=_maxtc(core_fp,sc.cfps) if (cfv and sc.cfps) else msw
    pw=DataStructs.TanimotoSimilarity(mol_fp,pfp) if pfp else msw
    pc=DataStructs.TanimotoSimilarity(core_fp,pcfp) if (pcfp and cfv) else pw
    pc_ctr=BAND_PARENT_CORE_CTR
    if scaff==SF.BN: pc_ctr=0.85
    elif scaff==SF.CO: pc_ctr=0.80
    ntwb=_gband(1.0-mtw,BAND_TRAIN_WHOLE_CTR,BAND_TRAIN_WHOLE_SIG)
    ntcb=_gband(1.0-mtc,0.30,0.15)
    npcb=_gband(pc,pc_ctr,BAND_PARENT_CORE_SIG)

    # ── CHANNEL A: Substitution-topology novelty (Hamming-based) ──
    # A1: distance to parent (unedited baseline = "none")
    st_dist_parent=subst_topo_novelty(edit_sig, "none")
    # A2: distance to nearest REAL seed-family topo signature
    st_dist_seed=st_dist_parent  # fallback
    if sc.topo_sigs:
        dists=[subst_topo_novelty(edit_sig,s) for s in sc.topo_sigs if s]
        if dists: st_dist_seed=min(dists)
    # A3: distance to nearest already-accepted topology
    st_dist_seen=st_dist_parent
    if seen_sigs:
        dists=[subst_topo_novelty(edit_sig,s) for s in seen_sigs[-200:]]
        if dists: st_dist_seen=min(dists)
    # Combined subst-topo novelty
    st_nov=(0.4*st_dist_parent + 0.35*st_dist_seed + 0.25*st_dist_seen)

    # ── CHANNEL B: Core-topology novelty (descriptor-based) ──
    ct_dist_parent=0.0
    if cand_ctd and parent_ctd:
        ct_dist_parent=core_topo_distance(cand_ctd,parent_ctd)
    ct_dist_seed=0.0
    if cand_ctd and sc.core_topos:
        dists=[core_topo_distance(cand_ctd,sctd) for sctd in sc.core_topos if sctd]
        if dists: ct_dist_seed=min(dists)
    # Bounded: reward small family-consistent changes, penalise large drift
    ct_nov=_gband(ct_dist_parent, 0.15, 0.10)

    # Symmetry pattern + edit depth
    sp_sig=sym_pattern_sig(edit_sig)
    sp_nov={"sym_preserving":0.3,"sym_breaking":0.7,"mixed":0.5,"no_edit":0.0}.get(sp_sig,0.0)
    v=_subst_topo_vec(edit_sig)
    ed_nov=min(1.0, len(v)*0.3) if v else 0.0

    # ── Combined novelty (both channels) ──
    combined=(0.10*nid+0.15*ntwb+0.08*ntcb+0.12*npcb+
              0.18*st_nov+0.10*ct_nov+0.10*sp_nov+0.07*ed_nov+
              0.05*min(1.0,(1.0-pw)*3)+0.05*min(ct_dist_seed*2,1.0))
    # Seed-family cluster assignment
    clust_id, clust_dist = _nearest_cluster(edit_sig, sc)
    return {"novelty_identity":nid,"novelty_train_whole":ntwb,"novelty_train_core":ntcb,
            "novelty_parent_core":npcb,
            # Explainable channels (Part 2)
            "novelty_subst_parent":st_dist_parent,
            "novelty_subst_family":st_dist_seed,
            "novelty_subst_shortlist":st_dist_seen,
            "novelty_core_parent":ct_dist_parent,
            "novelty_core_family":ct_dist_seed,
            "novelty_symmetry":sp_nov,
            "novelty_edit_depth":ed_nov,
            "subst_topo_nov":st_nov,"core_topo_nov":ct_nov,
            "sym_pattern_nov":sp_nov,"novelty_score":combined,
            "nearest_train_tanimoto_whole":mtw,"nearest_train_tanimoto_core":mtc,
            "nearest_seed_tanimoto_whole":msw,"nearest_seed_tanimoto_core":msc,
            "parent_tanimoto_whole":pw,"parent_tanimoto_core":pc,
            "subst_topo_sig":edit_sig,"sym_pattern_sig":sp_sig,
            "subst_topo_dist_parent":st_dist_parent,
            "subst_topo_dist_seed_family":st_dist_seed,
            "subst_topo_dist_shortlist":st_dist_seen,
            "core_topo_dist_parent":ct_dist_parent,
            "core_topo_dist_seed_family":ct_dist_seed,
            "seed_family_cluster_id":clust_id,
            "seed_family_cluster_distance":clust_dist}

# ═══════════════════════════════════════════════════════════════════
#  TRUST REGIONS (Part E, family-conditioned)
# ═══════════════════════════════════════════════════════════════════
def _trust_parent(pw,pc,scaff,cfv):
    """Family-conditioned parent trust (v14).

    BN-MR is strictest (0.75); BO-MR 0.72; CO-MR 0.70; NO-MR 0.70.
    Heavy-atom enhancers (SE, TE) and phosphine chalcogenides (PO, PS,
    PSE) are more permissive (0.60–0.65) because (a) the HOMO/LUMO
    pattern is dominated by the heavy-atom local lone-pair / σ*-PO
    motif which is *not* destroyed by peripheral edits, and (b) the
    SOC budget is so large that small electronic perturbations leave
    k_RISC essentially unchanged.
    """
    mc=MIN_CORE_SIM_PARENT
    if scaff==SF.BN:           mc=0.75
    elif scaff==SF.BO:         mc=0.72
    elif scaff==SF.CO:         mc=0.70
    elif scaff==SF.NO:         mc=0.70
    elif scaff==SF.SE:         mc=0.65
    elif scaff==SF.TE:         mc=0.60
    elif scaff in (SF.PO,SF.PS,SF.PSE): mc=0.65
    if not cfv: return True,0.6
    if pc<mc: return False,0.0
    if pw>MAX_WHOLE_SIM_PARENT: return False,0.0
    p=1.0
    if pw>0.98: p*=0.3
    elif pw>0.95: p*=0.6
    return True,p
def _trust_train(tw,tc,cfv):
    if tc<MIN_CORE_SIM_TRAIN and cfv: return False,0.0
    p=1.0
    if tw>MAX_WHOLE_SIM_TRAIN: p*=0.3
    if cfv and tc<0.70: p*=0.7
    return True,p

# ═══════════════════════════════════════════════════════════════════
#  PART 6: IMPROVED PAIRWISE EDIT COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════
def _on_same_ring(mol,i,j):
    ri=mol.GetRingInfo()
    for r in ri.AtomRings():
        if i in r and j in r: return True
    return False

def _path_through_frontier(mol,i,j,roles):
    """Check if shortest path between i,j traverses HOMO/LUMO/OVERLAP atoms."""
    path=Chem.rdmolops.GetShortestPath(mol,i,j)
    for k in path[1:-1]:
        if roles.get(k,SR.FORBIDDEN) in (SR.HOMO,SR.LUMO,SR.OVERLAP):
            return True
    return False

def edits_ok(mol,edits,roles,scaff):
    """Scaffold-conditional pairwise edit compatibility.

    Edit-tag conventions:
      'aza:<src>><dst>'   — single aromatic-CH → heteroatom substitution
                            (e.g. 'aza:C>N', 'aza:S>Se'). v17 (#48): the
                            directional suffix is tracked so the novelty
                            signature distinguishes substitution direction
                            (C↔N, S↔Se↔Te) instead of collapsing all atom
                            swaps into one tag. `edits_ok` only cares
                            about the category prefix (extracted with
                            `_edit_cat`).
      'diaza:<src>>N'     — symmetry-paired aza substitution; the pair was
                            pre-vetted by `_SC.dz_p` (built only from
                            canonical-rank symmetry groups), so the
                            path-through-frontier check is bypassed for
                            diaza-vs-diaza pairs (the symmetric
                            perturbation cancels at the frontier by
                            construction; without this relaxation the
                            diaza branch is dead code — verified in v17
                            review).
    """
    if len(edits)<=1: return True
    # BN-MR: strictest — reject any multi-edit touching frontier-sensitive sites
    min_dist = 3 if scaff==SF.BN else 2
    for i in range(len(edits)):
        for j in range(i+1,len(edits)):
            ai,ti=edits[i]; aj,tj=edits[j]
            # v17 (#48): collapse directional aza/diaza tags to their
            # category for compatibility checks.
            ci=_edit_cat(ti); cj=_edit_cat(tj)
            ri=roles.get(ai,SR.FORBIDDEN); rj=roles.get(aj,SR.FORBIDDEN)
            if ri==SR.OVERLAP or rj==SR.OVERLAP: return False
            d=_gdist(mol,ai,{aj})
            # BN-MR: reject any multi-edit involving HOMO/LUMO sites
            if scaff==SF.BN and (ri in (SR.HOMO,SR.LUMO) or rj in (SR.HOMO,SR.LUMO)):
                return False
            # Bulky near bulky
            if ci in ('tBu','CH3') and cj in ('tBu','CH3') and d<min_dist: return False
            # diaza+diaza: symmetry-paired, frontier-perturbation cancels.
            # Only the same-ring check is enforced.
            if ci=='diaza' and cj=='diaza':
                if _on_same_ring(mol,ai,aj): return False
                if scaff==SF.CO and d<4: return False
                continue
            # aza+aza: always check ring-sharing and frontier path
            if ci=='aza' and cj=='aza':
                if _on_same_ring(mol,ai,aj): return False
                if _path_through_frontier(mol,ai,aj,roles): return False
                # CARBONYL-MR: extra restriction on aza pairs
                if scaff==SF.CO and d<4: return False
            # F+aza on sensitive segment
            if ('aza' in (ci,cj)) and ('F' in (ci,cj)):
                if _path_through_frontier(mol,ai,aj,roles): return False
            # General minimum distance
            if d<min_dist and ci!=cj: return False
    return True

# ═══════════════════════════════════════════════════════════════════
#  HARD FILTER + MR QUALITY (preserved)
# ═══════════════════════════════════════════════════════════════════
def _hard_filter(mol,aa,sascore_max=None,enable_sa=True,enable_retro=True,
                 enable_charge_radical=True):
    """Hard MR-TADF filter (v16: charge/radical/SA/retrosynthesis gates).

    Accept if at least one MR-defining motif is present:
      - aromatic boron (BN/BO-MR), OR
      - aromatic carbonyl (CO-MR), OR
      - aromatic Se (SE-MR), OR
      - aromatic Te (TE-MR), OR
      - tetracoordinate P=X with X∈{O,S,Se} (P-chalcogenide MR).
    AND the molecule has at least one N/O/S/Se/Te heteroatom anywhere
    (donor pool), and meets ring/heavy-atom/rotatable-bond bounds.

    v16 additions (in evaluation order, cheapest first):
      - charge/radical/open-shell gate (atom & molecule);
      - retrosynthesis-feasibility heuristic (SMARTS + macrocycle +
        excessive-heteroatom-diversity rejection);
      - SAscore gate (Ertl-Schuffenhauer fragment-frequency complexity).

    Rotatable-bond cap is relaxed for molecules carrying a grafted
    auxiliary donor (Cz, DPA, etc.), which inherently add 1–2 rotatable
    bonds at the donor–core junction.
    """
    if mol is None: return False
    for a in mol.GetAtoms():
        if a.GetAtomicNum() not in aa: return False
    # ── v16 charge / radical / open-shell gate ────────────────────────
    if enable_charge_radical and _has_charge_or_radical(mol): return False
    has_mr_motif=(mol.HasSubstructMatch(_BP) or mol.HasSubstructMatch(_CP)
                  or mol.HasSubstructMatch(_SeP) or mol.HasSubstructMatch(_TeP)
                  or mol.HasSubstructMatch(_PXP))
    if not has_mr_motif: return False
    if not mol.HasSubstructMatch(_HP): return False
    if rdMolDescriptors.CalcNumAromaticRings(mol)<MIN_AROMATIC_RINGS: return False
    nh=mol.GetNumHeavyAtoms()
    if nh<HEAVY_ATOM_RANGE[0] or nh>HEAVY_ATOM_RANGE[1]: return False
    # Donor-graft aware rotatable-bond cap: grafted Cz/DPA contributes one
    # rotatable C–N bond. We accept up to MAX_ROTATABLE_BONDS + n_graft.
    nrot=Descriptors.NumRotatableBonds(mol)
    n_aux=len(mol.GetSubstructMatches(_CZ_PAT)) if _CZ_PAT else 0
    if nrot>MAX_ROTATABLE_BONDS+min(n_aux,3): return False
    c=_get_core(mol)
    if len(c)<6: return False
    # ── v16 retrosynthesis feasibility ────────────────────────────────
    if enable_retro and _retro_unfeasible(mol): return False
    # ── v16 SAscore (last because it is the most expensive) ──────────
    if enable_sa and _GLOBAL_SASCORER._fitted:
        cap=sascore_max if sascore_max is not None else SASCORE_MAX
        try:
            if _GLOBAL_SASCORER.score(mol)>cap: return False
        except Exception:
            return False
    return True

def _mr_quality(mol):
    """Heuristic structural quality score for MR-TADF candidates.

    Rewards:
      - large fused π-core (≥12 atoms ideal)
      - donor + acceptor co-presence (HOMO/LUMO complementarity)
      - planar (low sp3 fraction), rigid (no bi-aryl single bond in core)
      - canonical-rank symmetry (rewards symmetric MR cores)

    Note: Se's heavy-atom enhancement of SOC is *not* rewarded structurally
    here — it propagates physically through the predicted SOC values into
    the TADF figure-of-merit. Including it twice would double-count.
    """
    s=1.0; core=_get_core(mol)
    if len(core)<8: s*=0.3
    elif len(core)<12: s*=0.7
    hd=any(mol.GetAtomWithIdx(i).GetAtomicNum() in (7,8,16,34) for i in core)
    ha=any(mol.GetAtomWithIdx(i).GetAtomicNum()==5 for i in core)
    if not ha:
        cp=Chem.MolFromSmarts("[#6X3](=[OX1])")
        if mol.HasSubstructMatch(cp):
            for m in mol.GetSubstructMatches(cp):
                if m[0] in core: ha=True; break
    # If still no acceptor, treat aromatic Se in core as a weak acceptor
    # surrogate (Se 4p* low-lying LUMO contribution).
    if not ha and any(mol.GetAtomWithIdx(i).GetAtomicNum()==34 for i in core):
        ha=True
    if not (hd and ha): s*=0.5
    f=rdMolDescriptors.CalcFractionCSP3(mol)
    if f>0.35: s*=0.6
    elif f>0.25: s*=0.85
    for b in mol.GetBonds():
        a1,a2=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        if a1 in core and a2 in core and not b.IsInRing() and b.GetBondTypeAsDouble()==1.0:
            s*=0.4; break
    rk=list(Chem.CanonicalRankAtoms(mol,breakTies=False))
    nu=len(set(rk)); sr=1.0-nu/max(len(rk),1); s*=(0.8+0.2*sr)
    return max(0.01,min(1.0,s))

# AD scorer
class ADScorer:
    """Applicability-domain scorer (descriptor + fingerprint + uncertainty).

    v17 bug fix: the median nearest-neighbour distance is now computed
    against the SECOND-closest training point (excluding self), not the
    first. v16 fit `n_neighbors=1` and queried the training set against
    itself, so every distance was 0, `self.med = 1e-6` (the floor), and
    every candidate's normalised distance `nd = d/med` exploded to
    O(10⁶). The resulting `exp(-AD_ALPHA * nd)` factor was always 0,
    collapsing AD_score to 0 for every candidate and silently disabling
    the AD gate downstream (verified in v17 review test C-18).
    """
    def __init__(self,Xs,tc):
        n=len(Xs)
        # n_neighbors=2 so we can read the second-closest distance for
        # each training point (the first column is always self-distance=0).
        # When n_train < 2 we fall back to n_neighbors=1 — med is then a
        # sentinel and AD scores will be uncalibrated, but the pipeline
        # still runs without crashing.
        k_fit = 2 if n >= 2 else 1
        self.nn=NearestNeighbors(n_neighbors=k_fit,metric='euclidean')
        self.nn.fit(Xs)
        d,_=self.nn.kneighbors(Xs)
        if d.shape[1] >= 2:
            # d[:,1] is the distance to the true nearest non-self training pt.
            self.med=max(np.median(d[:,1]),1e-6)
        else:
            # 1-sample training: no meaningful median; use a sentinel so
            # AD scores degrade gracefully (rather than divide by ~0).
            self.med=1.0
        self.tc=tc
    def score(self,xr,wfp,cfp,cfv,unc):
        # Query mode: candidate is NOT in training, so n_neighbors=1 gives
        # the true nearest training point.
        d,_=self.nn.kneighbors(xr.reshape(1,-1), n_neighbors=1)
        nd=d[0,0]/self.med; fd=1.0-_maxtc(wfp,self.tc.wfps)
        cd=1.0-_maxtc(cfp,self.tc.cfps) if cfv else fd
        nu=unc/0.10
        return float(np.clip(np.exp(-AD_ALPHA*nd)*np.exp(-AD_BETA*fd)*np.exp(-AD_GAMMA*cd)*np.exp(-AD_DELTA*nu),0,1))

# ═══════════════════════════════════════════════════════════════════
#  v13 — UNIT CONVERSION & TADF FIGURE-OF-MERIT
# ═══════════════════════════════════════════════════════════════════
def to_eV(x, src_unit):
    """Convert energy from src_unit → eV (the internal unit)."""
    if src_unit not in _EV_FROM:
        raise ValueError(f"Unknown energy unit '{src_unit}'. Allowed: {list(_EV_FROM)}")
    return float(x)*_EV_FROM[src_unit]

def to_invcm(x, src_unit):
    """Convert SOC from src_unit → cm⁻¹ (the internal unit)."""
    if src_unit not in _SOC_FROM:
        raise ValueError(f"Unknown SOC unit '{src_unit}'. Allowed: {list(_SOC_FROM)}")
    return float(x)*_SOC_FROM[src_unit]

def tadf_figure_of_merit(dEST, soc1, soc2, fosc, dEST_unit="eV", soc_unit="cm-1"):
    """Physics-based scalar objective for MR-TADF inverse design.

    Marcus high-T expression for the TADF rate:
        k_TADF ∝ k_r · k_RISC / (k_r + k_nr)
        k_RISC ∝ |H_SOC|² / (ΔE_ST)²    (after activation prefactor cancellation
                                          when ΔE_ST is small relative to k_B T)
        k_r    ∝ f_OSC · ν³  ≈  f_OSC   (constant emission energy assumption)

    Hence:                  k_TADF ∝ f_OSC · |H_SOC|² / (ΔE_ST)²
    In log10 (numerically well-behaved):
        log10(FoM) = log10(fOSC) + 2·log10(SOC_eff/cm⁻¹) − 2·log10(ΔE_ST/eV)

    SOC_eff is taken as max(SOC(T1↔S1), SOC(T2↔S1)) — whichever dominates
    the upconversion channel. Returns log10(FoM); the caller can take 10**x
    if a linear scale is needed.

    Input contract (v17):
      All inputs are physically non-negative (ΔE_ST ≥ 0, SOC ≥ 0, fOSC ≥ 0).
      ΔE_ST = 0 is supported (the inverted-singlet design target).
      Negative inputs are clamped to 0 then floored — they should not
      occur in practice; if they do, the resulting FoM is finite (worst
      possible) so downstream ranking is well-defined but the upstream
      bug is detectable via the AD/trust gates.
      NaN inputs return -inf (the worst possible FoM) instead of NaN, so
      np.argsort / np.argmax remain meaningful.
    """
    # Coerce to ndarray once so scalar/array inputs share the same code
    # path and so np.where / np.isnan work uniformly.
    dEST_a = np.asarray(dEST, dtype=np.float64)
    soc1_a = np.asarray(soc1, dtype=np.float64)
    soc2_a = np.asarray(soc2, dtype=np.float64)
    fosc_a = np.asarray(fosc, dtype=np.float64)

    # Clamp negatives to 0 (per the non-negative contract) and apply
    # per-quantity floors in INTERNAL units. The dEST floor is 1 µeV so
    # inverted-singlet candidates (|dEST| ≪ 1 meV) keep ranking
    # discrimination at the design target.
    dEST_eV  = np.maximum(np.clip(dEST_a, 0.0, None) * _EV_FROM[dEST_unit],
                          _FOM_FLOOR_DEST)
    soc1_cm  = np.clip(soc1_a, 0.0, None) * _SOC_FROM[soc_unit]
    soc2_cm  = np.clip(soc2_a, 0.0, None) * _SOC_FROM[soc_unit]
    soc_eff  = np.maximum(np.maximum(soc1_cm, soc2_cm), _FOM_FLOOR_SOC)
    fosc_cl  = np.maximum(np.clip(fosc_a, 0.0, None),    _FOM_FLOOR_FOSC)

    fom = (np.log10(fosc_cl)
           + 2.0*np.log10(soc_eff)
           - 2.0*np.log10(dEST_eV))

    # Replace NaN FoM (from any NaN input) with -inf so downstream sort
    # / argmax behave sensibly (rank as worst possible candidate).
    nan_mask = (np.isnan(dEST_a) | np.isnan(soc1_a)
                | np.isnan(soc2_a) | np.isnan(fosc_a))
    if nan_mask.any() if hasattr(nan_mask, "any") else nan_mask:
        fom = np.where(nan_mask, -np.inf, fom)
    return fom

# ═══════════════════════════════════════════════════════════════════
#  v13 — CONFORMAL σ CALIBRATION
# ═══════════════════════════════════════════════════════════════════
def calibrate_sigma(ens, sc, X_cal, y_cal, target_coverage=0.683):
    """Hold-out conformal scale factor κ for ensemble σ.

    Solves κ = quantile_q( |y - μ| / σ ), q = target_coverage. With κ applied
    as σ_calib = κ·σ, the implied predictive band has empirical coverage
    ≈ target_coverage on the held-out set. This makes the Gaussian factor
    Φ((c - μ)/σ_calib) in constraint-EI an honest probability (assuming
    rough Gaussianity of residuals).
    """
    Xs = sc.transform(X_cal)
    p = np.empty((len(ens), Xs.shape[0]), dtype=np.float64)
    for i, m in enumerate(ens): p[i] = m.predict(Xs)
    mu = p.mean(0); sig = np.maximum(p.std(0), 1e-8)
    z = np.abs(y_cal - mu) / sig
    if len(z) < 5:
        return 1.0
    return float(np.quantile(z, target_coverage))

def pred_unc_calibrated(ens, sc, X, kappa=1.0):
    """Ensemble prediction with conformal-scaled σ."""
    Xs = sc.transform(X)
    p = np.empty((len(ens), Xs.shape[0]), dtype=np.float32)
    for i, m in enumerate(ens): p[i] = m.predict(Xs)
    return p.mean(0).astype(np.float64), (kappa*p.std(0)).astype(np.float64)

# ═══════════════════════════════════════════════════════════════════
#  v13 — SCAFFOLD-STRATIFIED CV  &  UNCERTAINTY-ERROR CORRELATION
# ═══════════════════════════════════════════════════════════════════
def scaffold_groups(smiles_list):
    """Assign each training SMILES to a scaffold-family group label."""
    groups=[]
    for s in smiles_list:
        m=MolFromSmiles(s) if isinstance(s,str) else None
        groups.append(_classify(m) if m is not None else SF.OT)
    return groups

def scaffold_stratified_cv(X, y, smiles_list, label="", n_splits=5, n_estimators=300):
    """Group-K-fold CV grouped by scaffold family.

    This evaluates surrogate transferability *across* MR families — a far more
    honest test than random KFold for inverse design where novel families
    are precisely the goal. Returns dict of fold metrics keyed by held-out
    family, plus the macro-MAE across families.
    """
    groups = scaffold_groups(smiles_list)
    unique = sorted(set(groups))
    out = {"per_family": {}, "macro_MAE": None}
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    bp = dict(n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
              min_child_weight=3, tree_method="hist", random_state=SEED, n_jobs=-1)
    fam_maes=[]
    for fam in unique:
        train_idx = [i for i,g in enumerate(groups) if g != fam]
        test_idx  = [i for i,g in enumerate(groups) if g == fam]
        if len(test_idx) < 2 or len(train_idx) < 5:
            continue
        m = xgb.XGBRegressor(**bp); m.fit(Xs[train_idx], y[train_idx])
        yp = m.predict(Xs[test_idx])
        mae = float(mean_absolute_error(y[test_idx], yp))
        out["per_family"][fam] = {"n_test": int(len(test_idx)), "MAE": mae}
        fam_maes.append(mae)
        log.info("    [%s] held-out %s (n=%d): MAE=%.4f", label, fam, len(test_idx), mae)
    out["macro_MAE"] = float(np.mean(fam_maes)) if fam_maes else None
    return out

def uncertainty_error_correlation(mu, sig, y_true, label=""):
    """Spearman ρ between predicted σ and absolute error.

    A well-calibrated surrogate has ρ > 0 (large σ → large |error|).
    ρ ≈ 0 means σ is uninformative — a known failure mode of bagged-tree
    ensembles. Returns dict with rho, p, decile-binned MAE table.
    """
    err = np.abs(np.asarray(mu) - np.asarray(y_true))
    sig = np.asarray(sig)
    if len(err) < 5: return {"spearman_rho": None, "p_value": None}
    rho, p = spearmanr(sig, err)
    bins = np.quantile(sig, np.linspace(0, 1, 11))
    bin_idx = np.clip(np.digitize(sig, bins) - 1, 0, 9)
    decile_mae = []
    for b in range(10):
        mask = bin_idx == b
        decile_mae.append(float(err[mask].mean()) if mask.any() else None)
    if label: log.info("    [%s] σ–|err| Spearman ρ=%.3f (p=%.3g)", label, rho, p)
    return {"spearman_rho": float(rho), "p_value": float(p),
            "decile_sigma": [float(x) for x in bins.tolist()],
            "decile_MAE": decile_mae}

# ═══════════════════════════════════════════════════════════════════
#  v13 — OPTIONAL GP SURROGATE
# ═══════════════════════════════════════════════════════════════════
class GPEnsemble:
    """Single Gaussian-process surrogate exposing the same (mean, std) API
    as the bagged-XGBoost ensemble. Used when --use_gpr is supplied.

    For modest training sizes (< ~2000 mol) and high-dimensional descriptor
    spaces, a Matérn-5/2 + WhiteKernel GP gives well-calibrated posterior
    σ — an honest replacement for bootstrap σ in the constraint-EI factors.
    """
    def __init__(self, length_scale=1.0, noise_level=1e-2):
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                  Matern(length_scale=length_scale, length_scale_bounds=(1e-2,1e2), nu=2.5) +
                  WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6,1e1)))
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, alpha=1e-6,
            random_state=SEED)
    def fit(self, X, y):
        self.gpr.fit(X, y); return self
    def predict_with_std(self, X):
        mu, sig = self.gpr.predict(X, return_std=True)
        return mu.astype(np.float64), sig.astype(np.float64)

def train_gpr(X, y, lb=""):
    """Fit a single GP. Wrapped to mimic train_ens() return signature."""
    log.info("Train GPR %s (n=%d, p=%d)", lb, X.shape[0], X.shape[1])
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    gp = GPEnsemble().fit(Xs, y)
    return [gp], sc

def pred_unc_gpr(model_list, sc, X):
    """Match the (mean, std) signature used by the rest of the pipeline."""
    gp = model_list[0]
    Xs = sc.transform(X)
    return gp.predict_with_std(Xs)

# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def load_data(dp,tp):
    """Load descriptor & target tables, merge, and return arrays.

    Descriptor leakage guard (v13): any column whose name matches a target
    column (or a fuzzy variant) is *removed* from the descriptor matrix.
    This prevents the pipeline from "predicting" a target from itself when
    the descriptor file accidentally contains target columns.

    v17 fix: the leakage guard now runs BEFORE the merge. v16 ran the
    guard after merging, but if a descriptor column had the verbatim
    name of a target (e.g. "DeltaEST"), pandas auto-suffixed both sides
    to "DeltaEST_x" / "DeltaEST_y" and the subsequent `df["DeltaEST"]`
    access KeyErrored, crashing the loader entirely. The guard now
    drops leaky descriptor columns up front so the merge is collision-free.
    """
    dd=pd.read_excel(dp); dd.rename(columns={dd.columns[0]:"Name",dd.columns[1]:"SMILES"},inplace=True)
    dt=pd.read_excel(tp); dt.rename(columns={dt.columns[0]:"Name"},inplace=True)
    tc=["DeltaEST","T2-T1","T1-S1(SOC)","T2-S1(SOC)","Oscillator Strengths","Singlets"]
    # ── descriptor leakage guard (v17: runs BEFORE merge) ──────────
    blocklist=set()
    target_keywords={"deltaest","dest","s1t1","st_gap","t2-t1","t2_t1",
                     "t1-s1","t1_s1","t2-s1","t2_s1","soc","oscillator",
                     "fosc","singlet","s1_energy","triplet"}
    for col in dd.columns:
        cl=str(col).lower().replace(" ","").replace("(","").replace(")","")
        if col in tc or any(k in cl for k in target_keywords):
            blocklist.add(col)
    if blocklist:
        log.warning("  Descriptor leakage guard: blocking %d columns: %s",
                    len(blocklist), sorted(blocklist))
        dd=dd.drop(columns=list(blocklist))
    df=dd.merge(dt[["Name"]+tc],on="Name",how="inner")
    for c in tc: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=tc).reset_index(drop=True)
    log.info("Merged: %d mol",len(df))
    smi=df["SMILES"].tolist()
    dc=[c for c in dd.columns if c not in ("Name","SMILES")]
    X=df[dc].values.astype(np.float64)
    imp=SimpleImputer(strategy="median"); X=imp.fit_transform(X)
    return (df,X,df["DeltaEST"].values.astype(np.float64),
            df["T2-T1"].values.astype(np.float64),
            df["T1-S1(SOC)"].values.astype(np.float64),
            df["T2-S1(SOC)"].values.astype(np.float64),
            df["Oscillator Strengths"].values.astype(np.float64),
            df["Singlets"].values.astype(np.float64),dc,smi,imp)

def load_seed_smiles(args,smi_train):
    """Part 2: Load seed SMILES from file or fallback to training."""
    if args.seed_file:
        log.info("  Loading seeds from %s",args.seed_file)
        if args.seed_file.endswith(".xlsx"):
            sf=pd.read_excel(args.seed_file); seeds=sf.iloc[:,1 if sf.shape[1]>1 else 0].dropna().tolist()
        else:
            with open(args.seed_file) as f: seeds=[l.strip() for l in f if l.strip()]
        log.info("  Loaded %d seed SMILES",len(seeds))
        return seeds, True
    log.info("  No --seed_file provided; using training set as seeds")
    return smi_train, False

# ═══════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════
def _dgpu():
    try:
        import subprocess; r=subprocess.run(["nvidia-smi"],capture_output=True,timeout=5)
        if r.returncode!=0: return "cpu"
    except Exception: return "cpu"
    try:
        t=xgb.XGBRegressor(n_estimators=1,max_depth=1,device="cuda",tree_method="hist",verbosity=0)
        t.fit(np.array([[1.,2.],[3.,4.]]),np.array([0.,1.])); t.predict(np.array([[1.,2.]])); return "cuda"
    except Exception: return "cpu"
def _t1(a):
    # v16.1: tuple is (Xs, y, sw, bp, mi, sd) where sw is an optional
    # sample-weight vector or None. Backward-compatible — callers that
    # don't supply weights should pass sw=None.
    Xs,y,sw,bp,mi,sd=a; p=bp.copy(); p["random_state"]=sd+mi
    rng=np.random.RandomState(sd+mi); p["subsample"]=rng.uniform(0.7,0.9)
    p["colsample_bytree"]=rng.uniform(0.6,0.8)
    if p.get("device")=="cuda": p["n_jobs"]=1
    idx=rng.choice(len(Xs),len(Xs),replace=True)
    m=xgb.XGBRegressor(**p)
    if sw is not None:
        m.fit(Xs[idx],y[idx],sample_weight=np.asarray(sw)[idx])
    else:
        m.fit(Xs[idx],y[idx])
    return m
def train_ens(X,y,n=20,lb="",mt=None,sample_weight=None):
    """Train a bagged-XGB ensemble.

    v16.1: optional sample_weight vector (length len(X)) is forwarded
    to the underlying XGBRegressor.fit calls and applied per bootstrap.
    Used by the label-source-weighting pathway. None preserves legacy
    behaviour (uniform weights).

    v17: sample_weight may be any 1-D sequence (list, tuple, ndarray);
    it is normalised to ndarray once and length-checked up front so a
    mis-sized vector produces a clear ValueError instead of an opaque
    IndexError from inside a worker thread.
    """
    dev=_dgpu(); log.info("Train(%d) %s [%s]",n,lb,dev)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.ndim != 1 or sample_weight.shape[0] != len(X):
            raise ValueError(
                f"train_ens({lb}): sample_weight has shape "
                f"{sample_weight.shape}, expected ({len(X)},)")
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    bp=dict(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.7,
            reg_alpha=0.1,reg_lambda=1.0,min_child_weight=3,device=dev,tree_method="hist",
            random_state=SEED,n_jobs=-1)
    cv=cross_val_score(xgb.XGBRegressor(**bp),Xs,y,cv=5,scoring="neg_mean_absolute_error")
    log.info("  CV MAE:%.4f±%.4f",-cv.mean(),cv.std())
    if mt is None: mt=min(4 if dev=="cuda" else mp.cpu_count(),n)
    t0=time.time()
    with ThreadPoolExecutor(max_workers=mt) as ex:
        ens=list(ex.map(_t1,[(Xs,y,sample_weight,bp,i,SEED) for i in range(n)]))
    log.info("  Trained in %.1fs",time.time()-t0); return ens,sc
def pred_unc(ens,sc,X):
    Xs=sc.transform(X); p=np.empty((len(ens),Xs.shape[0]),dtype=np.float32)
    for i,m in enumerate(ens): p[i]=m.predict(Xs)
    return p.mean(0).astype(np.float64),p.std(0).astype(np.float64)
def _plot(yt,yp,lb,od):
    fig,ax=plt.subplots(figsize=(5.5,5))
    ax.scatter(yt,yp,s=22,alpha=0.6,edgecolors='k',linewidths=0.3,c='#2166ac',zorder=3)
    mn,mx=min(yt.min(),yp.min()),max(yt.max(),yp.max()); pad=(mx-mn)*0.08
    ax.plot([mn-pad,mx+pad],[mn-pad,mx+pad],'r--',lw=1.2,alpha=0.7)
    mae=mean_absolute_error(yt,yp); r2=r2_score(yt,yp)
    ax.set_xlabel(f"Actual {lb}"); ax.set_ylabel(f"Predicted {lb}")
    ax.set_title(f"{lb}\nMAE={mae:.4f}  R²={r2:.4f}")
    ax.set_xlim(mn-pad,mx+pad); ax.set_ylim(mn-pad,mx+pad)
    ax.set_aspect('equal'); ax.grid(True,alpha=0.2); fig.tight_layout()
    fn=os.path.join(od,f"regression_{lb.replace(' ','_').replace('(','').replace(')','')}.png")
    fig.savefig(fn,dpi=200); plt.close(fig)
    log.info("    Saved %s (MAE=%.4f R²=%.4f)",os.path.basename(fn),mae,r2)

# ═══════════════════════════════════════════════════════════════════
#  SEED CACHE + GENERATION
# ═══════════════════════════════════════════════════════════════════
class _SC:
    __slots__=('smi','mol','core','scaff','csz','roles','f_s','ch3_s','aza_s','tbu_s',
               'sg','pf_g','pc_g','dz_p','ann_e','ne','wfp','cfp','cfv',
               # v14 additions
               'graft_s','cn_s','q4_e','so2_s',
               # v15 additions
               'rim_e','p_e','n_boron_in_core')
    def __init__(self,smi,mol,exp,eb=False,ed=False):
        self.smi=smi; self.mol=mol; self.core=_get_core(mol); self.csz=len(self.core)
        self.scaff=_classify(mol,self.core); self.roles=_label_sites(mol,self.core,self.scaff)
        self.wfp=_mfp(mol); cf,v=_cfp(mol,self.core); self.cfp=cf; self.cfv=v
        subs=SUBSTITUTIONS_EXPLR if exp else SUBSTITUTIONS; sc=self.scaff
        self.f_s=[i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic()
                  and mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetTotalNumHs()>0
                  and _allowed(sc,self.roles.get(i,SR.FORBIDDEN),'F')]
        self.ch3_s=[i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic()
                    and mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetTotalNumHs()>0
                    and mol.GetAtomWithIdx(i).GetDegree()<=2
                    and _allowed(sc,self.roles.get(i,SR.FORBIDDEN),'CH3')]
        self.tbu_s=([i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic()
                     and mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetTotalNumHs()>0
                     and _allowed(sc,self.roles.get(i,SR.FORBIDDEN),'tBu')] if eb else [])
        self.aza_s=[]
        for atom in mol.GetAtoms():
            idx=atom.GetIdx(); z=atom.GetAtomicNum()
            if (z in subs and atom.GetIsAromatic() and idx in self.core and atom.GetTotalNumHs()>0
                    and _allowed(sc,self.roles.get(idx,SR.FORBIDDEN),'aza')):
                tot=sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
                for nz in subs[z]:
                    if tot<=_MAX_VALENCE.get(nz,0): self.aza_s.append((idx,nz))
        ranks=list(Chem.CanonicalRankAtoms(mol,breakTies=False))
        grp=defaultdict(list)
        for i,r in enumerate(ranks): grp[r].append(i)
        self.sg=[g for g in grp.values() if len(g)>=2]
        fs=set(self.f_s); cs=set(self.ch3_s); az=set(a[0] for a in self.aza_s)
        self.pf_g=[g for g in self.sg if sum(1 for i in g if i in fs)>=2]
        self.pc_g=[g for g in self.sg if sum(1 for i in g if i in cs)>=2]
        self.dz_p=([ag[:2] for g in self.sg for ag in [[i for i in g if i in az]] if len(ag)>=2] if ed else [])
        # v15 fix: ann_e enumerates fusable perimeter edges directly from
        # the core (bonded aryl-CH/CH pairs). The v12 PERIM-label check
        # was over-restrictive — for fused MR cores the perimeter atoms
        # often have a neighbour outside core but don't satisfy the
        # degree-2 + aromatic-PERIM-neighbour condition that the labeller
        # required for the SR.PERIM tag.
        self.ann_e=[]
        seen_e=set()
        for i in self.core:
            ai=mol.GetAtomWithIdx(i)
            if not (ai.GetIsAromatic() and ai.GetAtomicNum()==6
                    and ai.GetTotalNumHs()>0): continue
            for nb in ai.GetNeighbors():
                ni=nb.GetIdx()
                if ni not in self.core or ni<=i: continue
                an=mol.GetAtomWithIdx(ni)
                if not (an.GetIsAromatic() and an.GetAtomicNum()==6
                        and an.GetTotalNumHs()>0): continue
                e=(i,ni)
                if e not in seen_e:
                    self.ann_e.append(e); seen_e.add(e)
        # v14: graft sites — aryl-CH at NODE_SAFE or PERIM where the family
        # PERM matrix permits 'graft_D'. Restricted to degree ≤2 to avoid
        # over-substitution at ring junctions.
        self.graft_s=[i for i in range(mol.GetNumAtoms())
                      if mol.GetAtomWithIdx(i).GetIsAromatic()
                      and mol.GetAtomWithIdx(i).GetAtomicNum()==6
                      and mol.GetAtomWithIdx(i).GetTotalNumHs()>0
                      and mol.GetAtomWithIdx(i).GetDegree()<=2
                      and _allowed(sc,self.roles.get(i,SR.FORBIDDEN),'graft_D')]
        # v14: CN sites — same role permission as F but tighter (NODE_SAFE
        # is *not* a CN site; only MODERATE/PERIM where electron withdrawal
        # is desirable).
        self.cn_s=[i for i in range(mol.GetNumAtoms())
                   if mol.GetAtomWithIdx(i).GetIsAromatic()
                   and mol.GetAtomWithIdx(i).GetAtomicNum()==6
                   and mol.GetAtomWithIdx(i).GetTotalNumHs()>0
                   and _allowed(sc,self.roles.get(i,SR.FORBIDDEN),'CN')]
        # v14: q4 candidate edges — biaryl single bonds outside rings.
        # These appear when two aromatic systems are linked by a rotatable
        # single bond. q4 inserts a sp3 C(R)2 spacer to enforce planarity
        # at the molecular junction (DMAC-like).
        self.q4_e=[]
        for b in mol.GetBonds():
            if b.GetBondTypeAsDouble()==1.0 and not b.IsInRing():
                a1=mol.GetAtomWithIdx(b.GetBeginAtomIdx())
                a2=mol.GetAtomWithIdx(b.GetEndAtomIdx())
                if (a1.GetIsAromatic() and a2.GetIsAromatic()
                    and a1.GetAtomicNum()==6 and a2.GetAtomicNum()==6):
                    self.q4_e.append((a1.GetIdx(), a2.GetIdx()))
        # v14: sulfone-oxidation candidates — divalent aromatic S atoms
        # (thiophene-S, dibenzothiophene-S). Only one will be picked per edit.
        self.so2_s=[i for i in range(mol.GetNumAtoms())
                    if mol.GetAtomWithIdx(i).GetAtomicNum()==16
                    and mol.GetAtomWithIdx(i).GetIsAromatic()
                    and mol.GetAtomWithIdx(i).GetDegree()==2]
        # v16: ν-DABNA rim-extension candidate EDGES — perimeter (i, j)
        # pairs of bonded aromatic CH carbons in a B-containing core.
        # Both endpoints must have ≥1 H (consumed by ring fusion) and be
        # connected by an aromatic bond (the new ring's shared edge).
        # Only meaningful for cores that already contain B; otherwise we
        # would be inventing BN-MR character from a non-B parent. The
        # operator (`_ann_BN(rw, i, j)`) fuses a new 1,2-azaborine ring
        # across the (i, j) edge so the parent's central Kekulé pattern
        # remains undisturbed.
        self.rim_e = []   # list of (i, j) perimeter edges
        n_boron_in_core = sum(1 for i in self.core
                              if mol.GetAtomWithIdx(i).GetAtomicNum() == 5)
        self.n_boron_in_core = n_boron_in_core
        if n_boron_in_core >= 1 and self.scaff in (SF.BN, SF.BO, SF.OT):
            seen_re=set()
            for i in self.core:
                a = mol.GetAtomWithIdx(i)
                if not (a.GetIsAromatic() and a.GetAtomicNum() == 6
                        and a.GetTotalNumHs() > 0 and a.GetDegree() <= 2):
                    continue
                for nb in a.GetNeighbors():
                    j = nb.GetIdx()
                    if j not in self.core or j <= i: continue
                    aj = mol.GetAtomWithIdx(j)
                    if not (aj.GetIsAromatic() and aj.GetAtomicNum() == 6
                            and aj.GetTotalNumHs() > 0
                            and aj.GetDegree() <= 2):
                        continue
                    bij = mol.GetBondBetweenAtoms(i, j)
                    if bij is None: continue
                    if bij.GetBondType() != Chem.BondType.AROMATIC: continue
                    e=(i, j)
                    if e in seen_re: continue
                    self.rim_e.append(e); seen_re.add(e)
        # v15: P-chalcogenide insertion candidates — same biaryl single
        # bonds that q4 uses, but the resulting molecule must still satisfy
        # the hard filter (handled at edit time).
        self.p_e = list(self.q4_e)
        self.ne = (len(self.f_s)+len(self.ch3_s)+len(self.aza_s)+len(self.tbu_s)+
                   len(self.pf_g)+len(self.pc_g)+len(self.dz_p)
                   + len(self.graft_s) + len(self.cn_s) + len(self.q4_e) + len(self.so2_s)
                   + len(self.rim_e) + len(self.p_e))

def _safe(rw):
    try:
        Chem.SanitizeMol(rw); s=MolToSmiles(rw.GetMol())
        if not s: return None
        m=MolFromSmiles(s); return (s,m) if m else None
    except Exception: return None
def _af(rw,i):
    a=rw.GetAtomWithIdx(i)
    if a.GetTotalNumHs()<=0: return False
    fi=rw.AddAtom(Chem.Atom(9)); rw.AddBond(i,fi,Chem.BondType.SINGLE)
    if a.GetNoImplicit(): a.SetNumExplicitHs(max(0,a.GetNumExplicitHs()-1))
    return True
def _am(rw,i):
    a=rw.GetAtomWithIdx(i)
    if a.GetTotalNumHs()<=0: return False
    ci=rw.AddAtom(Chem.Atom(6)); rw.AddBond(i,ci,Chem.BondType.SINGLE)
    if a.GetNoImplicit(): a.SetNumExplicitHs(max(0,a.GetNumExplicitHs()-1))
    return True
def _at(rw,i):
    a=rw.GetAtomWithIdx(i)
    if a.GetTotalNumHs()<=0: return False
    c0=rw.AddAtom(Chem.Atom(6)); rw.AddBond(i,c0,Chem.BondType.SINGLE)
    for _ in range(3): ci=rw.AddAtom(Chem.Atom(6)); rw.AddBond(c0,ci,Chem.BondType.SINGLE)
    if a.GetNoImplicit(): a.SetNumExplicitHs(max(0,a.GetNumExplicitHs()-1))
    return True

def _ann(rw,i,j):
    """Real benzannulation across perimeter edge (i,j).

    Adds 4 new aromatic carbons to fuse a benzene ring on the (i,j) bond.
    Both i and j must be perimeter aromatic CH atoms (one H each, sharing
    a ring). After mutation, sanitisation re-aromatises the new ring.

    Returns True on success, False if the edit cannot be applied
    (insufficient H, invalid topology, sanitization failure handled upstream).
    """
    ai=rw.GetAtomWithIdx(i); aj=rw.GetAtomWithIdx(j)
    if ai.GetTotalNumHs()<=0 or aj.GetTotalNumHs()<=0: return False
    # Build the new ring with kekulé alternation (single/double) so RDKit
    # can perceive aromaticity on sanitise. Fused edge (i,j) is the existing
    # aromatic bond; the new path is i — c0 = c1 — c2 = c3 — j.
    c=[rw.AddAtom(Chem.Atom(6)) for _ in range(4)]
    rw.AddBond(i,    c[0], Chem.BondType.SINGLE)
    rw.AddBond(c[0], c[1], Chem.BondType.DOUBLE)
    rw.AddBond(c[1], c[2], Chem.BondType.SINGLE)
    rw.AddBond(c[2], c[3], Chem.BondType.DOUBLE)
    rw.AddBond(c[3], j,    Chem.BondType.SINGLE)
    # Consume the H atoms at i and j: they are now ring junctions.
    if ai.GetNoImplicit(): ai.SetNumExplicitHs(max(0,ai.GetNumExplicitHs()-1))
    if aj.GetNoImplicit(): aj.SetNumExplicitHs(max(0,aj.GetNumExplicitHs()-1))
    return True

# ═══════════════════════════════════════════════════════════════════
#  v14 — DONOR LIBRARY + GRAFT OPERATOR + CN + SULFONE
# ═══════════════════════════════════════════════════════════════════
# Donor library. Each entry is (label, SMILES with one [*:1] mark on the
# atom that will form the new bond, weight in random pick).
# Connection conventions:
#   - Cz / tCz / PXZ / PTZ: connect via the ring-N (lone-pair forms aryl–N).
#   - DPA: connect via the central N (NH → N–aryl).
#   - DBF / DBT / DBSe: connect via aryl-CH (biaryl coupling).
DONOR_LIB=[
    # label,  SMILES with [*:1] marker,                          relative weight
    ('Cz',    'c1ccc2c(c1)[nH:1]c1ccccc12',                       3.0),
    ('tCz',   'CC(C)(C)c1ccc2c(c1)[nH:1]c1cc(C(C)(C)C)ccc12',     2.0),
    ('DPA',   '[NH:1](c1ccccc1)c1ccccc1',                         2.5),
    ('DTPA',  '[NH:1](c1ccc(C(C)(C)C)cc1)c1ccc(C(C)(C)C)cc1',     1.0),
    ('PXZ',   'c1ccc2c(c1)[nH:1]c1ccccc1O2',                      0.6),
    ('PTZ',   'c1ccc2c(c1)[nH:1]c1ccccc1S2',                      0.6),
    ('DBF',   'c1ccc2c(c1)oc1[cH:1]cccc12',                       0.6),
    ('DBT',   'c1ccc2c(c1)sc1[cH:1]cccc12',                       0.6),
    ('DBSe',  'c1ccc2c(c1)[se]c1[cH:1]cccc12',                    0.3),
]
# Pre-parse and resolve marker once (shared by all workers via fork).
_DONOR_CACHE=[]
for _lbl,_smi,_wt in DONOR_LIB:
    _dm=MolFromSmiles(_smi)
    if _dm is None: continue
    _conn=None
    for _ai,_at_obj in enumerate(_dm.GetAtoms()):
        if _at_obj.GetAtomMapNum()==1:
            _conn=_ai; _at_obj.SetAtomMapNum(0); break
    if _conn is None: continue
    _DONOR_CACHE.append((_lbl,_dm,_conn,_wt))
_DONOR_WEIGHTS=np.array([d[3] for d in _DONOR_CACHE],dtype=np.float64)
if _DONOR_WEIGHTS.sum()>0: _DONOR_WEIGHTS/=_DONOR_WEIGHTS.sum()

def _grft(rw, parent_i, donor_idx, rng):
    """Graft donor #donor_idx onto aryl-CH at parent_i.

    Adds donor atoms/bonds to rw and forms the new connection bond.
    Consumes one H on the parent and one H on the donor's connection atom.
    Returns the donor label on success, or None on failure.
    """
    if not _DONOR_CACHE: return None
    pa=rw.GetAtomWithIdx(parent_i)
    if pa.GetTotalNumHs()<=0: return None
    lbl, donor_mol, conn_idx, _ = _DONOR_CACHE[donor_idx]
    da_orig=donor_mol.GetAtomWithIdx(conn_idx)
    if da_orig.GetTotalNumHs()<=0: return None
    # Copy donor atoms into rw, remembering index map.
    map_d2new={}
    for atom in donor_mol.GetAtoms():
        new_idx=rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        nat=rw.GetAtomWithIdx(new_idx)
        nat.SetIsAromatic(atom.GetIsAromatic())
        nat.SetFormalCharge(atom.GetFormalCharge())
        # Carry explicit H count for the donor's connection N (it had 1 H);
        # we will subtract 1 below.
        if atom.GetNumExplicitHs()>0:
            nat.SetNumExplicitHs(atom.GetNumExplicitHs())
            nat.SetNoImplicit(True)
        map_d2new[atom.GetIdx()]=new_idx
    for bond in donor_mol.GetBonds():
        rw.AddBond(map_d2new[bond.GetBeginAtomIdx()],
                   map_d2new[bond.GetEndAtomIdx()],
                   bond.GetBondType())
    new_conn=map_d2new[conn_idx]
    # Form the new connection bond (single bond, aryl–donor).
    rw.AddBond(parent_i, new_conn, Chem.BondType.SINGLE)
    # Consume one H on parent (aryl-CH → aryl-C-bonded).
    if pa.GetNoImplicit():
        pa.SetNumExplicitHs(max(0, pa.GetNumExplicitHs()-1))
    # Consume one H on donor connection atom (NH → N or CH → C).
    da_new=rw.GetAtomWithIdx(new_conn)
    if da_new.GetNumExplicitHs()>0:
        da_new.SetNumExplicitHs(max(0, da_new.GetNumExplicitHs()-1))
    # Aryl carbons in donor that lost an H must allow implicit Hs again.
    da_new.SetNoImplicit(False)
    return lbl

def _cn(rw, i):
    """Cyano substitution: aryl-CH at i → aryl-C(-C#N).

    Adds two atoms (one sp C, one N) and the triple bond.
    """
    a=rw.GetAtomWithIdx(i)
    if a.GetTotalNumHs()<=0: return False
    c1=rw.AddAtom(Chem.Atom(6))   # sp carbon of CN
    n1=rw.AddAtom(Chem.Atom(7))   # nitrile N
    rw.GetAtomWithIdx(c1).SetIsAromatic(False)
    rw.GetAtomWithIdx(n1).SetIsAromatic(False)
    rw.AddBond(i,  c1, Chem.BondType.SINGLE)
    rw.AddBond(c1, n1, Chem.BondType.TRIPLE)
    if a.GetNoImplicit(): a.SetNumExplicitHs(max(0, a.GetNumExplicitHs()-1))
    return True

def _so2_oxidize(rw, s_idx):
    """Promote a sulfide-S in the ring to a sulfone S(=O)2.

    The S atom must be aromatic and divalent (typical thiophene/dibenzothiophene
    S). After oxidation the S is no longer aromatic; ring may remain aromatic
    only if other atoms compensate (rare). In practice we expect this to break
    aromaticity of the immediate ring — we therefore mark the molecule as
    requiring the sanitiser to re-perceive aromaticity. Add two =O atoms.
    """
    s=rw.GetAtomWithIdx(s_idx)
    if s.GetAtomicNum()!=16: return False
    if s.GetDegree()!=2: return False
    s.SetIsAromatic(False)
    o1=rw.AddAtom(Chem.Atom(8)); rw.GetAtomWithIdx(o1).SetIsAromatic(False)
    o2=rw.AddAtom(Chem.Atom(8)); rw.GetAtomWithIdx(o2).SetIsAromatic(False)
    rw.AddBond(s_idx, o1, Chem.BondType.DOUBLE)
    rw.AddBond(s_idx, o2, Chem.BondType.DOUBLE)
    # Convert S–C ring bonds from aromatic to single (sulfone S is sp3-like).
    for b in list(s.GetBonds()):
        oi=b.GetOtherAtomIdx(s_idx)
        if rw.GetAtomWithIdx(oi).GetAtomicNum()==6 and b.GetBondType()==Chem.BondType.AROMATIC:
            b.SetBondType(Chem.BondType.SINGLE)
    return True

def _q4(rw, i, j):
    """sp3 quaternary bridge: convert biaryl single bond (i,j) into a
    C(Me)2 bridge by inserting one sp3 carbon between i and j (replacing
    the existing single bond).

    Pre-condition: i,j are aromatic carbons connected by a single bond
    that is *not* in a ring (so removing it doesn't break aromaticity).
    Post: i — Cq(Me)(Me) — j; both aromatic rings preserved; molecule
    becomes locally sp3-bridged (DMAC/9,9-dimethylfluorene-like).
    """
    bond=rw.GetBondBetweenAtoms(i,j)
    if bond is None: return False
    if bond.IsInRing(): return False
    if bond.GetBondType()!=Chem.BondType.SINGLE: return False
    # Add the sp3 quaternary carbon and two methyl carbons.
    cq=rw.AddAtom(Chem.Atom(6))
    rw.GetAtomWithIdx(cq).SetIsAromatic(False)
    me1=rw.AddAtom(Chem.Atom(6)); rw.GetAtomWithIdx(me1).SetIsAromatic(False)
    me2=rw.AddAtom(Chem.Atom(6)); rw.GetAtomWithIdx(me2).SetIsAromatic(False)
    rw.RemoveBond(i,j)
    rw.AddBond(i, cq, Chem.BondType.SINGLE)
    rw.AddBond(cq, j, Chem.BondType.SINGLE)
    rw.AddBond(cq, me1, Chem.BondType.SINGLE)
    rw.AddBond(cq, me2, Chem.BondType.SINGLE)
    return True

# ═══════════════════════════════════════════════════════════════════
#  v15 — ν-DABNA RIM EXTENSION  &  P-CHALCOGENIDE INSERTION
# ═══════════════════════════════════════════════════════════════════
def _ann_BN(rw, i, j):
    """ν-DABNA-style ring-fusion B/N extension at perimeter edge (i, j).

    v17: emits the canonical Kekulé form RDKit returns for an N-aryl,
    B-aryl 1,2-azaborine. Fuses a NEW 6-membered ring across the
    existing aromatic edge (i, j) of the parent BN-MR core. The new
    ring contains exactly one N and one B at ADJACENT positions
    (1,2-azaborine motif), plus four ring carbons (counting the shared
    edge endpoints i and j).

    Topology (going around the new ring):
        i = c0 — N — B — c1 = j — (back to i via the shared edge,
        which is demoted from AROMATIC to SINGLE so the new ring's
        Kekulé form is locally consistent).

    IMPORTANT — RDKit aromaticity perception:
        With BOTH N and B carrying aryl pendants, RDKit does NOT
        perceive this 6-ring as aromatic. The canonical form RDKit
        returns is `C1=CB(c2ccccc2)N(c2ccccc2)C=C1` — explicit Kekulé,
        non-aromatic. (v16's earlier docstring claimed Hückel-6π
        aromaticity; that claim was incorrect for RDKit's perception
        model.) The π count argument (N lone pair + 4 C contributions
        + 0 from B = 6 e) is chemically defensible but the molecule is
        not flagged aromatic by RDKit.
        DOWNSTREAM IMPACT: `_get_core` Stage A (aromatic-only) will
        not include the new ring, but Stage B (all-rings, MR-motif)
        DOES — so the family classifier still labels these molecules
        as BN-MR and `n_boron_in_core` increments correctly. Verified
        in test ITEM 11.

    Both N and B carry a Kekulé phenyl pendant, matching the canonical
    ν-DABNA / DABNA-extension substitution pattern (N-aryl + B-aryl).
    The new ring bonds are emitted in explicit Kekulé form; the parent's
    i—j edge is demoted from AROMATIC to SINGLE so the sanitiser can
    re-kekulize the parent ring consistently.

    Pre: i and j are aromatic CH atoms on the perimeter of the parent
         core, connected by an aromatic bond, each with ≥1 H. The parent
         must already contain at least one B or BN motif so that the new
         azaborine extends an MR pattern rather than inventing one
         de novo (this is enforced upstream in `_SC.__init__`).

    Returns True on success.
    """
    ai=rw.GetAtomWithIdx(i)
    aj=rw.GetAtomWithIdx(j)
    if ai.GetTotalNumHs()<=0 or aj.GetTotalNumHs()<=0: return False
    if not (ai.GetIsAromatic() and aj.GetIsAromatic()): return False
    if not (ai.GetAtomicNum()==6 and aj.GetAtomicNum()==6): return False
    if ai.GetDegree()>2 or aj.GetDegree()>2: return False
    bond_ij=rw.GetBondBetweenAtoms(i, j)
    if bond_ij is None: return False
    # The shared edge must be aromatic (i.e. lie inside an aromatic ring).
    if bond_ij.GetBondType()!=Chem.BondType.AROMATIC: return False

    # ── New ring atoms ────────────────────────────────────────────────
    # Build the new ring in the canonical Kekulé form RDKit returns for
    # an N-aryl, B-aryl 1,2-azaborine: C=C-B(Ph)-N(Ph)-C=C (the parent
    # i—j bond becomes the SINGLE bond closing the ring). The
    # docstring's earlier "Hückel-aromatic" claim is incorrect: RDKit
    # does NOT perceive this 6-ring as aromatic when N and B both carry
    # aryl pendants, so we explicitly emit the Kekulé form and let the
    # sanitiser re-perceive whatever it can. The bond we ADD between i
    # and the new C must be DOUBLE (to satisfy the carbon valence of c0
    # at degree 2 in the new ring), and likewise c1—j is DOUBLE.
    # Sanitisation will kekulize the parent's adjoining ring so the
    # i—j bond becomes its single-bond Kekulé partner.
    c0=rw.AddAtom(Chem.Atom(6))   # C bonded to i (double) and to nN (single)
    nN=rw.AddAtom(Chem.Atom(7))   # N (no H) — bears N-phenyl pendant
    nB=rw.AddAtom(Chem.Atom(5))   # B (no H) — bears B-phenyl pendant
    c1=rw.AddAtom(Chem.Atom(6))   # C bonded to nB (single) and to j (double)
    aN=rw.GetAtomWithIdx(nN); aN.SetNumExplicitHs(0); aN.SetNoImplicit(True)
    aB=rw.GetAtomWithIdx(nB); aB.SetNumExplicitHs(0); aB.SetNoImplicit(True)
    aB.SetFormalCharge(0)

    # New ring bonds (canonical Kekulé pattern; matches the canonical
    # RDKit output `C1=CB(Ph)N(Ph)C=C1`):
    #     i = c0 — nN — nB — c1 = j  (with the existing i—j as single)
    rw.AddBond(i,   c0, Chem.BondType.DOUBLE)
    rw.AddBond(c0,  nN, Chem.BondType.SINGLE)
    rw.AddBond(nN,  nB, Chem.BondType.SINGLE)
    rw.AddBond(nB,  c1, Chem.BondType.SINGLE)
    rw.AddBond(c1,  j,  Chem.BondType.DOUBLE)

    # The shared edge i—j was AROMATIC; force it to SINGLE so the new
    # ring's Kekulé form is locally consistent. Sanitiser will re-perceive
    # aromaticity in the parent ring after kekulization.
    bond_ij.SetBondType(Chem.BondType.SINGLE)
    bond_ij.SetIsAromatic(False)
    ai.SetIsAromatic(False)
    aj.SetIsAromatic(False)

    # i and j become ring-junctions (degree 3 in the fused system).
    # Their H counts will be re-computed by the sanitiser; only force
    # decrement when the molecule had explicit H tracking.
    if ai.GetNoImplicit():
        ai.SetNumExplicitHs(max(0, ai.GetNumExplicitHs()-1))
    if aj.GetNoImplicit():
        aj.SetNumExplicitHs(max(0, aj.GetNumExplicitHs()-1))

    # ── N-phenyl pendant (Kekulé form; sanitiser will rearomatise) ───
    ph_n=[rw.AddAtom(Chem.Atom(6)) for _ in range(6)]
    rw.AddBond(nN,    ph_n[0], Chem.BondType.SINGLE)
    rw.AddBond(ph_n[0],ph_n[1],Chem.BondType.DOUBLE)
    rw.AddBond(ph_n[1],ph_n[2],Chem.BondType.SINGLE)
    rw.AddBond(ph_n[2],ph_n[3],Chem.BondType.DOUBLE)
    rw.AddBond(ph_n[3],ph_n[4],Chem.BondType.SINGLE)
    rw.AddBond(ph_n[4],ph_n[5],Chem.BondType.DOUBLE)
    rw.AddBond(ph_n[5],ph_n[0],Chem.BondType.SINGLE)

    # ── B-phenyl pendant (Kekulé form; sanitiser will rearomatise) ───
    ph_b=[rw.AddAtom(Chem.Atom(6)) for _ in range(6)]
    rw.AddBond(nB,    ph_b[0], Chem.BondType.SINGLE)
    rw.AddBond(ph_b[0],ph_b[1],Chem.BondType.DOUBLE)
    rw.AddBond(ph_b[1],ph_b[2],Chem.BondType.SINGLE)
    rw.AddBond(ph_b[2],ph_b[3],Chem.BondType.DOUBLE)
    rw.AddBond(ph_b[3],ph_b[4],Chem.BondType.SINGLE)
    rw.AddBond(ph_b[4],ph_b[5],Chem.BondType.DOUBLE)
    rw.AddBond(ph_b[5],ph_b[0],Chem.BondType.SINGLE)

    return True

def _pinsert(rw, i, j, chalcogen=8):
    """P-chalcogenide insertion: replace biaryl single bond (i,j) with a
    P(=X)(Ph) bridge, where X ∈ {O (Z=8), S (Z=16), Se (Z=34)}.

    Result topology: i — P(=X)(Ph) — j   (DPSPO / DPSP=Se motif).
    P is tetravalent (sp³): bonds to i, j, =X, and Ph.

    Pre-condition: (i,j) is a single bond outside any ring between two
    aromatic carbons (same as for q4).
    """
    if chalcogen not in (8, 16, 34): return False
    bond = rw.GetBondBetweenAtoms(i, j)
    if bond is None: return False
    if bond.IsInRing(): return False
    if bond.GetBondType() != Chem.BondType.SINGLE: return False
    # Add P, =X, and a Kekulé-form phenyl pendant on P.
    P_idx = rw.AddAtom(Chem.Atom(15))
    X_idx = rw.AddAtom(Chem.Atom(chalcogen))
    ph = [rw.AddAtom(Chem.Atom(6)) for _ in range(6)]
    rw.RemoveBond(i, j)
    rw.AddBond(i,     P_idx, Chem.BondType.SINGLE)
    rw.AddBond(P_idx, j,     Chem.BondType.SINGLE)
    rw.AddBond(P_idx, X_idx, Chem.BondType.DOUBLE)
    rw.AddBond(P_idx, ph[0], Chem.BondType.SINGLE)
    # Phenyl ring (Kekulé alternation).
    rw.AddBond(ph[0], ph[1], Chem.BondType.DOUBLE)
    rw.AddBond(ph[1], ph[2], Chem.BondType.SINGLE)
    rw.AddBond(ph[2], ph[3], Chem.BondType.DOUBLE)
    rw.AddBond(ph[3], ph[4], Chem.BondType.SINGLE)
    rw.AddBond(ph[4], ph[5], Chem.BondType.DOUBLE)
    rw.AddBond(ph[5], ph[0], Chem.BondType.SINGLE)
    return True

_gc=_gl=_gca=_gn=_ga=_gaa=_gti=_gann=None
def _pinit(co,lo,ss,nt,att,exp,ti,eb,ed,ena=False,sa_ref=None):
    global _gc,_gl,_gca,_gn,_ga,_gaa,_gti,_gann
    _gc=co;_gl=lo;_gn=nt;_ga=att;_gaa=ALLOWED_ATOMS_EXPLR if exp else ALLOWED_ATOMS;_gti=ti
    _gann=bool(ena)
    # On macOS (default 'spawn' since Py3.8) and Windows, worker processes
    # re-import the module fresh and see _GLOBAL_SASCORER._fitted == False,
    # silently skipping the SA gate. Re-fit it here from the reference
    # corpus the parent passes in when supplied. On Linux fork, this is a
    # no-op (state is already inherited).
    if sa_ref and not _GLOBAL_SASCORER._fitted:
        try:
            _GLOBAL_SASCORER.fit(list(sa_ref))
        except Exception:
            pass
    _gca=[]
    for s in ss:
        m=MolFromSmiles(s)
        if m:
            try:
                c=_SC(s,m,exp,eb,ed)
                if c.ne>0 or (ena and len(c.ann_e)>0): _gca.append(c)
            except Exception: pass
def _wgen(wid):
    if not _gca: return []
    rng=np.random.RandomState(SEED+wid*1000); found=[]; seen=set(); ns=len(_gca)
    # Operator weights (v15). Slot order:
    #   F, paired_F, CH3, aza, F+aza, paired_CH3, diaza, tBu,
    #   [annul], graft_D, CN, SO2, q4,
    #   rim_BN, P_insert
    # Rim-extension and P-chalcogenide insertion are scientifically
    # consequential but expensive (low success rate due to sanitization
    # constraints), so they receive small but non-zero mass.
    if _gann:
        W=np.array([0.15,0.06,0.08,0.10,0.10,0.05,0.05,0.04,
                    0.06,    # annul (plain benzannulation)
                    0.15,    # graft_D
                    0.05,    # CN
                    0.04,    # SO2
                    0.03,    # q4
                    0.02,    # rim_BN
                    0.02])   # P_insert
    else:
        W=np.array([0.17,0.06,0.08,0.11,0.11,0.06,0.06,0.04,
                    0.17,    # graft_D
                    0.05,    # CN
                    0.02,    # SO2
                    0.02,    # q4
                    0.03,    # rim_BN  (more weight when annul is off)
                    0.02])   # P_insert
    W=W/W.sum(); CW=np.cumsum(W)
    # v17 (#47): np.cumsum on a renormalised W may leave CW[-1] just below
    # 1.0 due to float drift; if r=rng.random() ∈ [0,1) lands in that gap
    # the elif chain bottoms out with no operator firing (wasted attempt
    # via `else: continue`). Clamp the final cumulative weight to exactly
    # 1.0 so every r ∈ [0,1) maps to some operator branch.
    CW[-1]=1.0
    for att in range(_ga):
        # Lockless read of the shared counter every 2000 attempts: this is
        # safe on POSIX because Python's c_int reads are atomic on word-
        # aligned platforms (x86-64, ARM64). The worst case is a stale read
        # that delays the break by one polling cycle, never an over-shoot
        # of the candidate target — increments still go through `_gc.get_lock`
        # below. This pattern was stress-tested at nw∈{16,24,32}: 19/19
        # repeats produced identical candidate counts (G-31 in v17 review).
        if att%2000==0 and _gc.value>=_gn: break
        c=_gca[rng.randint(ns)]; rw=Chem.RWMol(Chem.Mol(c.mol)); ea=[]; mf=""
        try:
            r=rng.random()
            if r<CW[0] and c.f_s:
                si=c.f_s[rng.randint(len(c.f_s))]
                if _af(rw,si): ea.append((si,'F')); mf='F'
                else: continue
            elif r<CW[1] and c.pf_g:
                g=c.pf_g[rng.randint(len(c.pf_g))]; fs=set(c.f_s)
                for gi in g:
                    if gi in fs:
                        a=rw.GetAtomWithIdx(gi)
                        if a.GetTotalNumHs()>0:
                            fi=rw.AddAtom(Chem.Atom(9)); rw.AddBond(gi,fi,Chem.BondType.SINGLE)
                            if a.GetNoImplicit(): a.SetNumExplicitHs(max(0,a.GetNumExplicitHs()-1))
                            ea.append((gi,'F'))
                if not ea: continue
                mf='paired_F'
            elif r<CW[2] and c.ch3_s:
                si=c.ch3_s[rng.randint(len(c.ch3_s))]
                if _am(rw,si): ea.append((si,'CH3')); mf='CH3'
                else: continue
            elif r<CW[3] and c.aza_s:
                ai,nz=c.aza_s[rng.randint(len(c.aza_s))]
                a=rw.GetAtomWithIdx(ai); src_z=a.GetAtomicNum()
                a.SetAtomicNum(nz); a.SetFormalCharge(0)
                a.SetNumExplicitHs(0); a.SetNoImplicit(False)
                ea.append((ai,_aza_tag('aza',src_z,nz))); mf='aza'
            elif r<CW[4] and c.f_s and c.aza_s:
                si=c.f_s[rng.randint(len(c.f_s))]; _af(rw,si); ea.append((si,'F'))
                ai,nz=c.aza_s[rng.randint(len(c.aza_s))]
                a=rw.GetAtomWithIdx(ai); src_z=a.GetAtomicNum()
                a.SetAtomicNum(nz); a.SetFormalCharge(0)
                a.SetNumExplicitHs(0); a.SetNoImplicit(False)
                ea.append((ai,_aza_tag('aza',src_z,nz))); mf='F+aza'
            elif r<CW[5] and c.pc_g:
                g=c.pc_g[rng.randint(len(c.pc_g))]; cs=set(c.ch3_s)
                for gi in g:
                    if gi in cs: _am(rw,gi); ea.append((gi,'CH3'))
                if not ea: continue
                mf='paired_CH3'
            elif r<CW[6] and c.dz_p:
                pair=c.dz_p[rng.randint(len(c.dz_p))]
                for pi in pair:
                    a=rw.GetAtomWithIdx(pi); src_z=a.GetAtomicNum()
                    a.SetAtomicNum(7); a.SetFormalCharge(0)
                    a.SetNumExplicitHs(0); a.SetNoImplicit(False)
                    # tag as 'diaza' (not 'aza') so edits_ok recognises the
                    # symmetry-paired case and skips the path-through-frontier
                    # check (paired aza substitutions cancel by construction).
                    # v17 (#48): tag includes source>target so the novelty
                    # signature distinguishes paired-C>N from paired-S>N etc.
                    ea.append((pi,_aza_tag('diaza',src_z,7)))
                mf='diaza'
            elif r<CW[7] and c.tbu_s:
                si=c.tbu_s[rng.randint(len(c.tbu_s))]
                if _at(rw,si): ea.append((si,'tBu')); mf='tBu'
                else: continue
            elif _gann and len(CW)>=13 and r<CW[8] and c.ann_e:
                # Benzannulation operator (only when --enable_annulation).
                edge=c.ann_e[rng.randint(len(c.ann_e))]
                ai,aj=edge
                if not (_allowed(c.scaff,c.roles.get(ai,SR.FORBIDDEN),'annul')
                        and _allowed(c.scaff,c.roles.get(aj,SR.FORBIDDEN),'annul')):
                    continue
                if _ann(rw,ai,aj):
                    ea.append((ai,'annul')); ea.append((aj,'annul'))
                    mf='annul'
                else: continue
            elif r<CW[8 if not _gann else 9] and c.graft_s and _DONOR_CACHE:
                # Donor-graft operator (v14): aryl-CH → aryl-N(donor) bond.
                si=c.graft_s[rng.randint(len(c.graft_s))]
                # Weighted donor pick.
                d_choice=rng.choice(len(_DONOR_CACHE), p=_DONOR_WEIGHTS)
                lbl=_grft(rw, si, int(d_choice), rng)
                if lbl is None: continue
                ea.append((si,'graft_D'))
                mf=f'graft_D[{lbl}]'
            elif r<CW[9 if not _gann else 10] and c.cn_s:
                # Cyano substitution: aryl-CH → aryl-C#N.
                si=c.cn_s[rng.randint(len(c.cn_s))]
                if _cn(rw, si): ea.append((si,'CN')); mf='CN'
                else: continue
            elif r<CW[10 if not _gann else 11] and c.so2_s:
                # Sulfone oxidation: ring S → S(=O)2.
                si=c.so2_s[rng.randint(len(c.so2_s))]
                if _so2_oxidize(rw, si): ea.append((si,'SO2')); mf='SO2'
                else: continue
            elif r<CW[11 if not _gann else 12] and c.q4_e:
                # sp3 quaternary bridge insertion.
                ai,aj=c.q4_e[rng.randint(len(c.q4_e))]
                if _q4(rw, ai, aj):
                    ea.append((ai,'q4')); ea.append((aj,'q4'))
                    mf='q4_bridge'
                else: continue
            elif r<CW[12 if not _gann else 13] and c.rim_e:
                # v16: ν-DABNA-style rim B/N extension. Fuses a new
                # 1,2-azaborine 6-ring across a perimeter aryl edge
                # (i, j) of the parent BN-MR core. Increases n_B and n_N
                # in the extended core simultaneously, mimicking the real
                # ν-DABNA / DABNA-extension topology.
                ai, aj = c.rim_e[rng.randint(len(c.rim_e))]
                if _ann_BN(rw, ai, aj):
                    ea.append((ai, 'rim_BN'))
                    ea.append((aj, 'rim_BN'))
                    mf = 'rim_BN'
                else: continue
            elif r<CW[13 if not _gann else 14] and c.p_e:
                # v15: P-chalcogenide insertion at biaryl single bond.
                # Pick chalcogen at random with PO:PS:PSe ≈ 5:2:1 (corpus mass).
                chalc = int(rng.choice([8, 16, 34], p=[0.62, 0.25, 0.13]))
                ai, aj = c.p_e[rng.randint(len(c.p_e))]
                if _pinsert(rw, ai, aj, chalcogen=chalc):
                    chalc_sym = {8:'O', 16:'S', 34:'Se'}[chalc]
                    ea.append((ai, f'P={chalc_sym}'))
                    ea.append((aj, f'P={chalc_sym}'))
                    mf = f'P_insert[P={chalc_sym}]'
                else: continue
            else: continue
            if len(ea)>1 and not edits_ok(c.mol,ea,c.roles,c.scaff): continue
            res=_safe(rw)
            if res is None: continue
            ns_,nm=res
            if ns_ in seen: continue
            seen.add(ns_)
            if not _hard_filter(nm,_gaa): continue
            try: inchi=MolToInchi(nm)
            except Exception: inchi=None
            if inchi and inchi in _gti: continue
            ranks=list(Chem.CanonicalRankAtoms(c.mol,breakTies=False))
            ec=[f"R{ranks[ai]}:{et}" for ai,et in ea if ai<len(ranks)]
            ec.sort(); esig="|".join(ec) if ec else "none"
            found.append({"smiles":ns_,"parent_smi":c.smi,"parent_scaffold":c.scaff,
                          "parent_core_size":c.csz,"edit_sig":esig,
                          "edit_depth":len(ea),"mut_family":mf})
            with _gc.get_lock(): _gc.value+=1
        except Exception: continue
    return found

# ═══════════════════════════════════════════════════════════════════
#  v18: DEEP GENERATIVE MODELS — REMOVED
# ═══════════════════════════════════════════════════════════════════
# v17's VAE / Flow / Diffusion / RL stack and the
# `_train_and_sample_generative` helper used to live here. v18 drops
# them entirely (CLI flags, orchestrator class, and the merge step
# in main()). The rest of the pipeline (operator-based generation,
# filters, surrogates, scoring) is unchanged from v17.

# ═══════════════════════════════════════════════════════════════════
#  v16.1 — ENHANCEMENT MODULES (additive on top of v16 base)
# ═══════════════════════════════════════════════════════════════════
# All additions are modular and configurable via CLI flags. They do not
# modify the deep generative stack (which remains disabled by default),
# the chemistry filters, or the legacy output schema. Each block below
# is documented with its purpose, inputs, and outputs.
# ───────────────────────────────────────────────────────────────────

# --- (a) Murcko-core split CV (mandatory; complements scaffold-family CV) ---
def murcko_scaffold_key(smi):
    """Canonical Murcko scaffold SMILES used as a grouping key.

    Returns "" if the molecule is unparseable; the caller bins these into
    a `_misc_` cluster so they don't dominate or fragment leave-one-out
    splits. Uses RDKit's MurckoScaffold (no chirality).
    """
    try:
        m=MolFromSmiles(smi) if isinstance(smi,str) else None
        if m is None: return ""
        sc=MurckoScaffold.GetScaffoldForMol(m)
        if sc is None: return ""
        return MolToSmiles(sc) or ""
    except Exception:
        return ""

def core_split_groups(smiles_list, min_per_cluster=2):
    """Murcko-scaffold group label per training SMILES.

    Singletons (clusters of size <min_per_cluster) are merged into a
    deterministic '_misc_' bucket so the resulting groups are usable for
    leave-cluster-out CV.
    """
    raw=[murcko_scaffold_key(s) for s in smiles_list]
    counts=Counter(raw)
    return [r if r and counts[r]>=min_per_cluster else "_misc_" for r in raw]

def core_split_cv(X, y, smiles_list, label="", n_estimators=300,
                  min_test_size=2, min_train_size=5):
    """Leave-Murcko-scaffold-out cross-validation.

    Complements `scaffold_stratified_cv` (which is family-level). Reports
    per-cluster MAE and the macro-MAE across clusters with ≥min_test_size
    members. A small spread between this and the family-level macro-MAE
    indicates the surrogate transfers well across both axes; a large
    spread flags either family or local-topology overfitting.
    """
    groups=core_split_groups(smiles_list)
    out={"per_core":{},"macro_MAE":None,"n_clusters_evaluated":0,
         "n_total_clusters":len(set(groups))}
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    bp=dict(n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=3, tree_method="hist", random_state=SEED, n_jobs=-1)
    fam_maes=[]
    for cluster in sorted(set(groups)):
        train_idx=[i for i,g in enumerate(groups) if g!=cluster]
        test_idx =[i for i,g in enumerate(groups) if g==cluster]
        if len(test_idx)<min_test_size or len(train_idx)<min_train_size:
            continue
        m=xgb.XGBRegressor(**bp); m.fit(Xs[train_idx], y[train_idx])
        yp=m.predict(Xs[test_idx])
        mae=float(mean_absolute_error(y[test_idx], yp))
        # Truncate very long Murcko SMILES used as keys.
        out["per_core"][cluster[:80]]={"n_test":int(len(test_idx)),"MAE":mae}
        fam_maes.append(mae)
    out["macro_MAE"]=float(np.mean(fam_maes)) if fam_maes else None
    out["n_clusters_evaluated"]=len(fam_maes)
    if fam_maes:
        log.info("    [%s] core-split MAE: macro=%.4f over %d/%d clusters",
                 label, out["macro_MAE"], len(fam_maes),
                 out["n_total_clusters"])
    else:
        log.info("    [%s] core-split: 0 evaluable clusters", label)
    return out

# --- (b) Scaffold-aware conformal calibration ---
def calibrate_sigma_by_scaffold(ens, scaler_obj, X_cal, y_cal, smi_cal,
                                target_coverage=0.683, min_per_family=5,
                                pred_fn=None):
    """Compute global κ AND per-scaffold-family κ.

    Returns (kappa_global, kappa_by_family) where kappa_by_family is a
    dict[family → κ]. Families with fewer than `min_per_family` calibration
    points fall back to the global κ. The scoring path uses the candidate's
    parent scaffold to select the appropriate κ; novel families fall back
    to the global κ automatically (handled in `score_cands`).
    """
    if pred_fn is None: pred_fn=pred_unc
    if len(X_cal)==0: return 1.0, {}
    mu, sig = pred_fn(ens, scaler_obj, X_cal)
    sig=np.maximum(sig, 1e-8)
    z=np.abs(np.asarray(y_cal)-mu)/sig
    kappa_global=float(np.quantile(z, target_coverage)) if len(z)>=5 else 1.0
    fams=scaffold_groups(smi_cal)
    kappa_by_fam={}
    for fam in sorted(set(fams)):
        idx=[i for i,g in enumerate(fams) if g==fam]
        if len(idx)>=min_per_family:
            kappa_by_fam[fam]=float(np.quantile(z[idx], target_coverage))
        else:
            kappa_by_fam[fam]=kappa_global
    return kappa_global, kappa_by_fam

# --- (c) Residual covariance for joint target uncertainty ---
def compute_residual_covariance(ensembles_dict, X, target_keys, pred_fn=None):
    """Cross-target covariance of residuals (y - μ) over the full training
    set, in the user-supplied units (the conversion to internal MC units
    is handled inside `score_cands`).

    target_keys : ordered list of label names that exist in `ensembles_dict`.
                  The returned matrix's row/column order matches this list.

    Returns a (k×k) np.ndarray that is GUARANTEED PSD (positive
    semidefinite) so downstream Cholesky decomposition cannot fail.
    Falls back to identity if there are too few training samples.

    PSD enforcement (v17 strengthened):
      1. Symmetrise: cov ← (cov + cov.T) / 2  (numpy.cov is already
         symmetric, but rounding can introduce asymmetry).
      2. Eigen-clip: any eigenvalue λ_i < ε is clipped to ε. This
         repairs degenerate or near-rank-deficient cases (e.g. when
         two targets are perfectly correlated) without inflating the
         genuine spectrum.
      3. Diagonal jitter: cov ← cov + ε·diag(max(diag, ε)) as a
         conservative final safety margin.
    """
    if pred_fn is None: pred_fn=pred_unc
    res=[]
    for lb in target_keys:
        ens, sc, y = ensembles_dict[lb]
        mu, _ = pred_fn(ens, sc, X)
        res.append(np.asarray(y) - mu)
    R=np.vstack(res).T  # (n, k)
    if R.shape[0]<5:
        return np.eye(R.shape[1])
    cov=np.cov(R, rowvar=False)
    if np.ndim(cov)==0:
        cov=np.array([[float(cov)]])
    cov=np.asarray(cov, dtype=np.float64)
    # Step 1: symmetrise.
    cov=(cov + cov.T)*0.5
    # Step 2: eigen-clip (handles degenerate / indefinite inputs).
    eps=1e-8
    try:
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, eps)
        cov = (V * w) @ V.T
        cov = (cov + cov.T)*0.5
    except np.linalg.LinAlgError:
        # If eigh fails outright, fall back to an identity scaled by
        # the trace. Better than crashing.
        tr = float(np.trace(cov)) if cov.size else 1.0
        cov = (max(tr, eps) / max(R.shape[1], 1)) * np.eye(R.shape[1])
    # Step 3: diagonal jitter as conservative final margin.
    diag=np.maximum(np.diag(cov), eps)
    cov=cov + 1e-6*np.diag(diag)
    return cov.astype(np.float64)

# --- (d) Active-learning validation history ---
def update_validation_history(history_path, run_id, scored_df, qc_df=None,
                              top_n=50):
    """Append the current run's top-N predictions (and QC values, if any)
    to a persistent JSON history file, atomically.

    Schema:
        {"runs": [
            {"run_id": str,
             "timestamp": str,
             "n_top": int,
             "top": [
                {"smiles": str,
                 "pred":  {tgt: μ},
                 "unc":   {tgt: σ},
                 "qc":    {tgt: y}     # only if QC merged for this row
                }, ...]
            }, ...]
        }

    The history is meant to be loaded by future runs (via
    `summarize_validation_history`) to compute a per-iteration MAE / bias
    curve and surface drift across active-learning rounds.
    """
    payload={"run_id":str(run_id),
             "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
             "n_top":int(min(top_n, len(scored_df))),
             "top":[]}
    pred_cols=[c for c in scored_df.columns if c.startswith("pred_")]
    unc_cols =[c for c in scored_df.columns if c.startswith("unc_")]
    qc_map={}
    if qc_df is not None and len(qc_df)>0 and "smiles" in qc_df.columns:
        for _,row in qc_df.iterrows():
            try:
                cs=MolToSmiles(MolFromSmiles(str(row["smiles"])))
                qc_map[cs]=row.to_dict()
            except Exception:
                continue
    df_top=scored_df.head(top_n)
    for _,row in df_top.iterrows():
        rec={"smiles":str(row.get("smiles","")),
             "pred":{c.replace("pred_",""):float(row[c]) for c in pred_cols
                     if pd.notna(row[c])},
             "unc": {c.replace("unc_","") :float(row[c]) for c in unc_cols
                     if pd.notna(row[c])}}
        try:
            cs=MolToSmiles(MolFromSmiles(str(rec["smiles"])))
        except Exception:
            cs=rec["smiles"]
        if cs in qc_map:
            qc_rec={}
            for k,v in qc_map[cs].items():
                if k=="smiles": continue
                if isinstance(v,(int,float)) and pd.notna(v):
                    qc_rec[k]=float(v)
            if qc_rec: rec["qc"]=qc_rec
        payload["top"].append(rec)
    history={"runs":[]}
    if os.path.isfile(history_path):
        try:
            with open(history_path) as fh:
                history=json.load(fh)
                if not isinstance(history, dict) or "runs" not in history:
                    history={"runs":[]}
        except Exception:
            history={"runs":[]}
    history["runs"].append(payload)
    tmp=history_path+".tmp"
    with open(tmp,"w") as fh: json.dump(history, fh, indent=2)
    os.replace(tmp, history_path)
    return payload

def summarize_validation_history(history_path):
    """Return a list of per-run summary dicts (MAE / bias per target),
    computed only over rows where matched QC values exist. Empty list if
    the file is absent or has no QC-tagged rows.
    """
    if not os.path.isfile(history_path): return []
    try:
        with open(history_path) as fh: history=json.load(fh)
    except Exception:
        return []
    rows=[]
    for run in history.get("runs",[]):
        per_tgt=defaultdict(list)
        for rec in run.get("top",[]):
            qc=rec.get("qc",{}); pred=rec.get("pred",{})
            for k,vk in qc.items():
                if k in pred:
                    per_tgt[k].append((float(pred[k]), float(vk)))
        if not per_tgt: continue
        summary={"run_id":run.get("run_id",""),
                 "timestamp":run.get("timestamp",""),
                 "n_with_qc":sum(len(v) for v in per_tgt.values()),
                 "per_target":{}}
        for k,pairs in per_tgt.items():
            if not pairs: continue
            arr=np.array(pairs)
            summary["per_target"][k]={
                "n":int(len(arr)),
                "MAE":float(np.mean(np.abs(arr[:,0]-arr[:,1]))),
                "bias":float(np.mean(arr[:,0]-arr[:,1]))}
        rows.append(summary)
    return rows

# --- (e) Label-source weighting ---
def compute_label_source_weights(df, source_col=None, source_weights=None):
    """Per-row sample weights from a label-source column.

    df             : DataFrame in the same row order as X, y arrays.
    source_col     : column name to read; if None or absent, weights=1.
    source_weights : dict mapping source-column value → weight.

    Used by the XGB bag (via train_ens(sample_weight=...)) to up- or
    down-weight rows from different label sources (e.g. experimental vs
    DFT). GP path is untouched (sklearn GPR does not natively support
    sample weights, so a warning is logged at the call site).
    """
    n=len(df)
    if not source_col or source_col not in df.columns:
        return np.ones(n, dtype=np.float64), {}
    sw=source_weights or {}
    vals=df[source_col].astype(str).values
    out=np.array([float(sw.get(v, 1.0)) for v in vals], dtype=np.float64)
    counts=Counter(vals)
    summary={"col":source_col,
             "n_rows":int(n),
             "value_counts":{k:int(v) for k,v in counts.items()},
             "weights_applied":{k:float(sw.get(k, 1.0)) for k in counts}}
    return out, summary

# --- (f) Stacked ensemble (XGB bag → ridge meta-learner) ---
class StackedEnsemble:
    """One-level stacking surrogate.

    Inner: a list of base XGB models (the v15 bag).
    Meta : sklearn Ridge regressing y on the base predictions.

    At inference, base models produce a (n, N_base) matrix and the meta
    model produces a single μ. σ is the across-bag std (preserved from
    v15). Stacking improves μ without inflating σ. Compatible with the
    rest of the pipeline via `pred_unc_stacked`.
    """
    def __init__(self, base_models, meta_alpha=1.0):
        from sklearn.linear_model import Ridge
        self.base=list(base_models)
        self.meta=Ridge(alpha=meta_alpha, fit_intercept=True)
        self._fitted_meta=False
    def fit_meta(self, X_scaled, y, sample_weight=None):
        P=np.empty((X_scaled.shape[0], len(self.base)), dtype=np.float64)
        for i,m in enumerate(self.base):
            P[:,i]=m.predict(X_scaled)
        if sample_weight is not None:
            self.meta.fit(P, y, sample_weight=sample_weight)
        else:
            self.meta.fit(P, y)
        self._fitted_meta=True
        return self
    def predict_mu_sig(self, X_scaled):
        P=np.empty((X_scaled.shape[0], len(self.base)), dtype=np.float64)
        for i,m in enumerate(self.base):
            P[:,i]=m.predict(X_scaled)
        if self._fitted_meta:
            mu=self.meta.predict(P)
        else:
            mu=P.mean(axis=1)
        sig=P.std(axis=1)
        return mu.astype(np.float64), sig.astype(np.float64)

def train_stacked(X, y, n=20, lb="", mt=None, sample_weight=None,
                  meta_alpha=1.0):
    """Train a StackedEnsemble. Same call signature as `train_ens`.

    Returns (StackedEnsemble, scaler) — the StackedEnsemble plays the
    role of `ens` everywhere; `pred_unc_stacked` is the matching
    prediction function.
    """
    ens, sc = train_ens(X, y, n=n, lb=lb, mt=mt, sample_weight=sample_weight)
    Xs=sc.transform(X)
    stacker=StackedEnsemble(ens, meta_alpha=meta_alpha).fit_meta(
        Xs, y, sample_weight=sample_weight)
    log.info("  Stacked meta-learner fit (ridge α=%.2f) for %s", meta_alpha, lb)
    return stacker, sc

def pred_unc_stacked(stacker, sc, X):
    """Predict (μ, σ) from a StackedEnsemble, drop-in for `pred_unc`."""
    Xs=sc.transform(X)
    return stacker.predict_mu_sig(Xs)

# --- (g) Interpretability reports ---
def interpretability_report(ensembles_dict, desc_names, output_dir,
                            top_k=20, pred_fn=None):
    """Per-target top-feature importance summary.

    For each target's ensemble, aggregates `feature_importances_` across
    the bag (works for plain XGB list AND StackedEnsemble). GPEnsemble
    has no such attribute and is skipped with `available=False`.

    Outputs:
      <output_dir>/interpretability/<target>_top_features.csv
      <output_dir>/interpretability/feature_importance_summary.json

    Returns a dict suitable for embedding in metadata.
    """
    out_dir=os.path.join(output_dir, "interpretability")
    os.makedirs(out_dir, exist_ok=True)
    summary={}
    for lb,(en, _sc, _y) in ensembles_dict.items():
        bases=None
        if isinstance(en, list):
            bases=[m for m in en if hasattr(m, "feature_importances_")]
        elif isinstance(en, StackedEnsemble):
            bases=[m for m in en.base if hasattr(m, "feature_importances_")]
        if not bases:
            summary[lb]={"available":False, "reason":"no_tree_estimators"}
            continue
        try:
            imps=np.stack([m.feature_importances_ for m in bases], axis=0)
        except Exception as _e:
            summary[lb]={"available":False, "reason":f"stack_failed:{_e}"}
            continue
        mean_imp=imps.mean(axis=0); std_imp=imps.std(axis=0)
        order=np.argsort(-mean_imp)
        top_idx=order[:top_k]
        rows=[{"feature":(desc_names[i] if i<len(desc_names) else f"f{i}"),
               "mean_importance":float(mean_imp[i]),
               "std_importance":float(std_imp[i])}
              for i in top_idx]
        safe_lb=lb.replace(' ','_').replace('(','').replace(')','').replace('-','_')
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f"{safe_lb}_top_features.csv"),
            index=False)
        summary[lb]={"available":True,
                     "top_features":[r["feature"] for r in rows[:10]],
                     "n_features":int(len(mean_imp))}
    with open(os.path.join(out_dir, "feature_importance_summary.json"),"w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("  Interpretability report: %d targets",
             sum(1 for v in summary.values() if v.get("available")))
    return summary

# --- (h) Exploitation / Exploration / Control candidate queues ---
def make_candidate_queues(df_scored, n_exploit=30, n_explore=30, n_control=20,
                          ad_hard_threshold=AD_HARD_THRESHOLD):
    """Partition scored candidates into three disjoint queues.

    Exploitation : highest final_score among AD-clean, trust>0 candidates.
                   These are high-confidence "QC me next" recommendations.
    Exploration  : highest novelty_score among AD-clean, trust>0 candidates
                   not already in Exploitation. Drives diversity of
                   subsequent training data.
    Control      : highest parent_tanimoto_core among AD-clean, trust>0
                   candidates not already in Exploitation/Exploration.
                   Sanity check; QC disagreement on the control set
                   indicates the surrogate is mis-calibrated even on
                   familiar chemistry.

    A 'queue' column is added (values: 'exploit', 'explore', 'control', or
    '' for ineligible / leftover). Assignment is unique.
    """
    df=df_scored.copy()
    df["queue"]=""
    if "AD_score" not in df.columns: return df
    trust_p=df["trust_parent"] if "trust_parent" in df.columns else pd.Series(1.0, index=df.index)
    trust_t=df["trust_train"]  if "trust_train"  in df.columns else pd.Series(1.0, index=df.index)
    eligible=df[(df["AD_score"]>=ad_hard_threshold)
                & (trust_p>0) & (trust_t>0)].copy()
    if len(eligible)==0: return df
    # Exploitation
    if "final_score" in eligible.columns:
        exploit=eligible.sort_values("final_score", ascending=False).head(n_exploit)
        df.loc[exploit.index, "queue"]="exploit"
        rest=eligible.drop(exploit.index)
    else:
        rest=eligible
    # Exploration
    if len(rest)>0 and "novelty_score" in rest.columns:
        explore=rest.sort_values("novelty_score", ascending=False).head(n_explore)
        df.loc[explore.index, "queue"]="explore"
        rest=rest.drop(explore.index)
    # Control
    if len(rest)>0 and "parent_tanimoto_core" in rest.columns:
        control=rest.sort_values("parent_tanimoto_core", ascending=False).head(n_control)
        df.loc[control.index, "queue"]="control"
    return df

# --- v17 (#50): stable CSV output schema --------------------------------
# `score_cands` plus the post-score gating steps (`apply_strict_ad_gate`,
# `make_candidate_queues`, `div_select`) build up the candidate DataFrame
# incrementally, so the column count and order can drift if any step is
# skipped (e.g. AD_score absent → no AD_hard_rejected; queue gate empty
# → no `queue` column). Downstream tooling that joins on column index or
# expects a fixed header would silently break. The list below is the
# canonical schema for the `all_scored.csv` / `ranked_candidates.csv` /
# `top_diverse_candidates.csv` / `top50_candidates.csv` / `queue_*.csv`
# family of outputs. `_stable_csv_columns` reindexes the DataFrame to this
# schema (filling missing columns with NaN) and appends any post-hoc
# extras (e.g. `qc_*` from QC reconciliation) AFTER the canonical block,
# preserving their relative order. Callers writing CSVs route their
# DataFrame through this helper so the on-disk header is identical
# across runs and code paths.
_SCORED_CSV_SCHEMA=[
    "smiles",
    "pred_DeltaEST","unc_DeltaEST","pred_T2_T1","unc_T2_T1",
    "pred_T1S1_SOC","unc_T1S1_SOC","pred_T2S1_SOC","unc_T2S1_SOC",
    "pred_OscStr","unc_OscStr","pred_Singlets","unc_Singlets",
    "MR_quality","novelty_score",
    "subst_topo_novelty","core_topo_novelty",
    "sym_pattern_novelty","AD_score",
    "trust_parent","trust_train",
    "nearest_train_tanimoto_whole","nearest_train_tanimoto_core",
    "nearest_seed_tanimoto_whole","nearest_seed_tanimoto_core",
    "parent_tanimoto_whole","parent_tanimoto_core",
    "core_fp_valid","core_invalid_risk",
    "core_n_donor","core_n_acceptor","core_sym_index",
    "core_size","core_n_perim_expand","core_n_overlap",
    "core_n_boron","core_is_rim_extended",
    "n_Cz_pendants","n_DBhet_pendants","heavy_atom_z4sum",
    "core_topology_distance_to_parent",
    "core_topology_distance_to_seed_family",
    "subst_topo_dist_parent","subst_topo_dist_seed_family",
    "subst_topo_dist_shortlist",
    "parent_seed_smiles","parent_scaffold_type",
    "subst_topo_sig","sym_pattern_sig",
    "edit_depth","mutation_family",
    "log_TADF_FoM","EI_acquisition",
    "final_score","score_no_novelty","score_no_trust",
    "validation_priority_score",
    # post-score gating columns
    "AD_hard_rejected","queue",
    # diversity / validation tier (added downstream of dfs)
    "diversity_selected","validation_tier",
]
def _stable_csv_columns(df, schema=_SCORED_CSV_SCHEMA):
    """Return df reindexed onto `schema` (missing→NaN), then any extra
    columns (not in schema) appended in their original order. The output
    header is a deterministic function of `schema` plus the set of post-
    hoc additions, regardless of which optional gates fired."""
    out=df.copy()
    for c in schema:
        if c not in out.columns: out[c]=pd.NA
    extras=[c for c in df.columns if c not in schema]
    return out[list(schema)+extras]

# --- (i) Stricter applicability-domain gating ---
def apply_strict_ad_gate(df, ad_hard_threshold=AD_HARD_THRESHOLD,
                         ad_soft_threshold=AD_SOFT_PENALTY):
    """Hard-reject candidates with AD_score < ad_hard_threshold, and apply
    a multiplicative trust penalty in the soft band
    [ad_hard_threshold, ad_soft_threshold].

    The hard rejection is implemented by zeroing trust_parent (the existing
    rank_cands gate `trust_parent>0` then drops them naturally — preserving
    the legacy ranking schema). An 'AD_hard_rejected' flag column is added
    so post-hoc auditing can distinguish AD-rejections from drift-rejections.
    """
    if "AD_score" not in df.columns: return df
    out=df.copy()
    if "trust_parent" not in out.columns:
        return out
    hard_mask=out["AD_score"]<ad_hard_threshold
    out.loc[hard_mask, "trust_parent"]=0.0
    soft_mask=(out["AD_score"]>=ad_hard_threshold) & (out["AD_score"]<ad_soft_threshold)
    out.loc[soft_mask, "trust_parent"]=out.loc[soft_mask, "trust_parent"]*0.5
    out["AD_hard_rejected"]=hard_mask.astype(int)
    return out

# ═══════════════════════════════════════════════════════════════════
def gen_cands(seeds,nt=5000,nw=16,att=500_000,exp=False,eb=False,ed=False,
             train_smiles=None, enable_annulation=False):
    """Generate candidates. Rejects exact duplicates to both seed and training sets.

    PORTABILITY NOTE (Linux only):
        This function relies on multiprocessing.Pool with a Value/Lock
        passed via `initargs=`. On Linux (default 'fork'), the workers
        inherit the synchronisation primitives via copy-on-write — this
        works. On macOS (default 'spawn' since Python 3.8) and Windows
        (only 'spawn'), `Pool` PICKLES the initargs to send to fresh
        worker processes, and `multiprocessing.Value` / `Lock` are NOT
        picklable — verified in v17 review test ITEM 16. To run this on
        non-Linux you would need to (a) wrap Value/Lock in a
        `multiprocessing.Manager()` proxy, (b) move the counter to the
        worker's local state and aggregate post-hoc, or (c) call
        `mp.set_start_method('fork', force=True)` (macOS only). The
        current pipeline targets Linux compute clusters.
    """
    log.info("Generate (target=%d, workers=%d, annul=%s)",nt,nw,enable_annulation)
    # Build combined InChI set from seeds + training
    all_inchis=set()
    for smi_list in [seeds, train_smiles or []]:
        for s in smi_list:
            m=MolFromSmiles(s)
            if m:
                try:
                    i=MolToInchi(m)
                    if i: all_inchis.add(i)
                except Exception: pass
    log.info("  Dedup InChI set: %d (seed+training)", len(all_inchis))
    ti=frozenset(all_inchis); co=Value('i',0); lo=Lock()
    # Forward the SAScorer reference corpus so spawn-based workers (macOS,
    # Windows) can re-fit it. Linux fork inherits state automatically and
    # the worker re-fit becomes a no-op.
    sa_ref=list(dict.fromkeys(list(seeds)+list(train_smiles or []))) \
           if _GLOBAL_SASCORER._fitted else None
    t0=time.time()
    with Pool(processes=nw,initializer=_pinit,
              initargs=(co,lo,seeds,nt,att,exp,ti,eb,ed,enable_annulation,sa_ref)) as pool:
        res=pool.map(_wgen,list(range(nw)))
    cs=[]; ss=set(); si=set(all_inchis)
    for wl in res:
        for rec in wl:
            sm=rec["smiles"]
            if sm in ss: continue
            m=MolFromSmiles(sm)
            if m is None: continue
            try: i=MolToInchi(m)
            except Exception: i=None
            if i and i in si: continue
            if i: si.add(i)
            ss.add(sm); rec["mol"]=m; cs.append(rec)
            if len(cs)>=nt: break
        if len(cs)>=nt: break
    log.info("  Generated %d in %.1fs",len(cs),time.time()-t0)
    return cs

# Descriptors
_RN=[d[0] for d in Descriptors.descList]; _RC=MolecularDescriptorCalculator(_RN)
def _c1(sn):
    s,dn=sn; m=MolFromSmiles(s)
    if m is None: return None
    try:
        v=_RC.CalcDescriptors(m); rd=dict(zip(_RN,v))
        ef=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024)
        ae=np.zeros(1024,dtype=np.int8); DataStructs.ConvertToNumpyArray(ef,ae)
        sf=RDKFingerprint(m,fpSize=1024)
        asf=np.zeros(1024,dtype=np.int8); DataStructs.ConvertToNumpyArray(sf,asf)
        gf=AllChem.GetMorganFingerprintAsBitVect(m,3,nBits=1024)
        ag=np.zeros(1024,dtype=np.int8); DataStructs.ConvertToNumpyArray(gf,ag)
        o=np.empty(len(dn),dtype=np.float64)
        for k,nm in enumerate(dn):
            if nm.startswith("ExtFP"): i=int(nm[5:]); o[k]=ae[i] if i<1024 else np.nan
            elif nm.startswith("GraphFP"): i=int(nm[7:]); o[k]=ag[i] if i<1024 else np.nan
            elif nm.startswith("FP") and not nm.startswith("FpDensity"): i=int(nm[2:]); o[k]=asf[i] if i<1024 else np.nan
            elif nm in rd: v2=rd[nm]; o[k]=float(v2) if v2 is not None else np.nan
            else: o[k]=np.nan
        return o
    except Exception: return None
def comp_desc(cands,dn,nw=16):
    log.info("Descriptors for %d mol",len(cands))
    t0=time.time(); tasks=[(c["smiles"],dn) for c in cands]
    cs=max(32,len(tasks)//(nw*4)); vi=[]; va=[]
    with Pool(processes=nw) as pool:
        for i,r in enumerate(pool.imap(_c1,tasks,chunksize=cs)):
            if r is not None: vi.append(i); va.append(r)
    if not va: return np.empty((0,len(dn))),[],[]
    X=np.vstack(va); vc=[cands[i] for i in vi]; sm=[cands[i]["smiles"] for i in vi]
    log.info("  %d×%d in %.1fs",X.shape[0],X.shape[1],time.time()-t0)
    return X,sm,vc

# ═══════════════════════════════════════════════════════════════════
#  SCORING (Part 1: integrated final score)
# ═══════════════════════════════════════════════════════════════════
def score_cands(Xc,cands,smiles,ed,sd,et,st,e1,s1,e2,s2,ef,sf,esg,ssg,
                yb,tc,sc,adsc,Xsc,nov_mode,
                kappa=None, objective="dEST", energy_units="eV", soc_units="cm-1",
                pred_fn=None, fom_best=None,
                residual_cov=None, kappa_by_family=None):
    """Score candidates with constrained Bayesian optimisation acquisition.

    objective:
      "dEST"     — single-objective Expected Improvement on ΔE_ST
                    (legacy v12 behaviour).
      "TADF_FoM" — multi-objective scalarised EI on the physics-based TADF
                    figure-of-merit log10(fOSC) + 2·log10(SOC) − 2·log10(ΔE_ST).
                    fom_best is the current incumbent (max of training FoM).
    kappa : dict[str → float] of conformal scale factors keyed by target name.
            If supplied, σ is multiplied by κ before being used in EI/feasibility.
    pred_fn: prediction callable taking (ens, sc, X) → (μ, σ). Defaults to
             `pred_unc` (bagged-XGB); replaced by `pred_unc_gpr` when --use_gpr.

    v16.1 additions:
      residual_cov : optional 4×4 covariance matrix of training residuals over
                     the FoM-contributing targets in the order
                     [DeltaEST, T1-S1(SOC), T2-S1(SOC), OscStr] in the user's
                     declared units. When provided, the TADF_FoM Monte-Carlo
                     sampler draws CORRELATED noise via the Cholesky factor of
                     the implied residual correlation matrix instead of
                     independent normals. Tighter, more honest FoM EI.
      kappa_by_family : optional dict[scaffold_family → dict[target → κ]].
                     When provided, the per-target σ multiplier is selected
                     per-candidate from the candidate's parent_scaffold (with
                     fallback to the global scalar `kappa` for novel families).
    """
    log.info("Scoring %d candidates [objective=%s]",len(Xc),objective); t0=time.time()
    if pred_fn is None: pred_fn = pred_unc
    md,sd_=pred_fn(ed,sd,Xc); mt,st_=pred_fn(et,st,Xc)
    ms1,ss1=pred_fn(e1,s1,Xc); ms2,ss2=pred_fn(e2,s2,Xc)
    mf,sf_=pred_fn(ef,sf,Xc); msg,ssg_=pred_fn(esg,ssg,Xc)
    # ── v16.1: scaffold-aware (or scalar) conformal κ ─────────────────
    # When kappa_by_family is supplied, σ is rescaled per-candidate using
    # the parent scaffold's κ; if the family is missing, fall back to the
    # global scalar κ from `kappa`. When only `kappa` is supplied (legacy),
    # apply uniformly. When neither is supplied, σ is unchanged.
    if kappa or kappa_by_family:
        fams=[(c.get("parent_scaffold","") or SF.OT) for c in cands]
        kbf=kappa_by_family or {}
        kg =kappa or {}
        def _kvec(tgt):
            return np.array([float(kbf.get(f,{}).get(tgt, kg.get(tgt, 1.0)))
                              for f in fams], dtype=np.float64)
        sd_  = sd_  * _kvec("DeltaEST")
        st_  = st_  * _kvec("T2-T1")
        ss1  = ss1  * _kvec("T1-S1(SOC)")
        ss2  = ss2  * _kvec("T2-S1(SOC)")
        sf_  = sf_  * _kvec("OscStr")
        ssg_ = ssg_ * _kvec("Singlets")
    for s in [sd_,st_,ss1,ss2,sf_,ssg_]: s[:]=np.maximum(s,1e-8)

    # ── Acquisition: single-obj EI(ΔE_ST) or multi-obj EI(TADF_FoM) ──
    if objective=="TADF_FoM":
        # Sample-based EI: Monte-Carlo over Gaussian posteriors of the four
        # contributing predictions. The FoM is non-linear in (μ_i, σ_i), so
        # closed-form EI is unavailable; MC with N=64 samples is sufficient.
        n_mc=64; rng=np.random.RandomState(SEED)
        # Convert μ/σ for ΔE_ST and SOCs to internal units once.
        mu_d = md * _EV_FROM[energy_units];  sg_d = sd_ * _EV_FROM[energy_units]
        mu_s1= ms1 * _SOC_FROM[soc_units];   sg_s1= ss1 * _SOC_FROM[soc_units]
        mu_s2= ms2 * _SOC_FROM[soc_units];   sg_s2= ss2 * _SOC_FROM[soc_units]
        N=len(md)
        # ── v16.1: correlated noise via residual covariance ──
        # Order: [DeltaEST, T1-S1(SOC), T2-S1(SOC), OscStr]. residual_cov is
        # in user-supplied units; we convert to internal (eV, cm-1) by
        # element-wise scaling Σ_int = S Σ_user S where S is diag of
        # unit-conversion factors. We use the implied correlation matrix
        # only — magnitudes come from per-row σ from the surrogate.
        zd=z1=z2=zf=None
        if residual_cov is not None:
            try:
                rc=np.asarray(residual_cov, dtype=np.float64)
                if rc.shape==(4,4):
                    scale=np.array([_EV_FROM[energy_units],
                                    _SOC_FROM[soc_units],
                                    _SOC_FROM[soc_units], 1.0])
                    cov_int=rc*np.outer(scale, scale)
                    diag=np.maximum(np.diag(cov_int),1e-12)
                    rho=cov_int/np.sqrt(np.outer(diag,diag))
                    np.fill_diagonal(rho, 1.0)
                    L=np.linalg.cholesky(rho + 1e-6*np.eye(4))
                    eps=rng.normal(size=(n_mc, N, 4))
                    eps=np.einsum('mnk,jk->mnj', eps, L)
                    zd=eps[:,:,0]; z1=eps[:,:,1]
                    z2=eps[:,:,2]; zf=eps[:,:,3]
            except Exception as _e:
                log.warning("  residual_cov MC failed (%s); falling back to "
                            "independent sampling", _e)
                zd=z1=z2=zf=None
        if zd is None:
            zd=rng.normal(size=(n_mc,N)); z1=rng.normal(size=(n_mc,N))
            z2=rng.normal(size=(n_mc,N)); zf=rng.normal(size=(n_mc,N))
        fom_samples=np.empty((n_mc,N))
        for k in range(n_mc):
            # Per-quantity floors (v17): _FOM_FLOOR_DEST = 1 µeV gives
            # the inverted-singlet target adequate ranking discrimination.
            d_s = np.maximum(mu_d  + sg_d * zd[k],  _FOM_FLOOR_DEST)
            s1_s= np.maximum(mu_s1 + sg_s1* z1[k],  _FOM_FLOOR_SOC)
            s2_s= np.maximum(mu_s2 + sg_s2* z2[k],  _FOM_FLOOR_SOC)
            f_s = np.maximum(mf    + sf_  * zf[k],  _FOM_FLOOR_FOSC)
            soc_eff=np.maximum(s1_s, s2_s)
            fom_samples[k]=np.log10(f_s) + 2.0*np.log10(soc_eff) - 2.0*np.log10(d_s)
        if fom_best is None: fom_best=float(np.median(fom_samples.max(axis=0)))
        improvement=np.maximum(fom_samples - fom_best, 0.0)
        ei=improvement.mean(axis=0)
    else:
        # Standard minimisation EI on ΔE_ST.
        imp=yb-md-0.01; Z=imp/sd_
        ei=np.maximum(imp*norm.cdf(Z)+sd_*norm.pdf(Z),0.0)
    # Feasibility probabilities (always applied as multiplicative factors).
    pt=norm.cdf((T2_T1_CONSTRAINT-mt)/st_)
    p1=1.0-norm.cdf((SOC1_MIN-ms1)/ss1); p2=1.0-norm.cdf((SOC2_MIN-ms2)/ss2)
    pf=1.0-norm.cdf((FOSC_MIN-mf)/sf_)
    Xs=Xsc.transform(Xc); n=len(smiles)
    mrq=np.ones(n); nov=np.zeros(n); ad=np.ones(n)
    trp=np.ones(n); trt=np.ones(n); cfv_arr=np.zeros(n)
    # Output columns
    out={k:np.zeros(n) for k in ["nttw","nttc","nstw","nstc","ptw","ptc","ns","stn","spn",
                                    "stdp","stdsf","stdsl","ctdsf","ctn"]}
    ps=[""]*n; psc=[""]*n; esig=[""]*n; edep=np.zeros(n,dtype=int); mfam=[""]*n
    spsig=[""]*n; cir=np.zeros(n)
    ctd_nd=np.zeros(n); ctd_na=np.zeros(n); ctd_sym=np.zeros(n)
    ctd_csz=np.zeros(n); ctd_pe=np.zeros(n); ctd_ov=np.zeros(n); ctd_dist_par=np.zeros(n)
    # v14 additions
    ctd_nB=np.zeros(n,dtype=int); ctd_rim=np.zeros(n,dtype=bool)
    ctd_cz=np.zeros(n,dtype=int); ctd_dbhet=np.zeros(n,dtype=int)
    ctd_z4=np.zeros(n)
    seen_sigs=[]
    _pfpc={}; _pctd={}
    for s in set(c.get("parent_smi","") for c in cands):
        if s and s not in _pfpc:
            m=MolFromSmiles(s)
            if m:
                w=_mfp(m); co=_get_core(m); cf,v=_cfp(m,co); _pfpc[s]=(w,cf,v)
                rl=_label_sites(m,co); _pctd[s]=core_topo_desc(m,co,rl)
    for i in range(n):
        rec=cands[i] if i<len(cands) else {}; smi=smiles[i]; mol=MolFromSmiles(smi)
        if mol is None: mrq[i]=0.01; ad[i]=0.01; continue
        mrq[i]=_mr_quality(mol); core=_get_core(mol); wfp=_mfp(mol)
        cfp,cv=_cfp(mol,core); cfv_arr[i]=float(cv)
        if not cv:
            if nov_mode=="conservative": trp[i]=0.0; continue
            elif nov_mode=="balanced":
                mf_=rec.get("mut_family","")
                if mf_ in ("aza","diaza","F+aza"): trp[i]=0.0; continue
                mrq[i]*=0.3; cir[i]=1.0
            else: cir[i]=1.0
        try: inchi=MolToInchi(mol)
        except Exception: inchi=None
        psmi=rec.get("parent_smi",""); ps[i]=psmi; psc[i]=rec.get("parent_scaffold","")
        esig[i]=rec.get("edit_sig",""); edep[i]=rec.get("edit_depth",0); mfam[i]=rec.get("mut_family","")
        spsig[i]=sym_pattern_sig(esig[i])
        pfp=pcfp=None
        if psmi in _pfpc: pfp,pcfp,_=_pfpc[psmi]
        scaff=psc[i] if psc[i] else SF.OT
        roles=_label_sites(mol,core,scaff)
        ctd=core_topo_desc(mol,core,roles)
        ctd_nd[i]=ctd["n_donor"]; ctd_na[i]=ctd["n_acceptor"]; ctd_sym[i]=ctd["sym_index"]
        ctd_csz[i]=ctd["core_size"]; ctd_pe[i]=ctd["n_perim_expand"]; ctd_ov[i]=ctd["n_overlap"]
        ctd_nB[i]=ctd.get("n_boron_in_core",0)
        ctd_rim[i]=ctd.get("is_rim_extended",False)
        ctd_cz[i]=ctd.get("n_Cz_pendants",0)
        ctd_dbhet[i]=ctd.get("n_DBhet_pendants",0)
        ctd_z4[i]=ctd.get("heavy_atom_z4sum",0.0)
        par_ctd=_pctd.get(psmi) if psmi else None
        nv=compute_novelty(wfp,cfp,cv,inchi,tc,sc,pfp,pcfp,esig[i],scaff,
                           seen_sigs=seen_sigs[-200:] if seen_sigs else None,
                           cand_ctd=ctd, parent_ctd=par_ctd)
        nov[i]=nv["novelty_score"]; out["nttw"][i]=nv["nearest_train_tanimoto_whole"]
        out["nttc"][i]=nv["nearest_train_tanimoto_core"]; out["nstw"][i]=nv["nearest_seed_tanimoto_whole"]
        out["nstc"][i]=nv["nearest_seed_tanimoto_core"]; out["ptw"][i]=nv["parent_tanimoto_whole"]
        out["ptc"][i]=nv["parent_tanimoto_core"]; out["stn"][i]=nv["subst_topo_nov"]
        out["spn"][i]=nv["sym_pattern_nov"]; out["ns"][i]=nv["novelty_score"]
        out["stdp"][i]=nv["subst_topo_dist_parent"]
        out["stdsf"][i]=nv["subst_topo_dist_seed_family"]  # REAL subst-topo distance
        out["stdsl"][i]=nv["subst_topo_dist_shortlist"]
        out["ctdsf"][i]=nv["core_topo_dist_seed_family"]
        out["ctn"][i]=nv["core_topo_nov"]
        seen_sigs.append(esig[i])
        ok1,p1_=_trust_parent(out["ptw"][i],out["ptc"][i],scaff,cv); trp[i]=p1_ if ok1 else 0.0
        ok2,p2_=_trust_train(out["nttw"][i],out["nttc"][i],cv); trt[i]=p2_ if ok2 else 0.0
        ad[i]=adsc.score(Xs[i],wfp,cfp,cv,(sd_[i]+st_[i])/2)
        if par_ctd:
            ctd_dist_par[i]=nv["core_topo_dist_parent"]
            da_shift=abs(ctd["n_donor"]-par_ctd["n_donor"])+abs(ctd["n_acceptor"]-par_ctd["n_acceptor"])
            if da_shift>2: trp[i]*=0.5
            elif da_shift>0: trp[i]*=0.85
    # ── Diagnostic: log10(predicted TADF FoM) per candidate ──
    # v17: per-quantity floors + non-negative clamp (per the FoM contract).
    # mf, ms1, ms2 are surrogate point predictions; clamping silently
    # corrects out-of-distribution negative predictions while keeping
    # ranking well-defined.
    log_fom = (np.log10(np.maximum(np.clip(mf,0.0,None),
                                   _FOM_FLOOR_FOSC))
               + 2.0*np.log10(np.maximum(np.maximum(np.clip(ms1,0.0,None),
                                                    np.clip(ms2,0.0,None))
                                         *_SOC_FROM[soc_units],
                                         _FOM_FLOOR_SOC))
               - 2.0*np.log10(np.maximum(np.clip(md,0.0,None)
                                         *_EV_FROM[energy_units],
                                         _FOM_FLOOR_DEST)))
    # INTEGRATED FINAL SCORE
    # Magic exponents below are empirically tuned; ranking is sensitive
    # to all six (Spearman ρ vs the reference ranking drops below 0.7
    # under independent ±50% perturbations to ANY single exponent —
    # measured in the v17 review test ITEM 12). They are NOT calibrated
    # against held-out experimental TADF rate data; expect 1-2 rank
    # positions of jitter from this source. If you re-tune, retain the
    # ablation columns `score_no_novelty` / `score_no_trust` below so
    # the relative impact of each factor remains auditable.
    nov_boost=1.0+0.15*np.clip(nov,0,0.7)
    cei=ei*pt*p1*p2*pf
    final=(cei**0.6 * mrq**0.2 * ad**0.15 * trp**0.3 * trt**0.2 * nov_boost**0.1)
    fmax=final.max()
    if fmax>0: final/=fmax
    # Ablation columns (Part 5)
    final_no_nov=(cei**0.6 * mrq**0.2 * ad**0.15 * trp**0.3 * trt**0.2)
    fn=final_no_nov.max(); final_no_nov=final_no_nov/fn if fn>0 else final_no_nov
    final_no_trust=(cei**0.6 * mrq**0.2 * ad**0.15 * nov_boost**0.1)
    fn=final_no_trust.max(); final_no_trust=final_no_trust/fn if fn>0 else final_no_trust
    # Validation priority
    val_pri=(final**0.5 * np.clip(nov,0.01,1.0)**0.3 *
             np.clip(trp,0.01,1.0)**0.2 * np.clip(ad,0.01,1.0)**0.2)
    vmax=val_pri.max()
    if vmax>0: val_pri/=vmax
    log.info("  Done in %.1fs",time.time()-t0)
    return pd.DataFrame({
        "pred_DeltaEST":md,"unc_DeltaEST":sd_,"pred_T2_T1":mt,"unc_T2_T1":st_,
        "pred_T1S1_SOC":ms1,"unc_T1S1_SOC":ss1,"pred_T2S1_SOC":ms2,"unc_T2S1_SOC":ss2,
        "pred_OscStr":mf,"unc_OscStr":sf_,"pred_Singlets":msg,"unc_Singlets":ssg_,
        "MR_quality":mrq,"novelty_score":out["ns"],
        "subst_topo_novelty":out["stn"],"core_topo_novelty":out["ctn"],
        "sym_pattern_novelty":out["spn"],"AD_score":ad,
        "trust_parent":trp,"trust_train":trt,
        "nearest_train_tanimoto_whole":out["nttw"],"nearest_train_tanimoto_core":out["nttc"],
        "nearest_seed_tanimoto_whole":out["nstw"],"nearest_seed_tanimoto_core":out["nstc"],
        "parent_tanimoto_whole":out["ptw"],"parent_tanimoto_core":out["ptc"],
        "core_fp_valid":cfv_arr,"core_invalid_risk":cir,
        "core_n_donor":ctd_nd,"core_n_acceptor":ctd_na,"core_sym_index":ctd_sym,
        "core_size":ctd_csz,"core_n_perim_expand":ctd_pe,"core_n_overlap":ctd_ov,
        # v14: pattern-derived structural annotations
        "core_n_boron":ctd_nB,
        "core_is_rim_extended":ctd_rim,
        "n_Cz_pendants":ctd_cz,
        "n_DBhet_pendants":ctd_dbhet,
        "heavy_atom_z4sum":ctd_z4,
        "core_topology_distance_to_parent":ctd_dist_par,
        "core_topology_distance_to_seed_family":out["ctdsf"],
        "subst_topo_dist_parent":out["stdp"],
        "subst_topo_dist_seed_family":out["stdsf"],
        "subst_topo_dist_shortlist":out["stdsl"],
        "parent_seed_smiles":ps,"parent_scaffold_type":psc,
        "subst_topo_sig":esig,"sym_pattern_sig":spsig,
        "edit_depth":edep,"mutation_family":mfam,
        "log_TADF_FoM":log_fom,
        "EI_acquisition":ei,
        "final_score":final,"score_no_novelty":final_no_nov,
        "score_no_trust":final_no_trust,
        "validation_priority_score":val_pri,
    })

# ═══════════════════════════════════════════════════════════════════
#  PART 1: RANKING BY INTEGRATED SCORE
# ═══════════════════════════════════════════════════════════════════
def rank_cands(df):
    df_f=df[(df["unc_DeltaEST"]<UNCERTAINTY_THRESHOLD)&
            (df["pred_T2_T1"]<T2_T1_CONSTRAINT)&
            (df["pred_DeltaEST"]<DEST_CONSTRAINT)&
            (df["trust_parent"]>0)&(df["trust_train"]>0)].copy()
    if len(df_f)==0:
        log.warning("  Relaxing"); df_f=df[df["unc_DeltaEST"]<UNCERTAINTY_THRESHOLD*3].copy()
    df_f.sort_values("final_score",ascending=False,inplace=True)
    df_f.reset_index(drop=True,inplace=True); df_f.index.name="Rank"; df_f.index+=1
    return df_f

# ═══════════════════════════════════════════════════════════════════
#  PART 9: PARENT-AWARE FAMILY-AWARE DIVERSITY
# ═══════════════════════════════════════════════════════════════════
def div_select(df,top_n=50,lam=0.3,ww=0.35,wc=0.20,we=0.15,wp=0.15,wct=0.15,
               max_per_parent=10,max_per_family=30):
    """Multi-view diversity with core-topology distance.

    v17 bug fix: the per-parent and per-family caps are now respected
    even when len(df) ≤ top_n. v16 took an early-return path that
    set every candidate to `diversity_selected=True`, silently
    bypassing the user-supplied caps and allowing concentrated parents
    or families to dominate the shortlist (verified in v17 review
    test C-14).
    """
    if len(df)<=top_n:
        # Apply caps even when the full pool would otherwise be selected.
        df=df.copy()
        df["diversity_selected"]=False
        if "parent_seed_smiles" in df.columns and "parent_scaffold_type" in df.columns:
            par_count=Counter(); fam_count=Counter()
            sort_col="final_score" if "final_score" in df.columns else df.columns[0]
            order=df.sort_values(sort_col, ascending=False).index
            for idx in order:
                p=df.at[idx,"parent_seed_smiles"]; f=df.at[idx,"parent_scaffold_type"]
                if par_count[p]>=max_per_parent or fam_count[f]>=max_per_family:
                    continue
                df.at[idx,"diversity_selected"]=True
                par_count[p]+=1; fam_count[f]+=1
        else:
            df["diversity_selected"]=True
        return df
    log.info("  Diversity: %d→%d (λ=%.2f)",len(df),top_n,lam)
    fps=[]; cfps=[]; ctds=[]
    for _,row in df.iterrows():
        m=MolFromSmiles(row["smiles"])
        fps.append(_mfp(m) if m else None)
        if m:
            c=_get_core(m); cf,_=_cfp(m,c); cfps.append(cf)
            rl=_label_sites(m,c); ctds.append(core_topo_desc(m,c,rl))
        else: cfps.append(None); ctds.append(None)
    esigs=df["subst_topo_sig"].tolist()
    pcols=["pred_DeltaEST","pred_T2_T1","pred_OscStr"]
    pvals=df[pcols].values.astype(float)
    pmn=pvals.min(0); pmx=pvals.max(0); prng=np.maximum(pmx-pmn,1e-12)
    pnorm=(pvals-pmn)/prng
    parents=df["parent_seed_smiles"].tolist()
    families=df["parent_scaffold_type"].tolist()

    quality=df["final_score"].values; quality=quality/max(quality.max(),1e-12)
    selected=[0]; remaining=set(range(1,len(df)))
    par_count=Counter({parents[0]:1}); fam_count=Counter({families[0]:1})
    for _ in range(top_n-1):
        if not remaining: break
        best_i,best_s=-1,-1.0
        for ci in remaining:
            if fps[ci] is None: continue
            if par_count.get(parents[ci],0)>=max_per_parent: continue
            if fam_count.get(families[ci],0)>=max_per_family: continue
            min_d=1.0
            for si in selected:
                if fps[si] is None: continue
                dw=1.0-DataStructs.TanimotoSimilarity(fps[ci],fps[si])
                dc=1.0-(DataStructs.TanimotoSimilarity(cfps[ci],cfps[si]) if cfps[ci] and cfps[si] else 0.0)
                de=subst_topo_novelty(esigs[ci],esigs[si])
                dprop=np.linalg.norm(pnorm[ci]-pnorm[si])/(len(pcols)**0.5)
                dct=core_topo_distance(ctds[ci],ctds[si]) if (ctds[ci] and ctds[si]) else 0.0
                dt=ww*dw+wc*dc+we*de+wp*dprop+wct*dct
                min_d=min(min_d,dt)
            sc=(1.0-lam)*quality[ci]+lam*min_d
            if sc>best_s: best_s=sc; best_i=ci
        if best_i<0: break
        selected.append(best_i); remaining.discard(best_i)
        par_count[parents[best_i]]+=1; fam_count[families[best_i]]+=1
    df["diversity_selected"]=False
    for si in selected: df.iloc[si,df.columns.get_loc("diversity_selected")]=True
    return df

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def _cgroup_cpu_limit():
    """Return the effective CPU quota imposed by cgroups (containers,
    Slurm, k8s), or None if unbounded / not in a cgroup.

    Checks cgroups v2 (`/sys/fs/cgroup/cpu.max`, single file with
    "<quota> <period>") and falls back to cgroups v1
    (`cpu.cfs_quota_us` + `cpu.cfs_period_us`). Both express the limit
    in microseconds per period; effective CPU count = quota / period.
    """
    # cgroups v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            parts = fh.read().split()
        if len(parts) >= 2 and parts[0] != "max":
            q = int(parts[0]); p = int(parts[1])
            if q > 0 and p > 0:
                return max(1, q // p)
    except Exception:
        pass
    # cgroups v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            q = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            p = int(fh.read().strip())
        if q > 0 and p > 0:
            return max(1, q // p)
    except Exception:
        pass
    return None

def _pc():
    """Effective process CPU count.

    Order:
      1. cgroup CPU quota (containers, Slurm, k8s) — when set, return
         the smaller of (physical core count, quota).
      2. psutil physical-core count.
      3. /proc/cpuinfo physical-id / core-id parsing.
      4. mp.cpu_count() fallback.
    """
    base = None
    try:
        import psutil
        base = psutil.cpu_count(logical=False) or mp.cpu_count()
    except Exception:
        try:
            with open("/proc/cpuinfo") as f:
                cs=set(); p=c=None
                for l in f:
                    if l.startswith("physical id"): p=l.split(":")[1].strip()
                    elif l.startswith("core id"): c=l.split(":")[1].strip()
                    elif l.strip()=="" and p: cs.add((p,c)); p=c=None
                base = len(cs) if cs else mp.cpu_count()
        except Exception:
            base = mp.cpu_count()
    quota = _cgroup_cpu_limit()
    if quota is not None and quota < base:
        return quota
    return base

def main():
    global MAX_WHOLE_SIM_TRAIN,MIN_CORE_SIM_TRAIN,MIN_CORE_SIM_PARENT
    global BAND_TRAIN_WHOLE_CTR,BAND_PARENT_CORE_CTR
    ap=argparse.ArgumentParser(description="MR-TADF Pipeline v16 — Deep Generative Augmentation + Hard Chemical Gates")
    ap.add_argument("--data",default="updated_data.xlsx")
    ap.add_argument("--target",default="target5.xlsx")
    ap.add_argument("--seed_file",default=None,help="Separate seed SMILES file (.txt or .xlsx)")
    ap.add_argument("--benchmark_file",default=None,help="External benchmark SMILES for validation")
    ap.add_argument("--n_candidates",type=int,default=10000)
    ap.add_argument("--n_workers",type=int,default=0)
    ap.add_argument("--attempts_per_worker",type=int,default=500_000)
    ap.add_argument("--n_ensemble",type=int,default=20)
    ap.add_argument("--output",default="results")
    ap.add_argument("--exploratory",action="store_true")
    ap.add_argument("--max_train_threads",type=int,default=0)
    ap.add_argument("--min_core_sim_parent",type=float,default=MIN_CORE_SIM_PARENT)
    ap.add_argument("--min_core_sim_train",type=float,default=MIN_CORE_SIM_TRAIN)
    ap.add_argument("--max_whole_sim_train",type=float,default=MAX_WHOLE_SIM_TRAIN)
    ap.add_argument("--diversity_lambda",type=float,default=0.3)
    ap.add_argument("--top_diverse_n",type=int,default=50)
    ap.add_argument("--max_per_parent",type=int,default=10)
    ap.add_argument("--max_per_scaffold_family",type=int,default=30)
    ap.add_argument("--enable_bulky",action="store_true")
    ap.add_argument("--enable_diaza",action="store_true")
    ap.add_argument("--enable_annulation",action="store_true",
                    help="Activate the real benzannulation operator (v13: implemented).")
    ap.add_argument("--qc_results_file",default=None,
                    help="CSV with QC results to merge (must have 'smiles' column)")
    ap.add_argument("--novelty_mode",choices=["conservative","balanced","exploratory"],default="balanced")
    # ── v13 NEW FLAGS ───────────────────────────────────────────────
    ap.add_argument("--objective",choices=["dEST","TADF_FoM"],default="TADF_FoM",
                    help="Acquisition objective: single-target ΔE_ST EI (legacy) "
                         "or multi-objective TADF figure-of-merit (default).")
    ap.add_argument("--energy_units",default="eV",
                    help="Units of ΔE_ST/T2-T1 columns in --target file. "
                         "Choices: eV, meV, cm-1, kcal/mol, kJ/mol, hartree.")
    ap.add_argument("--soc_units",default="cm-1",
                    help="Units of SOC columns. Choices: cm-1, meV, eV, hartree.")
    ap.add_argument("--use_gpr",action="store_true",
                    help="Use Gaussian-process surrogate (Matern-5/2 + WhiteKernel) "
                         "instead of bagged-XGB. Recommended for n_train < 500 to "
                         "obtain calibrated posterior σ.")
    ap.add_argument("--calibrate_sigma",action="store_true",
                    help="Hold out 20%% of training data and compute conformal κ "
                         "to scale ensemble σ to honest predictive bands.")
    ap.add_argument("--scaffold_cv",action="store_true",
                    help="Run leave-family-out cross-validation per target and write "
                         "scaffold_stratified_cv.json — measures family-transfer.")
    # ── v16 NEW FLAGS ────────────────────────────────────────────────
    ap.add_argument("--sascore_max",type=float,default=SASCORE_MAX,
                    help="v16: SAscore cap. Reject candidates with SAscore > this.")
    ap.add_argument("--disable_sa_filter",action="store_true",
                    help="v16: skip SAscore gate at hard filter (testing only).")
    ap.add_argument("--disable_retro_filter",action="store_true",
                    help="v16: skip retrosynthesis-feasibility gate (testing only).")
    ap.add_argument("--disable_charge_radical_filter",action="store_true",
                    help="v16: skip charge/radical/open-shell gate (testing only).")
    # v18: --enable_deep_gen and --deep_gen_* flags removed with the
    # deep-generative stack. The CLI is otherwise byte-identical to v17.
    # ── v16.1 NEW FLAGS ──────────────────────────────────────────────
    ap.add_argument("--enable_stacking",action="store_true",
                    help="v16.1: train a stacked surrogate (XGB bag → ridge "
                         "meta-learner) instead of plain bagging. Improves "
                         "μ; σ remains across-bag std.")
    ap.add_argument("--stacking_alpha",type=float,default=1.0,
                    help="v16.1: ridge α for the stacking meta-learner.")
    ap.add_argument("--scaffold_conformal",action="store_true",
                    help="v16.1: per-scaffold-family conformal κ in addition "
                         "to the global κ (requires --calibrate_sigma).")
    ap.add_argument("--enable_residual_cov",action="store_true",
                    help="v16.1: use training residual covariance for "
                         "correlated MC sampling in TADF FoM EI. Requires "
                         "--objective TADF_FoM.")
    ap.add_argument("--label_source_col",default=None,
                    help="v16.1: column in --target file containing the "
                         "label-source identifier (e.g. 'source' or "
                         "'method'). Per-row sample weights are derived "
                         "via --label_source_weights and applied during "
                         "XGB training.")
    ap.add_argument("--label_source_weights",default=None,
                    help="v16.1: JSON dict mapping source-column value → "
                         "weight, e.g. '{\"experimental\":2.0,\"DFT\":1.0}'. "
                         "Missing values default to weight=1.")
    ap.add_argument("--ad_hard_threshold",type=float,default=AD_HARD_THRESHOLD,
                    help="v16.1: hard AD-rejection threshold. Candidates "
                         "with AD_score below this are rejected before "
                         "ranking.")
    ap.add_argument("--ad_soft_threshold",type=float,default=AD_SOFT_PENALTY,
                    help="v16.1: soft AD-penalty threshold. Candidates in "
                         "[hard, soft] receive a 0.5× trust_parent penalty.")
    ap.add_argument("--n_exploit",type=int,default=30,
                    help="v16.1: size of the exploitation queue.")
    ap.add_argument("--n_explore",type=int,default=30,
                    help="v16.1: size of the exploration queue.")
    ap.add_argument("--n_control",type=int,default=20,
                    help="v16.1: size of the control queue.")
    ap.add_argument("--validation_history_file",default=None,
                    help="v16.1: path to persistent JSON validation history. "
                         "Defaults to <output>/validation_history.json. "
                         "Pass an absolute path to share across runs.")
    ap.add_argument("--core_split_cv_min_test",type=int,default=2,
                    help="v16.1: minimum cluster size for Murcko-core "
                         "leave-cluster-out CV.")
    args=ap.parse_args()
    # Validate units up-front to avoid late surprises.
    if args.energy_units not in _EV_FROM:
        sys.exit(f"--energy_units must be one of {list(_EV_FROM)}")
    if args.soc_units not in _SOC_FROM:
        sys.exit(f"--soc_units must be one of {list(_SOC_FROM)}")
    MAX_WHOLE_SIM_TRAIN=args.max_whole_sim_train
    MIN_CORE_SIM_TRAIN=args.min_core_sim_train
    MIN_CORE_SIM_PARENT=args.min_core_sim_parent
    if args.novelty_mode=="conservative":
        MAX_WHOLE_SIM_TRAIN=0.98; MIN_CORE_SIM_PARENT=0.80; BAND_TRAIN_WHOLE_CTR=0.55
    elif args.novelty_mode=="exploratory":
        MAX_WHOLE_SIM_TRAIN=0.90; MIN_CORE_SIM_PARENT=0.60; BAND_TRAIN_WHOLE_CTR=0.75
    ph=_pc(); lo=mp.cpu_count()
    if args.n_workers<=0: args.n_workers=min(ph,128)
    if args.max_train_threads<=0: args.max_train_threads=None
    log.info("="*60); log.info("MR-TADF v16.1 — v16 + Hardened OOF / AD / Stacking / Queues")
    log.info("="*60)
    surr_label=("GPR" if args.use_gpr
                else ("Stacked-XGB+Ridge" if args.enable_stacking
                      else "bagged-XGB"))
    log.info("  Workers:%d  Mode:%s  Objective:%s  Surrogate:%s",
             args.n_workers, args.novelty_mode, args.objective, surr_label)
    log.info("  Units: energy=%s, SOC=%s", args.energy_units, args.soc_units)
    log.info("  Strict AD gate: hard=%.2f  soft=%.2f",
             args.ad_hard_threshold, args.ad_soft_threshold)
    log.info("  Queues: exploit=%d  explore=%d  control=%d",
             args.n_exploit, args.n_explore, args.n_control)
    t0=time.time(); os.makedirs(args.output,exist_ok=True)

    # Data
    (df,X,yd,yt,ys1,ys2,yf,ysg,dn,smi_train,imp)=load_data(args.data,args.target)
    smi_seeds,seeds_separate=load_seed_smiles(args,smi_train)
    tc=FPCache(smi_train,"train"); sc=FPCache(smi_seeds,"seed")

    # ── v16: fit global SAScorer on the training+seed corpus ─────────
    # Skipped only if the user explicitly disabled the SA filter.
    if not args.disable_sa_filter:
        log.info("="*60); log.info("v16: SAScorer fit"); log.info("="*60)
        ref=list(dict.fromkeys(list(smi_train)+list(smi_seeds)))
        _GLOBAL_SASCORER.fit(ref)
        # Sanity: log SAscore stats over the training set so the user can
        # see the distribution and pick a sensible --sascore_max.
        sa_train=[]
        for s in smi_train:
            m=MolFromSmiles(s) if isinstance(s,str) else None
            if m is not None: sa_train.append(_GLOBAL_SASCORER.score(m))
        if sa_train:
            log.info("  SAscore on training set: mean=%.2f  q50=%.2f  "
                     "q90=%.2f  max=%.2f  cap=%.2f",
                     float(np.mean(sa_train)), float(np.median(sa_train)),
                     float(np.quantile(sa_train, 0.90)),
                     float(np.max(sa_train)), float(args.sascore_max))
    # Module-level mirror so worker processes inherit it via fork.
    globals()["SASCORE_MAX"]=float(args.sascore_max)

    # ── v16.1: label-source weights ───────────────────────────────────
    # If --label_source_col is set, compute per-row sample weights from
    # the configured column in the target file. Weights are propagated to
    # XGB training (no-op for GP — sklearn GPR doesn't accept weights).
    label_source_summary=None
    sample_weight_full=None
    if args.label_source_col:
        try:
            _dt_raw=pd.read_excel(args.target)
            _dt_raw.rename(columns={_dt_raw.columns[0]:"Name"}, inplace=True)
            if args.label_source_col in _dt_raw.columns:
                src_map=dict(zip(_dt_raw["Name"].astype(str),
                                 _dt_raw[args.label_source_col].astype(str)))
                df_for_weights=df.copy()
                df_for_weights["__src__"]=df_for_weights["Name"].astype(str).map(src_map)
                _sw_dict={}
                if args.label_source_weights:
                    try:
                        _sw_dict=json.loads(args.label_source_weights)
                    except Exception as _e:
                        log.warning("  Invalid --label_source_weights JSON: %s", _e)
                sample_weight_full, label_source_summary = (
                    compute_label_source_weights(df_for_weights, "__src__", _sw_dict))
                if args.use_gpr:
                    log.warning("  --use_gpr ignores label-source weights "
                                "(sklearn GPR has no sample_weight). Weights "
                                "will only affect XGB stacking pathway.")
                else:
                    log.info("  Label-source weights: col='%s', mean=%.2f, "
                             "range=%.2f-%.2f, distinct=%d",
                             args.label_source_col,
                             float(sample_weight_full.mean()),
                             float(sample_weight_full.min()),
                             float(sample_weight_full.max()),
                             len(set(sample_weight_full)))
            else:
                log.warning("  --label_source_col '%s' not in target file; "
                            "weights skipped.", args.label_source_col)
        except Exception as _e:
            log.warning("  Label-source weighting failed: %s", _e)

    # ── (optional) Hold out a calibration split for conformal σ ──
    X_cal=y_cal_d=y_cal_t=y_cal_1=y_cal_2=y_cal_f=y_cal_g=None
    cal_idx=tr_idx=None
    sw_tr=None
    if args.calibrate_sigma and len(X)>=20:
        rng=np.random.RandomState(SEED)
        perm=rng.permutation(len(X)); ncal=max(5,int(0.2*len(X)))
        cal_idx=perm[:ncal]; tr_idx=perm[ncal:]
        X_cal=X[cal_idx]
        y_cal_d=yd[cal_idx]; y_cal_t=yt[cal_idx]
        y_cal_1=ys1[cal_idx]; y_cal_2=ys2[cal_idx]
        y_cal_f=yf[cal_idx]; y_cal_g=ysg[cal_idx]
        X_tr=X[tr_idx]
        y_tr_d=yd[tr_idx]; y_tr_t=yt[tr_idx]
        y_tr_1=ys1[tr_idx]; y_tr_2=ys2[tr_idx]
        y_tr_f=yf[tr_idx]; y_tr_g=ysg[tr_idx]
        if sample_weight_full is not None:
            sw_tr=sample_weight_full[tr_idx]
        log.info("  Conformal split: %d train / %d calib", len(tr_idx), len(cal_idx))
    else:
        X_tr=X; y_tr_d=yd; y_tr_t=yt; y_tr_1=ys1; y_tr_2=ys2; y_tr_f=yf; y_tr_g=ysg
        if sample_weight_full is not None: sw_tr=sample_weight_full

    # Train surrogates (XGB ensemble OR single GP OR stacked)
    log.info("="*60); log.info("Training surrogates"); log.info("="*60)
    if args.use_gpr:
        ed,sd  =train_gpr(X_tr,y_tr_d,"DeltaEST")
        et,st  =train_gpr(X_tr,y_tr_t,"T2-T1")
        e1,s1  =train_gpr(X_tr,y_tr_1,"T1-S1(SOC)")
        e2,s2  =train_gpr(X_tr,y_tr_2,"T2-S1(SOC)")
        ef,sf  =train_gpr(X_tr,y_tr_f,"OscStr")
        esg,ssg=train_gpr(X_tr,y_tr_g,"Singlets")
        pred_fn=pred_unc_gpr
    elif args.enable_stacking:
        log.info("  Surrogate: StackedEnsemble (XGB bag → ridge α=%.2f)",
                 args.stacking_alpha)
        ed,sd  =train_stacked(X_tr,y_tr_d,args.n_ensemble,"DeltaEST",
                              args.max_train_threads, sample_weight=sw_tr,
                              meta_alpha=args.stacking_alpha)
        et,st  =train_stacked(X_tr,y_tr_t,args.n_ensemble,"T2-T1",
                              args.max_train_threads, sample_weight=sw_tr,
                              meta_alpha=args.stacking_alpha)
        e1,s1  =train_stacked(X_tr,y_tr_1,args.n_ensemble,"T1-S1(SOC)",
                              args.max_train_threads, sample_weight=sw_tr,
                              meta_alpha=args.stacking_alpha)
        e2,s2  =train_stacked(X_tr,y_tr_2,args.n_ensemble,"T2-S1(SOC)",
                              args.max_train_threads, sample_weight=sw_tr,
                              meta_alpha=args.stacking_alpha)
        ef,sf  =train_stacked(X_tr,y_tr_f,args.n_ensemble,"OscStr",
                              args.max_train_threads, sample_weight=sw_tr,
                              meta_alpha=args.stacking_alpha)
        esg,ssg=train_stacked(X_tr,y_tr_g,args.n_ensemble,"Singlets",
                              args.max_train_threads, sample_weight=sw_tr,
                              meta_alpha=args.stacking_alpha)
        pred_fn=pred_unc_stacked
    else:
        ed,sd  =train_ens(X_tr,y_tr_d,args.n_ensemble,"DeltaEST",
                          args.max_train_threads, sample_weight=sw_tr)
        et,st  =train_ens(X_tr,y_tr_t,args.n_ensemble,"T2-T1",
                          args.max_train_threads, sample_weight=sw_tr)
        e1,s1  =train_ens(X_tr,y_tr_1,args.n_ensemble,"T1-S1(SOC)",
                          args.max_train_threads, sample_weight=sw_tr)
        e2,s2  =train_ens(X_tr,y_tr_2,args.n_ensemble,"T2-S1(SOC)",
                          args.max_train_threads, sample_weight=sw_tr)
        ef,sf  =train_ens(X_tr,y_tr_f,args.n_ensemble,"OscStr",
                          args.max_train_threads, sample_weight=sw_tr)
        esg,ssg=train_ens(X_tr,y_tr_g,args.n_ensemble,"Singlets",
                          args.max_train_threads, sample_weight=sw_tr)
        pred_fn=pred_unc
    edict={"DeltaEST":(ed,sd,yd),"T2-T1":(et,st,yt),"T1-S1(SOC)":(e1,s1,ys1),
           "T2-S1(SOC)":(e2,s2,ys2),"OscStr":(ef,sf,yf),"Singlets":(esg,ssg,ysg)}
    for lb,(en,sc_,y) in edict.items():
        mu,_=pred_fn(en,sc_,X)
        log.info("  %s MAE=%.4f R²=%.4f",lb,mean_absolute_error(y,mu),r2_score(y,mu))
    log.info("  Regression plots ...")
    for lb,(en,sc_,y) in edict.items():
        mu,_=pred_fn(en,sc_,X); _plot(y,mu,lb,args.output)
    Xsc=StandardScaler(); Xsc.fit(X); adsc=ADScorer(Xsc.transform(X),tc)

    # ── Conformal κ per target (scalar, legacy) + per-scaffold (v16.1) ──
    kappa={}
    kappa_by_family=None
    if args.calibrate_sigma and X_cal is not None:
        log.info("="*60); log.info("Conformal σ calibration"); log.info("="*60)
        cal_targets={"DeltaEST":(ed,sd,y_cal_d),"T2-T1":(et,st,y_cal_t),
                     "T1-S1(SOC)":(e1,s1,y_cal_1),"T2-S1(SOC)":(e2,s2,y_cal_2),
                     "OscStr":(ef,sf,y_cal_f),"Singlets":(esg,ssg,y_cal_g)}
        smi_cal_list=[smi_train[i] for i in cal_idx] if cal_idx is not None else []
        if args.scaffold_conformal:
            kappa_by_family={}
        for lb,(en,sc_,yy) in cal_targets.items():
            if args.use_gpr:
                # GP already produces calibrated σ; κ ≈ 1 by construction.
                mu,sg=pred_unc_gpr(en,sc_,X_cal)
                z=np.abs(yy-mu)/np.maximum(sg,1e-8)
                kappa[lb]=float(np.quantile(z,0.683)) if len(z)>=5 else 1.0
                if args.scaffold_conformal:
                    fams=scaffold_groups(smi_cal_list)
                    for fam in sorted(set(fams)):
                        idx=[i for i,g in enumerate(fams) if g==fam]
                        if len(idx)>=5:
                            kappa_by_family.setdefault(fam,{})[lb]=float(
                                np.quantile(z[idx],0.683))
                        else:
                            kappa_by_family.setdefault(fam,{})[lb]=kappa[lb]
            else:
                if args.scaffold_conformal:
                    kg, kfam = calibrate_sigma_by_scaffold(
                        en, sc_, X_cal, yy, smi_cal_list, pred_fn=pred_fn)
                    kappa[lb]=kg
                    for fam,kv in kfam.items():
                        kappa_by_family.setdefault(fam,{})[lb]=kv
                else:
                    kappa[lb]=calibrate_sigma(en,sc_,X_cal,yy)
            log.info("  κ[%s] = %.3f", lb, kappa[lb])
        with open(os.path.join(args.output,"conformal_kappa.json"),"w") as fjs:
            json.dump(kappa,fjs,indent=2)
        if kappa_by_family:
            with open(os.path.join(args.output,"conformal_kappa_by_scaffold.json"),"w") as fjs:
                json.dump(kappa_by_family, fjs, indent=2)
            log.info("  Scaffold-aware κ written for %d families",
                     len(kappa_by_family))

    # ── (v16.1) MANDATORY scaffold-stratified CV ──
    log.info("="*60); log.info("Scaffold-stratified CV (leave-family-out)"); log.info("="*60)
    cv_report={}
    for lb,(_,_,yy) in edict.items():
        cv_report[lb]=scaffold_stratified_cv(X,yy,smi_train,label=lb)
    with open(os.path.join(args.output,"scaffold_stratified_cv.json"),"w") as fjs:
        json.dump(cv_report,fjs,indent=2)

    # ── (v16.1) MANDATORY Murcko-core leave-cluster-out CV ──
    log.info("="*60); log.info("Murcko-core split CV (leave-core-out)"); log.info("="*60)
    core_cv_report={}
    for lb,(_,_,yy) in edict.items():
        core_cv_report[lb]=core_split_cv(X,yy,smi_train,label=lb,
                                          min_test_size=args.core_split_cv_min_test)
    with open(os.path.join(args.output,"core_split_cv.json"),"w") as fjs:
        json.dump(core_cv_report,fjs,indent=2)

    # ── (v16.1) Residual cross-target covariance for joint MC ──
    residual_cov=None
    if args.enable_residual_cov:
        log.info("="*60); log.info("Residual cross-target covariance"); log.info("="*60)
        cov_keys=["DeltaEST","T1-S1(SOC)","T2-S1(SOC)","OscStr"]
        try:
            residual_cov=compute_residual_covariance(edict, X, cov_keys, pred_fn=pred_fn)
            with open(os.path.join(args.output,"residual_covariance.json"),"w") as fjs:
                json.dump({"target_order":cov_keys,
                           "covariance":residual_cov.tolist(),
                           "units":{"DeltaEST":args.energy_units,
                                    "T1-S1(SOC)":args.soc_units,
                                    "T2-S1(SOC)":args.soc_units,
                                    "OscStr":"dimensionless"}}, fjs, indent=2)
            log.info("  Residual covariance computed and saved.")
            log.info("  Diag (variance) per target: %s",
                     {k:float(np.diag(residual_cov)[i])
                      for i,k in enumerate(cov_keys)})
        except Exception as _e:
            log.warning("  Residual covariance failed: %s", _e)
            residual_cov=None

    # ── (v13) Uncertainty-error correlation on full training set ──
    log.info("="*60); log.info("Uncertainty-error correlation"); log.info("="*60)
    unc_err_report={}
    for lb,(en,sc_,yy) in edict.items():
        mu,sg=pred_fn(en,sc_,X)
        if lb in kappa: sg=sg*kappa[lb]
        unc_err_report[lb]=uncertainty_error_correlation(mu,sg,yy,label=lb)
    with open(os.path.join(args.output,"uncertainty_error_correlation.json"),"w") as fjs:
        json.dump(unc_err_report,fjs,indent=2)

    # Generate
    log.info("="*60); log.info("Generating"); log.info("="*60)
    cands=gen_cands(smi_seeds,args.n_candidates,args.n_workers,
                    args.attempts_per_worker,args.exploratory,
                    args.enable_bulky,args.enable_diaza,
                    train_smiles=smi_train,
                    enable_annulation=args.enable_annulation)
    if not cands: log.error("No candidates!"); sys.exit(1)

    # v18: deep-generative augmentation (VAE / Flow / Diffusion / RL)
    # block removed. The candidate pool is now exclusively operator-
    # generated. Run-summary JSON no longer reports a "deep_generative"
    # section.

    # Descriptors
    log.info("="*60); log.info("Descriptors"); log.info("="*60)
    Xc,sm,vc=comp_desc(cands,dn,args.n_workers); Xc=imp.transform(Xc)

    # ── Compute training FoM incumbent for multi-objective EI ──
    fom_best=None
    if args.objective=="TADF_FoM":
        fom_train=tadf_figure_of_merit(yd,ys1,ys2,yf,
                                       dEST_unit=args.energy_units,
                                       soc_unit=args.soc_units)
        fom_best=float(np.max(fom_train))
        log.info("  Training FoM incumbent (max log10): %.3f", fom_best)

    # Score
    log.info("="*60); log.info("Scoring"); log.info("="*60)
    dfs=score_cands(Xc,vc,sm,ed,sd,et,st,e1,s1,e2,s2,ef,sf,esg,ssg,
                    yd.min(),tc,sc,adsc,Xsc,args.novelty_mode,
                    kappa=kappa, objective=args.objective,
                    energy_units=args.energy_units, soc_units=args.soc_units,
                    pred_fn=pred_fn, fom_best=fom_best,
                    residual_cov=residual_cov,
                    kappa_by_family=kappa_by_family)
    dfs.insert(0,"smiles",sm)
    dfs=dfs[dfs["smiles"].notna()&(dfs["smiles"]!="")].reset_index(drop=True)
    log.info("  Scored: %d",len(dfs))

    # ── (v16.1) Stricter applicability-domain gating ─────────────────
    # Hard-reject candidates with AD_score < ad_hard_threshold (zeroes
    # trust_parent so legacy rank_cands drops them naturally). Soft band
    # gets a 0.5× trust penalty. Reports counts via 'AD_hard_rejected'.
    n_before_ad=len(dfs)
    dfs=apply_strict_ad_gate(dfs,
                             ad_hard_threshold=args.ad_hard_threshold,
                             ad_soft_threshold=args.ad_soft_threshold)
    n_ad_hard=int(dfs.get("AD_hard_rejected", pd.Series([0]*len(dfs))).sum())
    log.info("  Strict AD gate: %d/%d hard-rejected (AD<%.2f); "
             "soft-penalty band [%.2f, %.2f).",
             n_ad_hard, n_before_ad, args.ad_hard_threshold,
             args.ad_hard_threshold, args.ad_soft_threshold)

    # ── (v16.1) Candidate queues (exploitation / exploration / control) ──
    dfs=make_candidate_queues(dfs,
                              n_exploit=args.n_exploit,
                              n_explore=args.n_explore,
                              n_control=args.n_control,
                              ad_hard_threshold=args.ad_hard_threshold)
    if "queue" in dfs.columns:
        qcounts=dfs["queue"].value_counts().to_dict()
        log.info("  Queues: exploit=%d  explore=%d  control=%d",
                 int(qcounts.get("exploit",0)),
                 int(qcounts.get("explore",0)),
                 int(qcounts.get("control",0)))

    # Rank + diversity
    log.info("="*60); log.info("Ranking + Diversity"); log.info("="*60)
    dfr=rank_cands(dfs); log.info("  Ranked: %d",len(dfr))
    dfr=div_select(dfr,args.top_diverse_n,args.diversity_lambda,
                   max_per_parent=args.max_per_parent,
                   max_per_family=args.max_per_scaffold_family)
    nd=dfr["diversity_selected"].sum(); log.info("  Diverse: %d",nd)

    # Save core outputs.
    # v17 (#50): route through _stable_csv_columns so the on-disk header
    # is identical across runs (missing columns padded with NaN, extras
    # appended at the end). Downstream tools can pin to the canonical
    # schema in _SCORED_CSV_SCHEMA without breaking when an optional
    # gate (AD, queue, QC, …) is skipped or its output column is absent.
    _stable_csv_columns(dfs).to_csv(
        os.path.join(args.output,"all_scored.csv"),index=False)
    _stable_csv_columns(dfr).to_csv(os.path.join(args.output,"ranked_candidates.csv"))
    dfd=dfr[dfr["diversity_selected"]].head(args.top_diverse_n)
    _stable_csv_columns(dfd).to_csv(os.path.join(args.output,"top_diverse_candidates.csv"))
    _stable_csv_columns(dfr.head(50)).to_csv(
        os.path.join(args.output,"top50_candidates.csv"))

    # ── (v16.1) Per-queue CSVs ───────────────────────────────────────
    queue_summary={}
    if "queue" in dfs.columns:
        for qtag in ("exploit","explore","control"):
            qsub=dfs[dfs["queue"]==qtag].copy()
            if len(qsub)>0:
                qsub=qsub.sort_values("final_score" if qtag!="explore" else "novelty_score",
                                       ascending=False)
                _stable_csv_columns(qsub).to_csv(
                    os.path.join(args.output, f"queue_{qtag}.csv"), index=False)
            queue_summary[qtag]={"n":int(len(qsub))}
        log.info("  queue_*.csv saved (%s)", queue_summary)

    # ─── Validation export (Part 6) ───
    if len(dfd)>0:
        vq_cols=["smiles","parent_seed_smiles","parent_scaffold_type",
                 "final_score","validation_priority_score","novelty_score",
                 "subst_topo_novelty","core_topology_distance_to_parent",
                 "AD_score","trust_parent","trust_train",
                 "pred_DeltaEST","pred_T2_T1","pred_OscStr",
                 "pred_T1S1_SOC","pred_T2S1_SOC","pred_Singlets",
                 "unc_DeltaEST","unc_T2_T1",
                 "core_n_donor","core_n_acceptor","core_size","core_sym_index",
                 "subst_topo_sig","sym_pattern_sig"]
        vq_cols=[c for c in vq_cols if c in dfd.columns]
        vq=dfd[vq_cols].copy()
        vq["validation_priority"]=range(1,len(vq)+1)
        vq.to_csv(os.path.join(args.output,"export_validation_queue.csv"),index=False)
        log.info("  Validation queue: %d molecules exported",len(vq))

    # ─── Topology summary (Part 11) ───
    if len(dfd)>0:
        topo_cols=["smiles","subst_topo_sig","sym_pattern_sig",
                   "subst_topo_dist_parent","subst_topo_dist_seed_family",
                   "subst_topo_dist_shortlist","subst_topo_novelty",
                   "core_topology_distance_to_parent","core_topology_distance_to_seed_family",
                   "core_topo_novelty","core_n_donor",
                   "core_n_acceptor","core_size","core_sym_index",
                   "core_n_perim_expand","core_n_overlap","parent_scaffold_type"]
        topo_cols=[c for c in topo_cols if c in dfd.columns]
        dfd[topo_cols].to_csv(os.path.join(args.output,"topology_summary.csv"),index=False)

    # ─── QC results reimport (Part 4: reconciliation) ───
    qc_merged=False
    # Validation tier tags
    dfd["validation_tier"]="Tier1_surrogate"
    if args.qc_results_file:
        log.info("="*60); log.info("QC Reconciliation"); log.info("="*60)
        try:
            qcdf=pd.read_csv(args.qc_results_file)
            if "smiles" in qcdf.columns:
                qc_can={}
                for _,row in qcdf.iterrows():
                    m=MolFromSmiles(str(row["smiles"]))
                    if m: qc_can[MolToSmiles(m)]=row.to_dict()
                qc_cols=[c for c in qcdf.columns if c!="smiles"]
                for qc in qc_cols: dfd[f"qc_{qc}"]=None
                n_matched=0
                for idx,row in dfd.iterrows():
                    m=MolFromSmiles(row["smiles"])
                    if m:
                        csmi=MolToSmiles(m)
                        if csmi in qc_can:
                            for qc in qc_cols:
                                dfd.at[idx,f"qc_{qc}"]=qc_can[csmi].get(qc)
                            n_matched+=1
                            dfd.at[idx,"validation_tier"]="Tier3_qc_merged"
                log.info("  Matched %d / %d to QC",n_matched,len(dfd))
                # Discrepancy + concordance flags. Threshold is unit-aware
                # via _CONCORD_DEFAULT (interpreted in the user-declared units
                # of the corresponding column).
                #
                # v17 fixes:
                #   (1) score_cands writes prediction columns with
                #       *underscored* identifier-safe names (pred_T2_T1,
                #       pred_T1S1_SOC, pred_T2S1_SOC) — but this loop
                #       previously built `pred_{tgt}` directly, producing
                #       pred_T2-T1 etc. that NEVER matched. The fix uses
                #       an explicit mapping.
                #   (2) The concordance flag was never assigned True
                #       because the prior "if cur is None: cur=True"
                #       reassigned a LOCAL variable, then the
                #       "if cur is None: dfd[...]=True" branch was dead.
                #       Rewritten as a clean per-row verdict that
                #       monotonically combines per-target outcomes
                #       (False overrides marginal overrides True).
                _PRED_COL_FOR={"DeltaEST":"pred_DeltaEST","T2-T1":"pred_T2_T1",
                               "T1-S1(SOC)":"pred_T1S1_SOC",
                               "T2-S1(SOC)":"pred_T2S1_SOC",
                               "OscStr":"pred_OscStr","Singlets":"pred_Singlets"}
                disc={}; dfd["qc_concordant"]=None
                for tgt in ["DeltaEST","T2-T1","OscStr","T1-S1(SOC)","T2-S1(SOC)","Singlets"]:
                    pcol=_PRED_COL_FOR.get(tgt, f"pred_{tgt}")
                    qcol=f"qc_{tgt}"
                    if pcol in dfd.columns and qcol in dfd.columns:
                        valid=dfd.dropna(subset=[pcol,qcol])
                        if len(valid)>0:
                            delta=(valid[pcol].astype(float)-valid[qcol].astype(float)).abs()
                            disc[tgt]={"n":int(len(valid)),
                                "mean_abs_discrepancy":float(delta.mean()),
                                "max_abs_discrepancy":float(delta.max())}
                            thresh=_CONCORD_DEFAULT.get(tgt,0.15)
                            for vi in valid.index:
                                d=abs(float(dfd.at[vi,pcol])-float(dfd.at[vi,qcol]))
                                # Per-target verdict for this row.
                                if d>thresh*2:    verdict=False
                                elif d>thresh:    verdict="marginal"
                                else:             verdict=True
                                cur=dfd.at[vi,"qc_concordant"]
                                # Monotone combine: False > marginal > True.
                                if cur is None:
                                    dfd.at[vi,"qc_concordant"]=verdict
                                elif cur is False:
                                    pass
                                elif verdict is False:
                                    dfd.at[vi,"qc_concordant"]=False
                                elif cur=="marginal":
                                    pass
                                elif verdict=="marginal":
                                    dfd.at[vi,"qc_concordant"]="marginal"
                                # else: cur==True and verdict==True → stays True
                # v17 (#50): canonical-schema-first ordering; qc_* and
                # other extras appear after the canonical block in their
                # original order.
                _stable_csv_columns(dfd).to_csv(
                    os.path.join(args.output,"validation_merged_results.csv"),index=False)
                if disc:
                    with open(os.path.join(args.output,"qc_discrepancy_summary.json"),"w") as f:
                        json.dump(disc,f,indent=2)
                    log.info("  QC discrepancy summary saved (%d targets)",len(disc))
                qc_merged=True
            else:
                log.warning("  QC file lacks 'smiles' column")
        except Exception as e:
            log.warning("  QC merge failed: %s",e)
    # Mark unmatched as Tier2 if QC was requested but not matched
    if args.qc_results_file and qc_merged:
        dfd.loc[dfd["validation_tier"]=="Tier1_surrogate","validation_tier"]="Tier2_qc_requested"

    # ─── Benchmark predictions (Part 4: labeled evaluation) ───
    bench_summary={}; bench_metrics={}
    if args.benchmark_file:
        log.info("="*60); log.info("Benchmark evaluation"); log.info("="*60)
        try:
            if args.benchmark_file.endswith(".xlsx"):
                bf=pd.read_excel(args.benchmark_file)
                bsmi=bf.iloc[:,1 if bf.shape[1]>1 else 0].dropna().astype(str).tolist()
            else:
                bf=None
                with open(args.benchmark_file) as f:
                    bsmi=[l.strip() for l in f if l.strip()]
            log.info("  Loaded %d benchmark SMILES",len(bsmi))
            # Check if benchmark file has labeled targets
            labels_df=bf if (bf is not None and bf.shape[1]>2) else None
            if labels_df is not None:
                log.info("  Labeled benchmark detected (%d columns)",bf.shape[1])
            bdf,bench_metrics=predict_benchmark(bsmi,edict,dn,imp,
                                                min(args.n_workers,4),labels_df)
            if len(bdf)>0:
                bdf.to_csv(os.path.join(args.output,"benchmark_predictions.csv"),index=False)
                log.info("  Predictions saved: %d rows",len(bdf))
            if bench_metrics:
                with open(os.path.join(args.output,"benchmark_metrics.json"),"w") as f:
                    json.dump(bench_metrics,f,indent=2)
                log.info("  benchmark_metrics.json saved with %d targets",len(bench_metrics))
            bench_summary=bench_metrics
        except Exception as e:
            log.warning("  Benchmark failed: %s",e)

    if len(dfd)>0:
        log.info("\n"+"="*60); log.info("TOP 10 DIVERSE"); log.info("="*60)
        for i,row in dfd.head(10).iterrows():
            log.info("  #%2d │ ΔEST=%.3f T2T1=%.3f fOSC=%.3f Nov=%.2f AD=%.2f │ %s",
                     i,row["pred_DeltaEST"],row["pred_T2_T1"],row["pred_OscStr"],
                     row["novelty_score"],row["AD_score"],row["smiles"][:35])

    # ─── (v16.1) Interpretability report ────────────────────────────
    log.info("="*60); log.info("Interpretability report"); log.info("="*60)
    try:
        interp_summary=interpretability_report(edict, dn, args.output,
                                                top_k=20, pred_fn=pred_fn)
    except Exception as _e:
        log.warning("  Interpretability report failed: %s", _e)
        interp_summary={}

    # ─── (v16.1) Active-learning validation history ─────────────────
    log.info("="*60); log.info("Active-learning validation history"); log.info("="*60)
    history_path=(args.validation_history_file
                  or os.path.join(args.output, "validation_history.json"))
    try:
        # Build a QC dataframe for matching (if QC was supplied this run).
        _qc_for_history=None
        if args.qc_results_file:
            try:
                _qc_for_history=pd.read_csv(args.qc_results_file)
            except Exception:
                _qc_for_history=None
        run_id=time.strftime("%Y%m%dT%H%M%S")
        update_validation_history(history_path, run_id, dfd if len(dfd)>0 else dfr,
                                   qc_df=_qc_for_history,
                                   top_n=max(args.top_diverse_n, 50))
        history_summary=summarize_validation_history(history_path)
        if history_summary:
            n_runs=len(history_summary)
        elif os.path.isfile(history_path):
            try:
                with open(history_path) as _hf:
                    n_runs=len(json.load(_hf).get("runs",[]))
            except Exception:
                n_runs=0
        else:
            n_runs=0
        log.info("  History updated: %s (runs=%d)", history_path, n_runs)
    except Exception as _e:
        log.warning("  Validation history update failed: %s", _e)
        history_summary=[]

    # ─── Publication diagnostic plots (Part 6) ───
    try:
        if len(dfs)>10:
            # Novelty vs AD scatter
            fig,ax=plt.subplots(figsize=(5,4))
            ax.scatter(dfs["AD_score"],dfs["novelty_score"],s=8,alpha=0.4,c='#2166ac')
            ax.set_xlabel("AD Score"); ax.set_ylabel("Novelty Score")
            ax.set_title("Novelty vs Applicability Domain"); ax.grid(True,alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output,"novelty_vs_AD.png"),dpi=150)
            plt.close(fig)
            # Parent-core similarity vs final_score
            fig,ax=plt.subplots(figsize=(5,4))
            ax.scatter(dfs["parent_tanimoto_core"],dfs["final_score"],s=8,alpha=0.4,c='#b2182b')
            ax.set_xlabel("Parent Core Similarity"); ax.set_ylabel("Final Score")
            ax.set_title("Parent-Core Similarity vs Final Score"); ax.grid(True,alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output,"parent_core_vs_score.png"),dpi=150)
            plt.close(fig)
            # Scaffold family barplot for diverse shortlist
            if len(dfd)>0 and "parent_scaffold_type" in dfd.columns:
                fig,ax=plt.subplots(figsize=(5,3))
                dfd["parent_scaffold_type"].value_counts().plot.bar(ax=ax,color='#4393c3')
                ax.set_ylabel("Count"); ax.set_title("Scaffold Families in Shortlist")
                fig.tight_layout()
                fig.savefig(os.path.join(args.output,"scaffold_family_barplot.png"),dpi=150)
                plt.close(fig)
            log.info("  Diagnostic plots saved")
    except Exception as e:
        log.warning("  Diagnostic plots failed: %s",e)

    elapsed=time.time()-t0
    log.info("\n"+"="*60); log.info("DONE (%.1f min)",elapsed/60); log.info("="*60)

    # ─── Metadata (Part 11) ───
    npr=int((dfs["trust_parent"]==0).sum()); ntr=int((dfs["trust_train"]==0).sum())
    ncf=int((dfs["core_fp_valid"]==0).sum())
    ncir=int((dfs["core_invalid_risk"]>0).sum())
    usig=dfd["subst_topo_sig"].nunique() if len(dfd)>0 else 0
    sfam=dfd["parent_scaffold_type"].value_counts().to_dict() if len(dfd)>0 else {}
    # Unique core-topology states
    uctopo=0
    if len(dfd)>0:
        ct_cols=[c for c in ["core_n_donor","core_n_acceptor","core_size"] if c in dfd.columns]
        if ct_cols: uctopo=dfd[ct_cols].drop_duplicates().shape[0]

    meta={"version":"v16.1","novelty_mode":args.novelty_mode,
          "objective":args.objective,
          "surrogate":("GPR" if args.use_gpr
                       else ("Stacked-XGB+Ridge" if args.enable_stacking
                             else "bagged-XGB")),
          "units":{"energy":args.energy_units,"soc":args.soc_units},
          "seeds_separate":seeds_separate,
          "benchmark_provided":args.benchmark_file is not None,
          "ranking_mode":"integrated_final_score",
          "annulation_active":bool(args.enable_annulation),
          "conformal_kappa":kappa if kappa else None,
          "scaffold_families_supported":[SF.BN,SF.BO,SF.CO,SF.SE,SF.TE,
                                          SF.PO,SF.PS,SF.PSE,SF.NO,SF.OT],
          "donor_library":[d[0] for d in _DONOR_CACHE],
          "trust":{"max_whole_sim_train":MAX_WHOLE_SIM_TRAIN,
                   "min_core_sim_train":MIN_CORE_SIM_TRAIN,
                   "min_core_sim_parent":MIN_CORE_SIM_PARENT},
          "diversity":{"lambda":args.diversity_lambda,"top_n":args.top_diverse_n,
                       "max_per_parent":args.max_per_parent,
                       "max_per_family":args.max_per_scaffold_family},
          "rejections":{"parent_drift":npr,"train_drift":ntr,
                        "invalid_core_fp":ncf,"core_invalid_risk_kept":ncir},
          "novelty_stats":{"mean":float(dfs["novelty_score"].mean()),
                           "q25":float(dfs["novelty_score"].quantile(0.25)),
                           "q75":float(dfs["novelty_score"].quantile(0.75))},
          "AD_stats":{"mean":float(dfs["AD_score"].mean()),
                      "q25":float(dfs["AD_score"].quantile(0.25)),
                      "q75":float(dfs["AD_score"].quantile(0.75))},
          "parent_core_stats":{"mean":float(dfs["parent_tanimoto_core"].mean()),
                               "q25":float(dfs["parent_tanimoto_core"].quantile(0.25))},
          "shortlist":{"unique_subst_topo_sigs":usig,
                       "unique_core_topo_states":uctopo,
                       "scaffold_families":sfam},
          "benchmark":bench_summary,
          "n_scored":len(dfs),"n_ranked":len(dfr),"n_diverse":int(nd),
          "elapsed_s":elapsed,
          # v16 additions (deep_generative removed in v18)
          "v16":{
              "sa_filter":{"enabled":not args.disable_sa_filter,
                           "cap":float(args.sascore_max),
                           "fitted_n_ref":int(_GLOBAL_SASCORER.n_ref)},
              "retro_filter":{"enabled":not args.disable_retro_filter},
              "charge_radical_filter":{"enabled":not args.disable_charge_radical_filter},
              "ann_BN_topology":"1,2-azaborine_ring_fusion_with_N-Ph_and_B-Ph",
          },
          # v16.1 additions (deep_gen_* keys removed in v18)
          "v16_1":{
              "stacking":{"enabled":bool(args.enable_stacking),
                          "alpha":float(args.stacking_alpha)},
              "scaffold_conformal":{"enabled":bool(args.scaffold_conformal),
                                     "n_families":(len(kappa_by_family)
                                                   if kappa_by_family else 0)},
              "residual_covariance":{"enabled":bool(args.enable_residual_cov),
                                      "available":residual_cov is not None},
              "label_source_weighting":{"col":args.label_source_col,
                                         "summary":label_source_summary},
              "ad_gating":{"hard":float(args.ad_hard_threshold),
                            "soft":float(args.ad_soft_threshold),
                            "n_hard_rejected":int(n_ad_hard)},
              "queues":queue_summary,
              "interpretability":interp_summary,
              "validation_history":{
                  "path":history_path,
                  "n_runs":len(history_summary),
                  "latest_summary":(history_summary[-1] if history_summary else None)},
              "scaffold_stratified_cv":{
                  lb:(r.get("macro_MAE") if isinstance(r,dict) else None)
                  for lb,r in cv_report.items()},
              "core_split_cv":{
                  lb:(r.get("macro_MAE") if isinstance(r,dict) else None)
                  for lb,r in core_cv_report.items()},
          }}
    with open(os.path.join(args.output,"metadata.json"),"w") as f:
        json.dump(meta,f,indent=2)

    # ─── Publication diagnostics (Part 6/11) ───
    pub_diag={
        "pipeline_version":"v16",
        "n_training":int(X.shape[0]),
        "n_candidates_generated":len(cands),
        "n_scored":len(dfs),
        "n_ranked":len(dfr),
        "n_diverse_selected":int(nd),
        "unique_subst_topo_sigs_in_shortlist":usig,
        "unique_core_topo_states_in_shortlist":uctopo,
        "scaffold_family_distribution":sfam,
        "novelty_summary":{
            "mean":float(dfs["novelty_score"].mean()),
            "median":float(dfs["novelty_score"].median()),
            "q10":float(dfs["novelty_score"].quantile(0.10)),
            "q90":float(dfs["novelty_score"].quantile(0.90))},
        "AD_summary":{
            "mean":float(dfs["AD_score"].mean()),
            "median":float(dfs["AD_score"].median()),
            "q10":float(dfs["AD_score"].quantile(0.10)),
            "q90":float(dfs["AD_score"].quantile(0.90))},
        "parent_core_sim_summary":{
            "mean":float(dfs["parent_tanimoto_core"].mean()),
            "median":float(dfs["parent_tanimoto_core"].median())},
        "invalid_core_rejections":ncf,
        "core_invalid_risk_retained":ncir,
        "trust_parent_rejections":npr,
        "trust_train_rejections":ntr,
        "benchmark":bench_summary,
        "annulation_status":("active" if args.enable_annulation
                             else "metadata_only_deferred"),
        "objective":args.objective,
        "surrogate":"GPR" if args.use_gpr else "bagged-XGB",
        "units":{"energy":args.energy_units,"soc":args.soc_units},
        "TADF_FoM_summary":{
            "mean":float(dfs["log_TADF_FoM"].mean()) if "log_TADF_FoM" in dfs else None,
            "q90":float(dfs["log_TADF_FoM"].quantile(0.90)) if "log_TADF_FoM" in dfs else None,
        },
        "uncertainty_error_correlation":{
            lb:r.get("spearman_rho") for lb,r in unc_err_report.items()
            if isinstance(r,dict)},
    }
    with open(os.path.join(args.output,"publication_diagnostics.json"),"w") as f:
        json.dump(pub_diag,f,indent=2)
    log.info("  publication_diagnostics.json saved")

if __name__=="__main__": main()
