"""
transfer_learning.py — QM9-pretrained backbone with domain-adaptive fine-tuning
================================================================================

Scientific rationale:
  With only 176 MR-TADF molecules, a randomly-initialised deep network cannot
  learn generalisable structure–property relationships.  QM9 provides ~134 k
  small organic molecules with DFT-computed HOMO, LUMO, gap, dipole, and
  atomisation energies.  Although QM9 molecules are smaller (≤9 heavy atoms)
  than typical MR-TADF emitters (40–130 heavy atoms), the *electronic-structure
  descriptors* (Moreau-Broto autocorrelations, ETA indices, PEOE_VSA, etc.)
  share the same feature space, so a backbone pre-trained on QM9 learns
  reusable descriptor→property mappings that transfer.

Strategy (3 phases):
  Phase 1 — Pre-train on QM9
    • Compute the same 2870 Mordred/RDKit descriptors for all QM9 molecules
    • Train backbone to predict HOMO, LUMO, gap, μ, α (5 tasks)
    • This teaches the model generic "descriptor → electronic property" patterns

  Phase 2 — Domain-adaptive alignment
    • Use Maximum Mean Discrepancy (MMD) loss to align the latent
      representations of QM9 and MR-TADF descriptor distributions
    • This closes the domain gap arising from molecular-size differences

  Phase 3 — Task-specific fine-tuning on MR-TADF
    • Replace the 5-target QM9 head with the 6-target MR-TADF head
    • Apply progressive unfreezing: first train only the new head,
      then unfreeze top blocks, then all parameters
    • Use discriminative learning rates (lower LR for early layers)

Dependencies:
  pip install torch pandas numpy scikit-learn openpyxl rdkit-pypi mordred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  1. QM9 DATA LOADER AND DESCRIPTOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════

class QM9DescriptorComputer:
    """
    Computes the SAME molecular descriptor set used for MR-TADF molecules
    from QM9 SMILES, ensuring feature-space alignment for transfer learning.

    QM9 properties (indices in the original dataset):
      0: μ (Debye)          — dipole moment
      1: α (Bohr³)          — isotropic polarisability
      2: ε_HOMO (Ha)        — HOMO energy
      3: ε_LUMO (Ha)        — LUMO energy
      4: Δε (Ha)            — HOMO–LUMO gap
      5: ⟨R²⟩ (Bohr²)      — electronic spatial extent
      6: ZPVE (Ha)          — zero-point vibrational energy
      7: U₀ (Ha)            — internal energy at 0 K
      8: U (Ha)             — internal energy at 298.15 K
      9: H (Ha)             — enthalpy at 298.15 K
     10: G (Ha)             — free energy at 298.15 K
     11: Cv (cal/mol·K)     — heat capacity at 298.15 K

    For transfer we use: HOMO, LUMO, gap, dipole, polarisability (indices 2-4, 0, 1).
    """

    # Subset of descriptors shared between QM9 and MR-TADF feature sets.
    # These are the Mordred/RDKit descriptors computable for any organic molecule.
    TRANSFER_DESCRIPTOR_MODULES = [
        "ATS",       # Moreau-Broto autocorrelation (mass, volume, electronegativity, polarisability)
        "AATS",      # averaged autocorrelation
        "ATSC",      # centred autocorrelation
        "AATSC",     # averaged centred autocorrelation
        "MATS",      # Moran autocorrelation
        "GATS",      # Geary autocorrelation
        "ETA",       # extended topochemical atom
        "BCUT",      # Burden eigenvalues
        "Chi",       # Kier-Hall connectivity
        "Kappa",     # Kier shape
        "PEOE_VSA",  # partial equalization of orbital electronegativity
        "SMR_VSA",   # molar refractivity VSA
        "SlogP_VSA", # LogP VSA
        "EState_VSA", # electrotopological state VSA
        "IC",        # information content
        "GGI",       # topological charge
    ]

    QM9_TARGET_NAMES = ["dipole", "polarisability", "HOMO", "LUMO", "gap"]
    QM9_TARGET_INDICES = [0, 1, 2, 3, 4]  # in the standard QM9 property ordering

    @staticmethod
    def compute_descriptors_rdkit(smiles_list: List[str],
                                  descriptor_names: List[str]) -> np.ndarray:
        """
        Compute molecular descriptors for a list of SMILES using RDKit + Mordred.

        This function mirrors the descriptor computation used for the MR-TADF
        dataset, ensuring the feature spaces are aligned.

        Args:
            smiles_list: list of SMILES strings
            descriptor_names: list of descriptor column names from the MR-TADF dataset

        Returns:
            np.ndarray of shape (n_molecules, n_descriptors)
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors as RDDescriptors
            from mordred import Calculator, descriptors as MordredDescriptors
        except ImportError:
            raise ImportError(
                "QM9 descriptor computation requires rdkit-pypi and mordred.\n"
                "Install with: pip install rdkit-pypi mordred"
            )

        # Build Mordred calculator
        calc = Calculator(MordredDescriptors, ignore_3D=True)

        results = []
        valid_indices = []

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            try:
                desc_values = calc(mol)
                # Convert to dict for name-based lookup
                desc_dict = {}
                for name, val in zip(calc.descriptors, desc_values):
                    desc_dict[str(name)] = float(val) if not isinstance(val, Exception) else 0.0

                # Also compute RDKit descriptors
                for name, func in RDDescriptors.descList:
                    try:
                        desc_dict[name] = float(func(mol))
                    except Exception:
                        desc_dict[name] = 0.0

                # Extract only the descriptors present in the MR-TADF feature set
                row = []
                for dname in descriptor_names:
                    row.append(desc_dict.get(dname, 0.0))

                results.append(row)
                valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Descriptor computation failed for {smi}: {e}")
                continue

        logger.info(f"Computed descriptors for {len(results)}/{len(smiles_list)} molecules")
        return np.array(results, dtype=np.float64), valid_indices

    @staticmethod
    def load_qm9_from_sdf(qm9_path: str) -> Tuple[List[str], np.ndarray]:
        """
        Load QM9 dataset from the standard SDF file or CSV.

        Expected format (CSV): SMILES, property_0, property_1, ..., property_11

        Returns:
            smiles_list, properties array (n, 12)
        """
        path = Path(qm9_path)

        if path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(path)
            smiles_col = [c for c in df.columns if 'smi' in c.lower()][0]
            prop_cols = [c for c in df.columns if c != smiles_col]
            return df[smiles_col].tolist(), df[prop_cols].values.astype(np.float64)

        elif path.suffix == '.sdf':
            from rdkit import Chem
            supplier = Chem.SDMolSupplier(str(path), removeHs=True)
            smiles_list, properties = [], []
            for mol in supplier:
                if mol is None:
                    continue
                smiles_list.append(Chem.MolToSmiles(mol))
                props = []
                for i in range(12):
                    try:
                        props.append(float(mol.GetProp(f'prop_{i}')))
                    except Exception:
                        props.append(0.0)
                properties.append(props)
            return smiles_list, np.array(properties)

        else:
            raise ValueError(f"Unsupported QM9 file format: {path.suffix}")


# ═══════════════════════════════════════════════════════════════════
#  2. TRANSFER-LEARNING BACKBONE
# ═══════════════════════════════════════════════════════════════════

class TransferableBackbone(nn.Module):
    """
    Shared feature extractor that can be pre-trained on QM9 and
    fine-tuned on MR-TADF.

    Architecture:
      Input → ProjectionLayer → [ResBlock₁, ResBlock₂, ResBlock₃, ResBlock₄]
                                         ↑ frozen in early fine-tuning

    The backbone is separated from the task head so that:
      - Pre-training attaches a QM9 head (5 targets)
      - Fine-tuning replaces it with an MR-TADF head (6 targets)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_blocks: int = 4, dropout: float = 0.15):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Store blocks as individually addressable for progressive unfreezing
        self.blocks = nn.ModuleList([
            _ResidualBlockTL(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        self.attention = _SelfGatedAttention(hidden_dim)

        self.output_dim = hidden_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.attention(h)
        return h

    def freeze_all(self):
        """Freeze entire backbone (for head-only training)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_top_k(self, k: int = 2):
        """Unfreeze the top k residual blocks + attention."""
        self.freeze_all()
        # Always unfreeze attention
        for param in self.attention.parameters():
            param.requires_grad = True
        # Unfreeze last k blocks
        for block in self.blocks[-k:]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        """Return parameter groups for discriminative LR scheduling."""
        groups = [
            list(self.input_proj.parameters()),   # lowest LR
        ]
        for block in self.blocks:
            groups.append(list(block.parameters()))
        groups.append(list(self.attention.parameters()))  # highest LR
        return groups


class _ResidualBlockTL(nn.Module):
    """Pre-activation residual block with layer normalisation for stability."""
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _SelfGatedAttention(nn.Module):
    """Gated self-attention pooling for feature interaction modelling."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.value = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate(x) * self.value(x)


# ═══════════════════════════════════════════════════════════════════
#  3. TASK HEADS
# ═══════════════════════════════════════════════════════════════════

class QM9Head(nn.Module):
    """Pre-training head: predicts 5 QM9 properties."""
    QM9_TARGETS = ["HOMO", "LUMO", "gap", "dipole", "polarisability"]

    def __init__(self, input_dim: int = 256, n_targets: int = 5):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, n_targets),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class MRTADFHead(nn.Module):
    """
    Fine-tuning head for MR-TADF: predicts 6 photophysical properties.

    Uses separate sub-heads per target for task-specific capacity,
    with a shared representation layer.
    """
    TADF_TARGETS = ["T1-S1", "T2-S1", "DeltaEST", "S1", "T1", "f"]

    def __init__(self, input_dim: int = 256, n_targets: int = 6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
        )
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim // 2, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            ) for _ in range(n_targets)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared = self.shared(features)
        outputs = [head(shared) for head in self.task_heads]
        return torch.cat(outputs, dim=-1)


# ═══════════════════════════════════════════════════════════════════
#  4. DOMAIN ADAPTATION — MAXIMUM MEAN DISCREPANCY (MMD)
# ═══════════════════════════════════════════════════════════════════

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy with a mixture of RBF kernels.

    MMD measures the distance between two probability distributions in a
    reproducing kernel Hilbert space.  By minimising MMD(backbone(X_qm9),
    backbone(X_tadf)), we align the internal representations so that
    features learned from QM9 transfer more effectively.

    Uses multi-scale RBF kernels (σ ∈ {0.1, 0.5, 1, 2, 5}) to capture
    distributional differences at multiple length scales.
    """
    def __init__(self, sigmas: Optional[List[float]] = None):
        super().__init__()
        if sigmas is None:
            sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        self.sigmas = sigmas

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix K(x, y) with multiple bandwidths."""
        xx = (x * x).sum(dim=-1, keepdim=True)
        yy = (y * y).sum(dim=-1, keepdim=True)
        dist_sq = xx + yy.t() - 2 * x @ y.t()

        kernel = torch.zeros_like(dist_sq)
        for sigma in self.sigmas:
            kernel += torch.exp(-dist_sq / (2 * sigma ** 2))
        return kernel / len(self.sigmas)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute unbiased MMD² estimate.

        source: (n_s, d) — source domain features (QM9)
        target: (n_t, d) — target domain features (MR-TADF)
        """
        n_s, n_t = source.size(0), target.size(0)

        k_ss = self._rbf_kernel(source, source)
        k_tt = self._rbf_kernel(target, target)
        k_st = self._rbf_kernel(source, target)

        # Unbiased estimator: exclude diagonal from k_ss and k_tt
        mmd = (k_ss.sum() - k_ss.trace()) / (n_s * (n_s - 1)) \
            + (k_tt.sum() - k_tt.trace()) / (n_t * (n_t - 1)) \
            - 2 * k_st.mean()

        return torch.clamp(mmd, min=0.0)


# ═══════════════════════════════════════════════════════════════════
#  5. TRANSFER LEARNING TRAINER
# ═══════════════════════════════════════════════════════════════════

class TransferLearningTrainer:
    """
    Three-phase training pipeline:
      Phase 1: Pre-train backbone + QM9 head on QM9 data
      Phase 2: Domain adaptation with MMD alignment
      Phase 3: Progressive fine-tuning on MR-TADF

    Progressive unfreezing schedule:
      Epoch [0, E/3):     freeze backbone, train head only
      Epoch [E/3, 2E/3):  unfreeze top 2 blocks + head
      Epoch [2E/3, E):    unfreeze all, discriminative LR

    Discriminative learning rates:
      input_proj:  base_lr × 0.01
      block_0:     base_lr × 0.05
      block_1:     base_lr × 0.1
      block_2:     base_lr × 0.3
      block_3:     base_lr × 1.0
      attention:   base_lr × 1.0
      head:        base_lr × 1.0
    """

    LR_MULTIPLIERS = [0.01, 0.05, 0.1, 0.3, 1.0, 1.0]  # per layer group

    def __init__(self, backbone: TransferableBackbone,
                 device: str = "cpu",
                 pretrain_lr: float = 1e-3,
                 finetune_lr: float = 5e-4,
                 mmd_weight: float = 0.1,
                 batch_size: int = 64):
        self.backbone = backbone.to(device)
        self.device = device
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.mmd_weight = mmd_weight
        self.batch_size = batch_size
        self.mmd_loss = MMDLoss()

    # ──── Phase 1: QM9 Pre-training ────

    def pretrain_on_qm9(self, X_qm9: np.ndarray, y_qm9: np.ndarray,
                         max_epochs: int = 100, patience: int = 15) -> Dict:
        """
        Pre-train backbone on QM9 molecular descriptors.

        X_qm9: (n, d) — descriptors (same feature set as MR-TADF)
        y_qm9: (n, 5) — [HOMO, LUMO, gap, dipole, polarisability]
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Pre-training on QM9")
        logger.info("=" * 60)

        qm9_head = QM9Head(input_dim=self.backbone.output_dim).to(self.device)
        self.backbone.unfreeze_all()

        params = list(self.backbone.parameters()) + list(qm9_head.parameters())
        optimiser = torch.optim.AdamW(params, lr=self.pretrain_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max_epochs
        )

        # Split QM9 into train/val (90/10)
        n = len(X_qm9)
        perm = np.random.RandomState(42).permutation(n)
        split = int(0.9 * n)
        train_idx, val_idx = perm[:split], perm[split:]

        train_loader = self._make_loader(X_qm9[train_idx], y_qm9[train_idx], shuffle=True)
        val_loader = self._make_loader(X_qm9[val_idx], y_qm9[val_idx], shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0

        for epoch in range(max_epochs):
            # Train
            self.backbone.train()
            qm9_head.train()
            train_loss = 0
            n_batch = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                features = self.backbone(xb)
                pred = qm9_head(features)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                train_loss += loss.item()
                n_batch += 1
            scheduler.step()

            # Validate
            self.backbone.eval()
            qm9_head.eval()
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = qm9_head(self.backbone(xb))
                    val_loss += F.mse_loss(pred, yb).item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in self.backbone.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if epoch % 10 == 0:
                logger.info(f"  QM9 Epoch {epoch:3d} | train={train_loss/n_batch:.4f} val={avg_val:.4f}")

            if patience_count >= patience:
                logger.info(f"  QM9 pre-training early stop at epoch {epoch}")
                break

        if best_state:
            self.backbone.load_state_dict(best_state)

        logger.info(f"  QM9 pre-training complete. Best val loss: {best_val_loss:.5f}")
        return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1}

    # ──── Phase 2: Domain Adaptation ────

    def domain_adaptation(self, X_source: np.ndarray, X_target: np.ndarray,
                           max_epochs: int = 50) -> Dict:
        """
        Align backbone representations between QM9 and MR-TADF using MMD.

        During this phase, no property labels are used — only the
        feature distributions are aligned.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Domain Adaptation (MMD alignment)")
        logger.info("=" * 60)

        self.backbone.unfreeze_all()
        optimiser = torch.optim.AdamW(
            self.backbone.parameters(), lr=self.pretrain_lr * 0.1, weight_decay=1e-5
        )

        X_src_t = torch.tensor(X_source, dtype=torch.float32, device=self.device)
        X_tgt_t = torch.tensor(X_target, dtype=torch.float32, device=self.device)

        history = []

        for epoch in range(max_epochs):
            self.backbone.train()
            optimiser.zero_grad()

            # Sample mini-batches from both domains
            n_s = min(128, len(X_source))
            n_t = min(len(X_target), 128)
            src_idx = np.random.choice(len(X_source), n_s, replace=False)
            tgt_idx = np.random.choice(len(X_target), n_t, replace=len(X_target) < 128)

            feat_src = self.backbone(X_src_t[src_idx])
            feat_tgt = self.backbone(X_tgt_t[tgt_idx])

            mmd_val = self.mmd_loss(feat_src, feat_tgt)
            mmd_val.backward()
            optimiser.step()

            history.append(mmd_val.item())

            if epoch % 10 == 0:
                logger.info(f"  MMD Epoch {epoch:3d} | MMD={mmd_val.item():.6f}")

        logger.info(f"  Domain adaptation complete. Final MMD: {history[-1]:.6f}")
        return {"mmd_history": history, "final_mmd": history[-1]}

    # ──── Phase 3: Progressive Fine-tuning ────

    def finetune_on_tadf(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          max_epochs: int = 300, patience: int = 30,
                          scaler_y=None) -> Tuple[MRTADFHead, Dict]:
        """
        Progressive fine-tuning on MR-TADF data.

        Phase schedule:
          [0, E/3):     head only (backbone frozen)
          [E/3, 2E/3):  top 2 blocks + head
          [2E/3, E):    all parameters with discriminative LR
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Progressive Fine-tuning on MR-TADF")
        logger.info("=" * 60)

        tadf_head = MRTADFHead(input_dim=self.backbone.output_dim).to(self.device)
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        phase_boundary_1 = max_epochs // 3
        phase_boundary_2 = 2 * max_epochs // 3

        best_val_loss = float("inf")
        best_backbone_state = None
        best_head_state = None
        patience_count = 0

        # Task weights: emphasise T1-S1 and T2-S1
        task_weights = torch.tensor([3.0, 2.5, 1.5, 1.0, 1.0, 0.8],
                                     device=self.device)

        current_phase = -1
        optimiser = None

        for epoch in range(max_epochs):
            # ── Phase transitions ──
            new_phase = 0 if epoch < phase_boundary_1 else (
                1 if epoch < phase_boundary_2 else 2
            )

            if new_phase != current_phase:
                current_phase = new_phase

                if current_phase == 0:
                    logger.info(f"  Phase A (epoch {epoch}): head only, backbone frozen")
                    self.backbone.freeze_all()
                    optimiser = torch.optim.AdamW(
                        tadf_head.parameters(), lr=self.finetune_lr
                    )

                elif current_phase == 1:
                    logger.info(f"  Phase B (epoch {epoch}): top 2 blocks + head")
                    self.backbone.unfreeze_top_k(k=2)
                    param_groups = [
                        {"params": list(self.backbone.blocks[-2].parameters()),
                         "lr": self.finetune_lr * 0.3},
                        {"params": list(self.backbone.blocks[-1].parameters()),
                         "lr": self.finetune_lr * 0.5},
                        {"params": list(self.backbone.attention.parameters()),
                         "lr": self.finetune_lr},
                        {"params": list(tadf_head.parameters()),
                         "lr": self.finetune_lr},
                    ]
                    optimiser = torch.optim.AdamW(param_groups, weight_decay=1e-4)

                elif current_phase == 2:
                    logger.info(f"  Phase C (epoch {epoch}): all params, discriminative LR")
                    self.backbone.unfreeze_all()
                    layer_groups = self.backbone.get_layer_groups()
                    param_groups = []
                    for i, group_params in enumerate(layer_groups):
                        mult = self.LR_MULTIPLIERS[i] if i < len(self.LR_MULTIPLIERS) else 1.0
                        param_groups.append({
                            "params": group_params,
                            "lr": self.finetune_lr * mult,
                        })
                    param_groups.append({
                        "params": list(tadf_head.parameters()),
                        "lr": self.finetune_lr,
                    })
                    optimiser = torch.optim.AdamW(param_groups, weight_decay=1e-4)

            # ── Training step ──
            self.backbone.train()
            tadf_head.train()
            epoch_loss = 0
            n_batch = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                features = self.backbone(xb)
                pred = tadf_head(features)
                per_target_mse = ((pred - yb) ** 2).mean(dim=0)
                loss = (per_target_mse * task_weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(tadf_head.parameters()), 1.0
                )
                optimiser.step()
                epoch_loss += loss.item()
                n_batch += 1

            # ── Validation ──
            val_metrics = self._validate(
                tadf_head, val_loader, task_weights, scaler_y
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_backbone_state = {
                    k: v.cpu().clone() for k, v in self.backbone.state_dict().items()
                }
                best_head_state = {
                    k: v.cpu().clone() for k, v in tadf_head.state_dict().items()
                }
                patience_count = 0
            else:
                patience_count += 1

            if epoch % 15 == 0:
                logger.info(
                    f"  Epoch {epoch:3d} (phase {'ABC'[current_phase]}) | "
                    f"train={epoch_loss/n_batch:.4f} val={val_metrics['loss']:.4f} "
                    f"MAE_T1S1={val_metrics.get('mae_T1-S1', 0):.4f} "
                    f"MAE_T2S1={val_metrics.get('mae_T2-S1', 0):.4f}"
                )

            if patience_count >= patience:
                logger.info(f"  Fine-tuning early stop at epoch {epoch}")
                break

        # Restore best state
        if best_backbone_state:
            self.backbone.load_state_dict(best_backbone_state)
        if best_head_state:
            tadf_head.load_state_dict(best_head_state)

        final_metrics = self._validate(tadf_head, val_loader, task_weights, scaler_y)
        logger.info(f"  Final: val_loss={final_metrics['loss']:.4f}")

        return tadf_head, final_metrics

    # ──── Utility methods ────

    @torch.no_grad()
    def _validate(self, head, val_loader, task_weights, scaler_y=None) -> Dict:
        self.backbone.eval()
        head.eval()
        total_loss = 0
        all_pred, all_true = [], []
        n = 0
        for xb, yb in val_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            features = self.backbone(xb)
            pred = head(features)
            per_target_mse = ((pred - yb) ** 2).mean(dim=0)
            loss = (per_target_mse * task_weights).mean()
            total_loss += loss.item()
            all_pred.append(pred.cpu().numpy())
            all_true.append(yb.cpu().numpy())
            n += 1

        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        metrics = {"loss": total_loss / max(n, 1)}

        # Per-target MAE in original units
        if scaler_y is not None:
            pred_raw = scaler_y.inverse_transform(all_pred)
            true_raw = scaler_y.inverse_transform(all_true)
        else:
            pred_raw, true_raw = all_pred, all_true

        target_names = MRTADFHead.TADF_TARGETS
        for i, name in enumerate(target_names):
            metrics[f"mae_{name}"] = float(np.mean(np.abs(pred_raw[:, i] - true_raw[:, i])))
            ss_res = np.sum((true_raw[:, i] - pred_raw[:, i]) ** 2)
            ss_tot = np.sum((true_raw[:, i] - np.mean(true_raw[:, i])) ** 2)
            metrics[f"r2_{name}"] = float(1 - ss_res / max(ss_tot, 1e-8))

        return metrics

    def _make_loader(self, X, y, shuffle=True):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=shuffle, drop_last=False)


# ═══════════════════════════════════════════════════════════════════
#  6. CONVENIENCE: FULL TRANSFER-LEARNING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_transfer_learning_pipeline(
    qm9_descriptors: np.ndarray,
    qm9_properties: np.ndarray,
    tadf_X_train: np.ndarray,
    tadf_y_train: np.ndarray,
    tadf_X_val: np.ndarray,
    tadf_y_val: np.ndarray,
    input_dim: int,
    scaler_y=None,
    device: str = "cpu",
) -> Tuple[TransferableBackbone, MRTADFHead, Dict]:
    """
    End-to-end convenience function for the complete transfer pipeline.

    Returns: (backbone, tadf_head, metrics_dict)
    """
    backbone = TransferableBackbone(input_dim=input_dim)

    trainer = TransferLearningTrainer(
        backbone=backbone,
        device=device,
        pretrain_lr=1e-3,
        finetune_lr=5e-4,
        batch_size=64,
    )

    # Phase 1
    qm9_metrics = trainer.pretrain_on_qm9(
        qm9_descriptors, qm9_properties,
        max_epochs=100, patience=15,
    )

    # Phase 2
    all_tadf_X = np.vstack([tadf_X_train, tadf_X_val])
    mmd_metrics = trainer.domain_adaptation(
        qm9_descriptors, all_tadf_X, max_epochs=50,
    )

    # Phase 3
    tadf_head, ft_metrics = trainer.finetune_on_tadf(
        tadf_X_train, tadf_y_train,
        tadf_X_val, tadf_y_val,
        max_epochs=300, patience=30,
        scaler_y=scaler_y,
    )

    all_metrics = {
        "qm9_pretrain": qm9_metrics,
        "domain_adaptation": {"final_mmd": mmd_metrics["final_mmd"]},
        "finetune": ft_metrics,
    }

    return backbone, tadf_head, all_metrics
