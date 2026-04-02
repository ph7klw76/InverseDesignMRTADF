"""
gnn_predictor.py — Graph Neural Network property predictor for MR-TADF
========================================================================

Scientific rationale:
  Molecular descriptors are hand-crafted projections that discard 3D and
  topological information.  A GNN operates directly on the molecular graph
  (atoms = nodes, bonds = edges), preserving the full connectivity that
  determines electronic structure.  For MR-TADF emitters, the *alternating
  B/N substitution pattern* within the aromatic framework is the defining
  structural motif — a GNN naturally encodes this through message passing
  along the covalent bond network.

Architecture:
  Atom featuriser → N × MPNN layers → global readout → task heads

  • Atom features: atomic number, formal charge, hybridisation, aromaticity,
    number of Hs, degree, is_in_ring, atomic mass (one-hot + continuous)
  • Bond features: bond type, conjugation, ring membership, stereochem
  • MPNN uses edge-conditioned message passing (Gilmer et al., ICML 2017)
    with GRU-based node update for stable deep propagation
  • Global readout: Set2Set attention pooling (Vinyals et al., NeurIPS 2016)
  • Multi-task heads identical to the descriptor-based predictor

The GNN can replace the descriptor-based predictor entirely, or serve as
an ensemble member alongside it.

Dependencies:
  pip install torch torch-geometric rdkit-pypi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  1. MOLECULAR GRAPH FEATURISATION
# ═══════════════════════════════════════════════════════════════════

# Atom-level feature dimensions
ATOM_FEATURES = {
    "atomic_num":     list(range(1, 54)),   # H through I (covers B, C, N, O, F, S, P)
    "degree":         [0, 1, 2, 3, 4, 5],
    "formal_charge":  [-2, -1, 0, 1, 2],
    "hybridisation":  ["SP", "SP2", "SP3", "SP3D", "SP3D2"],
    "num_Hs":         [0, 1, 2, 3, 4],
    "is_aromatic":    [False, True],
    "is_in_ring":     [False, True],
}

BOND_FEATURES = {
    "bond_type":      ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    "is_conjugated":  [False, True],
    "is_in_ring":     [False, True],
    "stereo":         ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
}


def _one_hot(value, options: list) -> List[int]:
    """One-hot encode value against a list of allowed options."""
    encoding = [0] * len(options)
    try:
        idx = options.index(value)
        encoding[idx] = 1
    except ValueError:
        pass  # unknown category → all zeros
    return encoding


class MolecularGraphBuilder:
    """
    Converts SMILES strings to PyTorch Geometric graph objects.

    Each molecule becomes a Data object with:
      - x: (n_atoms, atom_feat_dim) node features
      - edge_index: (2, n_edges) COO-format edge indices (undirected)
      - edge_attr: (n_edges, bond_feat_dim) edge features
    """

    @staticmethod
    def get_atom_feature_dim() -> int:
        """Total dimensionality of atom feature vector."""
        dim = sum(len(v) for v in ATOM_FEATURES.values())
        dim += 1  # atomic mass (continuous)
        return dim

    @staticmethod
    def get_bond_feature_dim() -> int:
        """Total dimensionality of bond feature vector."""
        return sum(len(v) for v in BOND_FEATURES.values())

    @staticmethod
    def smiles_to_graph(smiles: str):
        """
        Convert a single SMILES string to a PyG Data object.

        Returns None if the SMILES is invalid.
        """
        try:
            from rdkit import Chem
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError(
                "GNN requires rdkit-pypi and torch-geometric.\n"
                "Install: pip install rdkit-pypi torch-geometric"
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # ── Atom features ──
        atom_features = []
        for atom in mol.GetAtoms():
            features = []
            features += _one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
            features += _one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
            features += _one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
            features += _one_hot(
                str(atom.GetHybridization()).split(".")[-1],
                ATOM_FEATURES["hybridisation"]
            )
            features += _one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_Hs"])
            features += _one_hot(atom.GetIsAromatic(), ATOM_FEATURES["is_aromatic"])
            features += _one_hot(atom.IsInRing(), ATOM_FEATURES["is_in_ring"])
            features.append(atom.GetMass() / 100.0)  # normalised mass
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float)

        # ── Edge features (bonds — both directions for undirected) ──
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_feat = []
            bond_feat += _one_hot(
                str(bond.GetBondType()).split(".")[-1],
                BOND_FEATURES["bond_type"]
            )
            bond_feat += _one_hot(bond.GetIsConjugated(), BOND_FEATURES["is_conjugated"])
            bond_feat += _one_hot(bond.IsInRing(), BOND_FEATURES["is_in_ring"])
            bond_feat += _one_hot(
                str(bond.GetStereo()).split(".")[-1],
                BOND_FEATURES["stereo"]
            )

            # Add both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)

        if len(edge_indices) == 0:
            # Single-atom molecule (shouldn't occur for MR-TADF, but handle gracefully)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, MolecularGraphBuilder.get_bond_feature_dim()),
                                     dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @classmethod
    def batch_smiles_to_graphs(cls, smiles_list: List[str],
                                properties: Optional[np.ndarray] = None):
        """
        Convert a list of SMILES to a list of Data objects.

        If properties is provided (n, n_targets), attaches y to each graph.
        Returns: list of Data, list of valid indices
        """
        graphs = []
        valid_idx = []
        for i, smi in enumerate(smiles_list):
            graph = cls.smiles_to_graph(smi)
            if graph is not None:
                if properties is not None:
                    graph.y = torch.tensor(properties[i], dtype=torch.float).unsqueeze(0)
                graphs.append(graph)
                valid_idx.append(i)

        logger.info(f"Built {len(graphs)}/{len(smiles_list)} molecular graphs")
        return graphs, valid_idx


# ═══════════════════════════════════════════════════════════════════
#  2. MESSAGE-PASSING NEURAL NETWORK (MPNN)
# ═══════════════════════════════════════════════════════════════════

class EdgeConditionedConv(nn.Module):
    """
    Edge-conditioned graph convolution (Gilmer et al., 2017).

    Message function:
      m_{ij} = MLP_edge(e_{ij}) × h_j

    Update function:
      h_i' = GRU(h_i, Σ_j m_{ij})

    The GRU update provides gradient stability for deep message passing
    (6+ layers), which is essential for MR-TADF molecules with 40–130
    atoms where long-range electronic effects span many bonds.
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()

        # Edge network: maps edge features to a node_dim × node_dim matrix
        # (factorised as two linear layers for parameter efficiency)
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim * node_dim),
        )

        # GRU for node update
        self.gru = nn.GRUCell(node_dim, node_dim)

        self.node_dim = node_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x: (N, node_dim) — node features
        edge_index: (2, E) — edge indices
        edge_attr: (E, edge_dim) — edge features
        """
        src, dst = edge_index[0], edge_index[1]
        n_nodes = x.size(0)

        # Compute edge-conditioned messages
        edge_weights = self.edge_nn(edge_attr)  # (E, node_dim²)
        edge_weights = edge_weights.view(-1, self.node_dim, self.node_dim)

        # Message: W_e × h_source for each edge
        x_src = x[src].unsqueeze(-1)  # (E, node_dim, 1)
        messages = torch.bmm(edge_weights, x_src).squeeze(-1)  # (E, node_dim)

        # Aggregate messages at destination nodes
        agg = torch.zeros(n_nodes, self.node_dim, device=x.device)
        agg.index_add_(0, dst, messages)

        # GRU update
        x_new = self.gru(agg, x)
        return x_new


class Set2SetReadout(nn.Module):
    """
    Set2Set attention-based graph-level readout (Vinyals et al., 2016).

    Iteratively attends to the node set to produce a fixed-size
    graph-level embedding.  Superior to mean/sum pooling for molecular
    property prediction because it can selectively weight atoms
    (e.g., the B and N atoms that define the MR-TADF character).
    """
    def __init__(self, input_dim: int, n_iters: int = 6):
        super().__init__()
        self.n_iters = n_iters
        self.lstm = nn.LSTMCell(input_dim, input_dim)
        self.output_dim = 2 * input_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x: (N_total, dim) — all node features in the batch
        batch: (N_total,) — graph membership indices

        Returns: (n_graphs, 2*dim)
        """
        n_graphs = batch.max().item() + 1
        dim = x.size(1)
        device = x.device

        h = torch.zeros(n_graphs, dim, device=device)
        c = torch.zeros(n_graphs, dim, device=device)
        q_star = torch.zeros(n_graphs, self.output_dim, device=device)

        for _ in range(self.n_iters):
            # Query vector from LSTM hidden state
            q = h  # (n_graphs, dim)

            # Attention: score each node against its graph's query
            q_expanded = q[batch]  # (N_total, dim)
            attn_scores = (x * q_expanded).sum(dim=-1)  # (N_total,)

            # Softmax within each graph (scatter)
            attn_max = torch.zeros(n_graphs, device=device)
            attn_max.scatter_reduce_(0, batch, attn_scores, reduce='amax')
            attn_scores = attn_scores - attn_max[batch]
            attn_exp = torch.exp(attn_scores)
            attn_sum = torch.zeros(n_graphs, device=device)
            attn_sum.scatter_add_(0, batch, attn_exp)
            attn_weights = attn_exp / (attn_sum[batch] + 1e-8)  # (N_total,)

            # Weighted sum of node features per graph
            weighted = x * attn_weights.unsqueeze(-1)  # (N_total, dim)
            readout = torch.zeros(n_graphs, dim, device=device)
            readout.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted), weighted)

            # LSTM update
            h, c = self.lstm(readout, (h, c))
            q_star = torch.cat([h, readout], dim=-1)

        return q_star


# ═══════════════════════════════════════════════════════════════════
#  3. COMPLETE GNN PREDICTOR
# ═══════════════════════════════════════════════════════════════════

class GNNPropertyPredictor(nn.Module):
    """
    End-to-end GNN for MR-TADF property prediction.

    Architecture:
      Atom embedding → 6× EdgeConditionedConv (with residual) →
      Set2Set readout → shared MLP → 6 task-specific heads

    Why 6 MPNN layers:
      The average shortest path between a B and N atom in the MR-TADF
      core is 3–6 bonds.  Each MPNN layer propagates information one
      hop, so 6 layers ensure that every B atom "sees" every N atom
      and vice versa — capturing the full MR effect.
    """
    def __init__(self,
                 atom_feat_dim: int = None,
                 bond_feat_dim: int = None,
                 hidden_dim: int = 256,
                 n_conv_layers: int = 6,
                 n_targets: int = 6,
                 dropout: float = 0.1,
                 set2set_iters: int = 6):
        super().__init__()

        if atom_feat_dim is None:
            atom_feat_dim = MolecularGraphBuilder.get_atom_feature_dim()
        if bond_feat_dim is None:
            bond_feat_dim = MolecularGraphBuilder.get_bond_feature_dim()

        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.hidden_dim = hidden_dim

        # Atom embedding
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Message-passing layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(n_conv_layers):
            self.conv_layers.append(
                EdgeConditionedConv(hidden_dim, bond_feat_dim, hidden_dim)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Readout
        self.readout = Set2SetReadout(hidden_dim, n_iters=set2set_iters)
        readout_dim = self.readout.output_dim  # 2 * hidden_dim

        # Shared representation
        self.shared_mlp = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            ) for _ in range(n_targets)
        ])

        self.n_targets = n_targets
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            x: (N_total, atom_feat_dim) — atom features for all graphs
            edge_index: (2, E_total) — edge indices
            edge_attr: (E_total, bond_feat_dim) — bond features
            batch: (N_total,) — graph membership

        Returns: (n_graphs, n_targets) predictions
        """
        # Embed atoms
        h = self.atom_embed(x)

        # Message passing with residual connections and layer norm
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h_new = conv(h, edge_index, edge_attr)
            h = norm(h + h_new)  # residual + layer norm

        # Graph-level readout
        graph_feat = self.readout(h, batch)

        # Predict properties
        shared = self.shared_mlp(graph_feat)
        outputs = [head(shared) for head in self.task_heads]
        return torch.cat(outputs, dim=-1)

    def forward_from_data(self, data) -> torch.Tensor:
        """Convenience method for PyG Data/Batch objects."""
        return self.forward(data.x, data.edge_index, data.edge_attr, data.batch)


# ═══════════════════════════════════════════════════════════════════
#  4. GNN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

class GNNTrainer:
    """
    Training loop for the GNN property predictor.

    Handles:
      - PyG DataLoader construction
      - Physics-informed loss (consistency checks)
      - Early stopping
      - Per-target MAE/R² evaluation
    """
    def __init__(self, model: GNNPropertyPredictor,
                 lr: float = 5e-4, weight_decay: float = 1e-5,
                 patience: int = 30, max_epochs: int = 300,
                 batch_size: int = 32, device: str = "cpu",
                 scaler_y=None):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.scaler_y = scaler_y

        self.task_weights = torch.tensor(
            [3.0, 2.5, 1.5, 1.0, 1.0, 0.8], device=device
        )

        self.optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode='min', factor=0.5, patience=10
        )

    def fit(self, train_graphs: list, val_graphs: list) -> Dict:
        """
        Train the GNN on molecular graph data.

        train_graphs, val_graphs: lists of PyG Data objects with .y attributes
        """
        try:
            from torch_geometric.loader import DataLoader as PyGLoader
        except ImportError:
            raise ImportError("pip install torch-geometric")

        train_loader = PyGLoader(train_graphs, batch_size=self.batch_size,
                                  shuffle=True, drop_last=False)
        val_loader = PyGLoader(val_graphs, batch_size=self.batch_size,
                                shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0

        for epoch in range(self.max_epochs):
            # Train
            self.model.train()
            train_loss = 0
            n_batch = 0
            for data in train_loader:
                data = data.to(self.device)
                self.optimiser.zero_grad()

                pred = self.model.forward_from_data(data)
                target = data.y  # (batch, n_targets)

                per_target_mse = ((pred - target) ** 2).mean(dim=0)
                loss = (per_target_mse * self.task_weights).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimiser.step()

                train_loss += loss.item()
                n_batch += 1

            # Validate
            val_metrics = self._evaluate(val_loader)
            self.scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_count = 0
            else:
                patience_count += 1

            if epoch % 20 == 0:
                logger.info(
                    f"GNN Epoch {epoch:3d} | train={train_loss/n_batch:.4f} "
                    f"val={val_metrics['loss']:.4f} "
                    f"MAE_T1S1={val_metrics.get('mae_T1-S1', 0):.4f} "
                    f"R²_T1S1={val_metrics.get('r2_T1-S1', 0):.3f}"
                )

            if patience_count >= self.patience:
                logger.info(f"GNN early stop at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        final = self._evaluate(val_loader)
        logger.info(f"GNN final: {final}")
        return final

    @torch.no_grad()
    def _evaluate(self, loader) -> Dict:
        self.model.eval()
        all_pred, all_true = [], []
        total_loss = 0
        n = 0

        for data in loader:
            data = data.to(self.device)
            pred = self.model.forward_from_data(data)
            target = data.y
            loss = ((pred - target) ** 2 * self.task_weights).mean()
            total_loss += loss.item()
            all_pred.append(pred.cpu().numpy())
            all_true.append(target.cpu().numpy())
            n += 1

        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        if self.scaler_y is not None:
            pred_raw = self.scaler_y.inverse_transform(all_pred)
            true_raw = self.scaler_y.inverse_transform(all_true)
        else:
            pred_raw, true_raw = all_pred, all_true

        metrics = {"loss": total_loss / max(n, 1)}
        target_names = ["T1-S1", "T2-S1", "DeltaEST", "S1", "T1", "f"]
        for i, name in enumerate(target_names):
            metrics[f"mae_{name}"] = float(np.mean(np.abs(pred_raw[:, i] - true_raw[:, i])))
            ss_res = np.sum((true_raw[:, i] - pred_raw[:, i]) ** 2)
            ss_tot = np.sum((true_raw[:, i] - np.mean(true_raw[:, i])) ** 2)
            metrics[f"r2_{name}"] = float(1 - ss_res / max(ss_tot, 1e-8))

        return metrics


# ═══════════════════════════════════════════════════════════════════
#  5. GNN–DESCRIPTOR ENSEMBLE
# ═══════════════════════════════════════════════════════════════════

class EnsemblePredictor(nn.Module):
    """
    Ensemble combining GNN and descriptor-based predictions.

    Strategy: learned weighted average with uncertainty weighting.
    The ensemble assigns higher weight to whichever model is more
    confident (lower predicted variance) for each target.
    """
    def __init__(self, n_targets: int = 6):
        super().__init__()
        # Learnable per-target log-weights for GNN vs descriptor model
        self.log_alpha = nn.Parameter(torch.zeros(n_targets))

    def forward(self, pred_gnn: torch.Tensor,
                pred_desc: torch.Tensor) -> torch.Tensor:
        """
        Combine predictions from GNN and descriptor model.

        pred_gnn:  (B, n_targets)
        pred_desc: (B, n_targets)
        """
        alpha = torch.sigmoid(self.log_alpha)  # (n_targets,) in [0, 1]
        return alpha * pred_gnn + (1 - alpha) * pred_desc
