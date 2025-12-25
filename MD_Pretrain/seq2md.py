import torch
import torch.nn as nn
import torch.nn.functional as F
import esm  
from typing import Optional, Tuple, List
from torch import Tensor
from EGNN_SE3 import EGNN as EGNNSE3
from torch_geometric.utils import to_dense_batch
from torch_cluster import knn_graph


###############################################################################
# 1) Basic Modules
###############################################################################
class MLP(nn.Module):
    """
    3-layer MLP with optional BatchNorm in the middle layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation=F.relu, bn=True):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)
        self.bn = bn
        if self.bn:
            self.bn_layer = nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.997)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [..., input_dim]
        Returns: [..., output_dim]
        """
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        if self.bn:
            # For shape [N, hidden_dim]
            orig_shape = x.shape
            x = x.reshape(-1, x.size(-1))  # flatten
            x = self.bn_layer(x)
            x = x.reshape(*orig_shape)
        x = self.activation(x)
        x = self.lin3(x)
        return x


def remap_batch_indices(b_flat: Tensor) -> Tensor:
    """
    Re-label batch indices in [0..(local_batch-1)], safe for multi-GPU sub-batching.
    """
    unique_vals = b_flat.unique(sorted=True)
    mapping = {}
    for new_id, old_id in enumerate(unique_vals):
        mapping[int(old_id.item())] = int(new_id)
    b_new = b_flat.clone()
    for i in range(b_new.size(0)):
        b_new[i] = mapping[int(b_new[i].item())]
    return b_new


###############################################################################
# 2) ESM-based Chain Embedding
###############################################################################
class ESMChainAEmbedder:
    """
    Wraps a pretrained ESM model (esm2_t33_650M_UR50D) to embed chain A or B, etc.
    Returns [1, L, 1280] for a single chain. Usually you'll do (B, L, 1280).
    """
    def __init__(self, device='cuda'):
        self.device = device
        # If you have ESM installed:
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed_chain(self, chain_seq: str) -> torch.Tensor:
        """
        chain_seq: e.g. "MKVL..."
        returns => [1, L, 1280]
        """
        data = [("chainA", chain_seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[33], need_head_weights=False)
        token_reprs = results["representations"][33]  # [1, L+2, d_esm]
        embedding = token_reprs[:, 1:-1, :]           # remove [CLS], [EOS]
        return embedding  # shape [1, L, 1280]

###############################################################################
# 4) ProteinGeomEncoder
###############################################################################
class ProteinGeomEncoder(nn.Module):
    """
    Encodes protein geometry using EGNN (imported as EGNNSE3) with mean pooling over frames.
    Takes in h_init (sequence + positional features) and coords_ca (Cα coordinates).
    """

    def __init__(self, hidden_dim=128, egnn_layers=2):
        """
        Args:
            hidden_dim (int): Dimensionality of node embeddings.
            egnn_layers (int): Number of EGNN layers to use.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.egnn_layers = egnn_layers

        # EGNN layers
        self.egnn_blocks = nn.ModuleList([
            EGNNSE3(
                in_node_nf=hidden_dim,
                hidden_nf=hidden_dim,
                out_node_nf=hidden_dim,
                in_edge_nf=hidden_dim,
                attention=False,  # Adjust as needed
                reflection_equiv=False  # Adjust as needed
            )
            for _ in range(egnn_layers)
        ])

        # LayerNorm after EGNN layers
        self.norm_blocks = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(egnn_layers)
        ])

    def forward(self, h_init, coords_ca, mask_2d):
        """
        Args:
            h_init (Tensor): Node embeddings [B, F, N, hidden_dim].
            coords_ca (Tensor): C coordinates [B, F, N, 3].
            mask_2d (Tensor): Valid mask [B, F, N].

        Returns:
            geom_emb (Tensor): Fused geometric embedding [B, N, hidden_dim].
        """
        B, F, N, _ = h_init.shape
        device = h_init.device

        # Initialize h_enc with h_init
        h_enc = h_init

        # Iterate over EGNN layers
        for l_idx in range(self.egnn_layers):
            new_frames_h = []  # To store processed frames for this layer
            block = self.egnn_blocks[l_idx]

            for fidx in range(F):
                h_f = h_enc[:, fidx, :, :]      # [B, N, hidden_dim]
                c_f = coords_ca[:, fidx, :, :] # [B, N, 3]
                mask_f = mask_2d[:, fidx, :]   # [B, N]

                valid_idx = mask_f.nonzero(as_tuple=False)  # Extract valid indices
                if valid_idx.numel() == 0:
                    # If no valid nodes, keep zeros
                    new_frames_h.append(torch.zeros_like(h_f))
                    continue

                # Flatten batch dimension
                indices = valid_idx[:, 0] * N + valid_idx[:, 1]
                indices = indices[indices < B * N]  # Safe indexing
                h_flat = h_f.reshape(-1, self.hidden_dim)[indices]
                c_flat = c_f.reshape(-1, 3)[indices]
                b_flat = torch.div(indices, N, rounding_mode='floor')  # Batch indices

                if c_flat.size(0) < 5:
                    # Skip EGNN if fewer than 5 points
                    dense_h = torch.zeros((B, N, self.hidden_dim), device=device)
                else:
                    # Process using EGNNSE3
                    edge_idx = None  # Let EGNNSE3 compute edges internally
                    h_out, coords_out = block(h_flat, c_flat, edge_idx, b_flat)

                    # Convert back to dense format
                    dense_h, _ = to_dense_batch(h_out, b_flat)

                    # Pad if necessary
                    if dense_h.size(0) < B:
                        pad_h = torch.zeros((B - dense_h.size(0), N, self.hidden_dim), device=device)
                        dense_h = torch.cat([dense_h, pad_h], dim=0)

                new_frames_h.append(dense_h)

            # Stack processed frames and apply normalization
            h_enc = torch.stack(new_frames_h, dim=1)  # [B, F, N, hidden_dim]
            h_enc = self.norm_blocks[l_idx](h_enc)

        # Mean pooling over frames
        geom_emb = h_enc.mean(dim=1)  # [B, N, hidden_dim]

        return geom_emb

class ProClipMDModel(nn.Module):
    """
    ProClipMD Model for contrastive learning between sequence and geometry embeddings.
    Core steps:
      1. Embed protein sequence at the residue level using ESM.
      2. Encode protein geometry (10 frames) using EGNN.
      3. Align sequence and geometry embeddings via projection for contrastive learning.
    """

    def __init__(self, config):
        super().__init__()
        model_conf = config.get('model', {})
        self.seq_dim_in = model_conf.get('seq_dim_in', 1280)  # ESM2 embedding dimension
        self.geom_dim = model_conf.get('geom_dim', 256)       # Geometry embedding dimension

        # 1) Sequence embedding projection: [1280 -> geom_dim]
        self.seq_project = nn.Linear(self.seq_dim_in, self.geom_dim)
        self.seq_ln = nn.LayerNorm(self.geom_dim)

        # 2) Geometry encoder using EGNN
        self.geom_encoder = ProteinGeomEncoder(
            hidden_dim=self.geom_dim,
            egnn_layers=model_conf.get('egnn_layers', 2)
        )

        # 3) Shared projection (optional): Map both embeddings to a common latent space
        self.shared_project = nn.Linear(self.geom_dim, self.geom_dim)

    def forward(
        self,
        seq_esm: Tensor,       # [B, N_seq, seq_dim_in] - Residue-level ESM embeddings
        coords_ca: Tensor,     # [B, frames=10, N_ca, 3] - Cα coordinates for 10 frames
        mask_2d: Optional[Tensor] = None  # [B, frames, N_ca] - Valid mask (optional)
    ):
        """
        Args:
            seq_esm (Tensor): Protein sequence embedding from ESM. [B, N_seq, seq_dim_in].
            coords_ca (Tensor): Cα coordinates for protein. [B, 10, N_ca, 3].
            mask_2d (Tensor): Boolean mask for valid residues. [B, 10, N_ca].

        Returns:
            dict: Contains global embeddings for sequence and geometry.
        """
        device = seq_esm.device
        B, N_seq, _ = seq_esm.shape
        B2, Ff, Nc, _ = coords_ca.shape

        assert B == B2, "Batch size mismatch between sequence and geometry inputs."

        # 1) Sequence projection to geom_dim
        seq_emb = self.seq_project(seq_esm)       # [B, N_seq, geom_dim]
        seq_emb = self.seq_ln(seq_emb)           # Apply LayerNorm

        # 2) Geometry encoding with EGNN
        if mask_2d is None:
            mask_2d = torch.ones(B, Ff, Nc, dtype=torch.bool, device=device)  # Default mask if none provided
        geom_emb = self.geom_encoder(seq_emb, coords_ca, mask_2d)  # [B, N_ca, geom_dim]

        # 3) Global pooling for contrastive embeddings
        seq_global = seq_emb.mean(dim=1)  # [B, geom_dim]
        geom_global = geom_emb.mean(dim=1)  # [B, geom_dim]

        # 4) Shared projection (optional)
        seq_global = self.shared_project(seq_global)  # [B, geom_dim]
        geom_global = self.shared_project(geom_global)  # [B, geom_dim]

        # Return embeddings
        return {
            "seq_global": seq_global,       # Global sequence embedding [B, geom_dim]
            "geom_global": geom_global,     # Global geometry embedding [B, geom_dim]
        }