import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from typing import Optional, Tuple, List
from torch import Tensor
from torch_geometric.utils import to_dense_batch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_cluster import knn_graph
from egnn_pytorch import EGNN_Network

###############################################################################
# 1) Basic Modules
###############################################################################

def grad_checkpoint(func, args, checkpointing=False):
    if checkpointing:
        return checkpoint(func, *args, use_reentrant=False)
    else:
        return func(*args)

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
    Wraps a pretrained ESM model (esm2_t33_650M_UR50D) to embed a protein chain.
    Returns [1, L, 2560] for one sequence; typically you batch multiple sequences.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
    
    def embed_batch(self, seq_list: List[str], max_ca: int) -> torch.Tensor:
        """Batch processing of sequences"""
        # Convert sequences to ESM input format
        data = [("seq{}".format(i), seq) for i, seq in enumerate(seq_list)]
        
        # Batch conversion
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[33], return_contacts=False)
        
        # Extract embeddings and truncate
        embeddings = results["representations"][33][:, 1:-1, :]  # [B, L, 2560]
        embeddings = embeddings[:, :max_ca, :]                   # [B, max_ca, 2560]
        
        return embeddings

    def embed_chain(self, chain_seq: str, max_ca: int) -> torch.Tensor:
        """
        chain_seq: e.g. "MKVL..."
        returns => [1, L, 2560]
        """
        data = [("chainA", chain_seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(next(self.model.parameters()).device)
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[33], need_head_weights=False)
        token_reprs = results["representations"][33]  # [1, L+2, 2560]
        embedding = token_reprs[:, 1:-1, :]           # remove [CLS], [EOS]
        embedding = embedding[:, :max_ca, :]
        return embedding
        
class ProteinDataset(Dataset):
    """
    A dataset that reads protein sequences and coordinates from an HDF5 file
    and generates:
      - feats: node features of shape [B, max_ca, feature_dim]
      - coors: node coordinates of shape [B, max_ca, 3]
      - mask:  boolean mask of shape [B, max_ca]
    where B = number of proteins, max_ca = the maximum number of residues
    (or CA atoms) across all proteins in the dataset.
    """

    def __init__(self, h5_path, node_initializer):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            node_initializer (ProteinNodeInitializer): A class instance
                that can encode amino acid sequences into numerical features.
        """
        super().__init__()
        self.h5_path = h5_path
        self.node_initializer = node_initializer

        # Lists to store each protein's sequence and coordinates
        self.sequences = []   # List[str]
        self.all_coords = []  # List[np.ndarray of shape (n, 3)]

        # Read each group from the HDF5 file
        with h5py.File(self.h5_path, 'r') as f:
            for group_name in f.keys():
                group = f[group_name]
                coords = np.array(group["representative_frames"])  # shape: (n, 3)
                seq = group.attrs["sequence"]                      # e.g. "MKVL..." of length n

                self.sequences.append(seq)
                self.all_coords.append(coords)

        # Number of proteins
        self.num_proteins = len(self.sequences)

        # Determine maximum length (max_ca) among all sequences
        ori_max_ca = max(len(seq) for seq in self.sequences)
        self.max_ca = min(ori_max_ca, 1024)

        # Pre-encode everything so __getitem__ is faster
        self.feats_tensor = self._encode_node_features(self.sequences, self.max_ca)
        self.coords_tensor = self._pad_coords(self.all_coords, self.max_ca)
        self.mask_tensor = self._create_mask(self.sequences, self.max_ca)

    def _encode_node_features(self, seq_list, max_ca):

        return self.node_initializer.encode_sequences(seq_list, max_ca)

    def _pad_coords(self, coords_list, max_ca):
        """
        Pad each protein's coordinates from [n, 3] up to [max_ca, 3],
        truncating if n > max_ca. Returns a tensor of shape [B, max_ca, 3].
        """
        B = len(coords_list)
        coords_padded = torch.zeros((B, max_ca, 3), dtype=torch.float32)
        for i, coords_np in enumerate(coords_list):
            n = coords_np.shape[0]
            length = min(n, max_ca)
            coords_padded[i, :length, :] = torch.from_numpy(coords_np[:length, :])
        return coords_padded

    def _create_mask(self, seq_list, max_ca):
        """
        Create a boolean mask of shape [B, max_ca],
        where True = valid residue, False = padded residue.
        """
        B = len(seq_list)
        mask = torch.zeros((B, max_ca), dtype=torch.bool)
        for i, seq in enumerate(seq_list):
            n = len(seq)
            length = min(n, max_ca)
            mask[i, :length] = True
        return mask

    def __len__(self):
        return self.num_proteins

    def __getitem__(self, idx):
        """
        Return (feats, coors, mask) for the protein at index idx:
          - feats shape:  [max_ca, feature_dim]
          - coors shape:  [max_ca, 3]
          - mask shape:   [max_ca]
        """
        feats = self.feats_tensor[idx]   # [max_ca, feature_dim]
        coors = self.coords_tensor[idx]  # [max_ca, 3]
        mask = self.mask_tensor[idx]     # [max_ca]
        return feats, coors, mask


########################################################
# 2) ProteinNodeInitializer: encode sequences into feats
########################################################

class ProteinNodeInitializer:
    """
    Converts a protein sequence (string of single-letter amino acids) into
    a numerical feature array combining one-hot encoding (20 possible AAs)
    and various physicochemical properties.
    """

    def __init__(self, hidden_dim=512):
        """
        Args:
            hidden_dim (int): If you plan to project the [26]-dim features
                              (20 one-hot + 6 properties) into a new space.
        """
        self.aa_features = {
            'hydrophobicity': {
                'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
                'W': -0.9, 'G': -0.4, 'T': -0.7, 'S': -0.8, 'Y': -1.3, 'P': -1.6, 'H': -3.2,
                'N': -3.5, 'D': -3.5, 'Q': -3.5, 'E': -3.5, 'K': -3.9, 'R': -4.5, 'others': 0
            },
            'charge': {
                'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1, 'others': 0
            },
            'polarity': {
                'R': 1, 'N': 1, 'D': 1, 'Q': 1, 'E': 1, 'H': 1, 'K': 1,
                'S': 1, 'T': 1, 'Y': 1, 'others': 0
            },
            'acceptor': {
                'D': 1, 'E': 1, 'N': 1, 'Q': 1, 'others': 0
            },
            'donor': {
                'R': 1, 'K': 1, 'W': 1, 'N': 1, 'Q': 1, 'H': 1, 'S': 1,
                'T': 1, 'Y': 1, 'others': 0
            },
            'disulfide bond': {
                'C': 1, 'others': 0
            }
        }

        # Map from single-letter amino acid to an integer index for one-hot
        self.aa2idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6,
            'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13,
            'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }

        self.hidden_dim = hidden_dim
        # Optional projection layer if desired
        self.linear_proj = torch.nn.Linear(26, hidden_dim)

    def encode_sequences(self, seq_list, max_ca):
        """
        Converts a list of protein sequences into a feature tensor of shape
        [B, max_ca, 26], combining one-hot + 6 physicochemical features.
        """
        B = len(seq_list)
        onehot_features = torch.zeros((B, max_ca, 20), dtype=torch.float32)
        phys_features = torch.zeros((B, max_ca, len(self.aa_features)), dtype=torch.float32)

        for b, seq in enumerate(seq_list):
            for i, aa in enumerate(seq[:max_ca]):
                aa_upper = aa.upper()
                # One-hot
                if aa_upper in self.aa2idx:
                    onehot_features[b, i, self.aa2idx[aa_upper]] = 1.0
                # Physicochemical properties
                for j, (prop_name, prop_dict) in enumerate(self.aa_features.items()):
                    if aa_upper in prop_dict:
                        phys_features[b, i, j] = prop_dict[aa_upper]
                    else:
                        phys_features[b, i, j] = prop_dict['others']

        # Concatenate [20 one-hot + 6 properties] = 26
        combined_features = torch.cat([onehot_features, phys_features], dim=-1)  # [B, max_ca, 26]
        return combined_features

class ProClipMD(nn.Module):
    def __init__(
        self,
        seq_dim_in: int = 2560,   # ESM embedding dimension
        feats_in_dim: int = 26,
        egnn_dim: int = 128,       # dimension expected by EGNN for node feats
        d_projection: int = 512,   # final projection dimension for both seq + geom
        egnn_depth: int = 4,      # number of EGNN layers
        num_nearest_neighbors: int = 64,
        coors_clamp: float = 2.0
    ):
        super().__init__()

        # 1) Linear projection: ESM embeddings -> d_projection
        self.seq_project = nn.Linear(seq_dim_in, d_projection)
        self.feats_linear_proj = nn.Linear(feats_in_dim, egnn_dim)

        # 2) EGNN for geometry
        #    - 'dim=egnn_dim' means the EGNN expects feats shaped [B, N, egnn_dim]
        self.egnn = EGNN_Network(
            num_tokens=None,             # no embedding lookup, we feed float feats
            dim=egnn_dim,
            depth=egnn_depth,
            num_nearest_neighbors=num_nearest_neighbors,
            norm_coors=True,
            coor_weights_clamp_value=coors_clamp
        )

        # 3) Linear projection: EGNN output (size = egnn_dim) -> d_projection
        self.geom_project = nn.Linear(egnn_dim, d_projection)

    def forward(
        self,
        seq_esm: torch.Tensor,       # [B, N, 2560]  (ESM residue-level embeddings)
        feats: torch.Tensor,         # [B, N, egnn_dim] (node features from ProteinDataset + optional projection)
        coors: torch.Tensor,         # [B, N, 3]        (residue coordinates)
        mask: Optional[torch.Tensor] # [B, N] boolean   (True=valid, False=padded)
    ):
        """
        Args:
            seq_esm: ESM embeddings, shape [B, N, seq_dim_in].
            feats:   Node features for geometry, shape [B, N, egnn_dim].
            coors:   Coordinates [B, N, 3].
            mask:    Boolean mask [B, N] for valid residues (optional).

        Returns:
            A dictionary with:
              "seq_emb":   [B, N, d_projection]  projected seq embeddings
              "geom_emb":  [B, N, d_projection]  projected geometry embeddings
              "coords_out": [B, N, 3]            updated coordinates from EGNN
        """
        B, N, _ = seq_esm.shape
        B2, N2, fdim = feats.shape
        B3, N3, _ = coors.shape

        # Basic sanity checks
        assert B == B2 == B3, "Batch size mismatch among seq_esm, feats, coors."
        assert N == N2 == N3, "Residue count mismatch among seq_esm, feats, coors."
        if mask is not None:
            assert mask.shape[0] == B and mask.shape[1] == N, "Mask shape must match [B, N]."

        # 1) Project ESM embeddings => d_projection
        seq_emb = self.seq_project(seq_esm)  # [B, N, d_projection]
        feats_for_egnn = self.feats_linear_proj(feats) 

        # 2) EGNN forward to get updated geometry feats and coords
        #    feats_out has shape [B, N, egnn_dim], coors_out is [B, N, 3]
        coors_masked = coors.clone()
        coors_masked[~mask] = 1e9
        feats_out, coors_out = self.egnn(feats_for_egnn, coors, mask=mask)

        # 3) Project geometry feats => d_projection
        geom_emb = self.geom_project(feats_out)  # [B, N, d_projection]

        return {
            "seq_emb": seq_emb,
            "geom_emb": geom_emb,
            "feats_out": feats_out,
            "coords_out": coors_out
        }

# 3) Student Distillation
class ProClipStudent(nn.Module):
    def __init__(self, d_projection=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.GELU(),
            nn.Linear(1024, d_projection)
        )
        
    def forward(self, esm_emb, *args):
        return {'geom_emb': self.projector(esm_emb)}