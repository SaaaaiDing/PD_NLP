import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import h5py
from EGNN_SE3 import EGNN as EGNNSE3  
from typing import List
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx, to_dense_batch
from torch_geometric.nn import BatchNorm, LayerNorm
from torch_scatter import scatter_add
from torch.nn import MultiheadAttention
from torch_cluster import knn_graph 

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, bn=False):
        super(MLP, self).__init__()
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

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        if self.bn:
            x = self.bn_layer(x)
        x = self.activation(x)
        x = self.lin3(x)
        return x

class EGNNPooling(torch.nn.Module):
    def __init__(self, hidden_dim=16, attn=False):
        super(EGNNPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.egnnse3 = EGNNSE3(
            in_node_nf=hidden_dim,
            hidden_nf=hidden_dim,
            out_node_nf=hidden_dim,
            in_edge_nf=hidden_dim,
            attention=attn,
            reflection_equiv=False
        )
        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
        self.bn_edge = LayerNorm(hidden_dim)
        self.bn_h = LayerNorm(hidden_dim)

    def forward(self, h, coords, batch=None, edge_index=None):
        """
        Forward pass for EGNNPooling.

        Args:
            h (Tensor): Node features, shape [N, F]
            coords (Tensor): Node coordinates, shape [N, 3]
            batch (Tensor): Batch indices, shape [N]
            edge_index (Tensor): Edge indices, shape [2, E]

        Returns:
            Tuple[Tensor, Tensor]: Updated node features and coordinates.
        """
        row, col = edge_index
        out = torch.cat([h[row], h[col]], dim=1)
        edge_attr = self.edge_mlp(out)
        edge_attr = self.bn_edge(edge_attr)

        h, coords = self.egnnse3(h, coords, edge_index, edge_attr, batch)

        h = self.bn_h(h)

        return h, coords

class Encoder(torch.nn.Module):
    def __init__(self, n_feat=1, hidden_dim=16, out_node_dim=32, layers=1,
                 pooling=True, attn=False, max_num_ca=500):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.layers = layers
        self.max_num_ca = max_num_ca

        if self.pooling:
            self.poolings = nn.ModuleList([
                EGNNPooling(hidden_dim=hidden_dim, attn=attn) for _ in range(self.layers)
            ])
        self.bn_pool = nn.ModuleList([
            LayerNorm(hidden_dim) for _ in range(self.layers)
        ])

    def forward(self, coords, h, masks, batch_indices):
        B = h.shape[0]
        T = h.shape[1]
        F = h.shape[-1]

        print(f"Encoder Forward - Batch size: {B}, Time steps: {T}, Features: {F}")
        print(f"Batch indices shape: {batch_indices.shape}")
        assert torch.all(batch_indices >= 0) and torch.all(batch_indices < B), "Batch indices out of range!"

        for i in range(self.layers):
            h_frame_pooled, coords_frame_pooled = [], []

            for t in range(T):
                h_t = h[:, t, :, :]  # Shape: (B, N, F)
                coords_t = coords[:, t, :, :]  # Shape: (B, N, 3)
                mask_t = masks[:, t, :]  # Shape: (B, N)

                valid_idx = mask_t.nonzero(as_tuple=False)
                if valid_idx.numel() == 0:
                    print(f"Skipping time step {t} as there are no valid nodes.")
                    continue

                indices = valid_idx[:, 0] * mask_t.shape[1] + valid_idx[:, 1]
                h_t_flat = h_t.reshape(-1, F)[indices]
                coords_t_flat = coords_t.reshape(-1, 3)[indices]
                batch_flat = batch_indices[indices]

                print(f"h_t_flat shape: {h_t_flat.shape}, coords_t_flat shape: {coords_t_flat.shape}, batch_flat shape: {batch_flat.shape}")

                if coords_t_flat.size(0) < 5:
                    print(f"Skipping time step {t} as there are not enough nodes for knn_graph.")
                    continue

                edge_index = knn_graph(coords_t_flat, k=5, batch=batch_flat, loop=False)
                print(f"Edge index shape: {edge_index.shape}")

                h_t_pooled, coords_t_pooled = self.poolings[i](h_t_flat, coords_t_flat, batch=batch_flat, edge_index=edge_index)

                h_t_pooled, _ = to_dense_batch(h_t_pooled, batch_flat)
                coords_t_pooled, _ = to_dense_batch(coords_t_pooled, batch_flat)

                h_frame_pooled.append(h_t_pooled)
                coords_frame_pooled.append(coords_t_pooled)

            if not h_frame_pooled:
                print(f"All time steps skipped for layer {i}.")
                continue

            h = torch.stack(h_frame_pooled, dim=1)
            coords = torch.stack(coords_frame_pooled, dim=1)

            current_num_nodes = h.shape[2]
            if current_num_nodes > self.max_num_ca:
                print(f"Number of nodes {current_num_nodes} exceeds max_num_ca {self.max_num_ca}. Truncating.")
                h = h[:, :, :self.max_num_ca, :]
                coords = coords[:, :, :self.max_num_ca, :]
            elif current_num_nodes < self.max_num_ca:
                pad_size = self.max_num_ca - current_num_nodes
                pad_h = torch.zeros((h.shape[0], h.shape[1], pad_size, self.hidden_dim), device=h.device)
                pad_coords = torch.zeros((coords.shape[0], coords.shape[1], pad_size, 3), device=coords.device)
                h = torch.cat([h, pad_h], dim=2)
                coords = torch.cat([coords, pad_coords], dim=2)
                print(f"Padded node count from {current_num_nodes} to {self.max_num_ca}.")

            h = self.bn_pool[i](h)

        return coords, h

class DecoderTranspose(torch.nn.Module):
    def __init__(self, hidden_dim=16, layers=1, attn=False, max_num_ca=500):
        super(DecoderTranspose, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.attn = attn
        self.max_num_ca = max_num_ca

        self.egnn_layers = nn.ModuleList([
            EGNNSE3(
                in_node_nf=hidden_dim,
                hidden_nf=hidden_dim,
                out_node_nf=hidden_dim,
                in_edge_nf=hidden_dim,
                attention=attn,
                reflection_equiv=False
            ) for _ in range(self.layers)
        ])
        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
        self.bn_edge = LayerNorm(hidden_dim)
        self.bn = torch.nn.ModuleList([
            LayerNorm(hidden_dim) for _ in range(self.layers)
        ])

        if self.attn:
            self.attention = MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, coords, h, batch_indices):
        B, T, _, F = h.shape

        print(f"DecoderTranspose Forward - Batch size: {B}, Time steps: {T}, Features: {F}")
        print(f"Batch indices max: {batch_indices.max()}, min: {batch_indices.min()}")
        assert torch.all(batch_indices >= 0) and torch.all(batch_indices < B), "Batch indices out of range!"

        for i in range(self.layers):
            h_frame_processed, coords_frame_processed = [], []

            for t in range(T):
                h_t = h[:, t, :, :]  # Shape: (B, N, F)
                coords_t = coords[:, t, :, :]  # Shape: (B, N, 3)

                for b in range(B):
                    mask_b = batch_indices == b
                    valid_idx = mask_b.nonzero(as_tuple=False).reshape(-1)

                    if valid_idx.numel() == 0:
                        continue

                    h_b_t = h_t[b, valid_idx, :]  # Shape: (N_b, F)
                    coords_b_t = coords_t[b, valid_idx, :]  # Shape: (N_b, 3)
                    batch_b_flat = torch.full((valid_idx.size(0),), b, dtype=torch.long).to(coords.device)

                    assert b < B, f"Batch index {b} out of range."

                    if coords_b_t.size(0) < 5:
                        print(f"Skipping batch {b}, time step {t}, due to insufficient valid nodes.")
                        continue

                    edge_index = knn_graph(coords_b_t, k=5, batch=batch_b_flat, loop=False)

                    row, col = edge_index
                    edge_attr = self.edge_mlp(torch.cat([h_b_t[row], h_b_t[col]], dim=1))
                    edge_attr = self.bn_edge(edge_attr)

                    h_b_t_processed, coords_b_t_processed = self.egnn_layers[i](
                        h_b_t, coords_b_t, edge_index, edge_attr, batch_b_flat
                    )

                    h_b_t_processed, _ = to_dense_batch(h_b_t_processed, batch_b_flat)
                    coords_b_t_processed, _ = to_dense_batch(coords_b_t_processed, batch_b_flat)

                    h_frame_processed.append(h_b_t_processed)
                    coords_frame_processed.append(coords_b_t_processed)

            if not h_frame_processed:
                print(f"All time steps skipped for layer {i}.")
                continue

            h = torch.stack(h_frame_processed, dim=1)
            coords = torch.stack(coords_frame_processed, dim=1)

            current_num_nodes = h.shape[2]
            if current_num_nodes > self.max_num_ca:
                print(f"Number of nodes {current_num_nodes} exceeds max_num_ca {self.max_num_ca}. Truncating.")
                h = h[:, :, :self.max_num_ca, :]
                coords = coords[:, :, :self.max_num_ca, :]
            elif current_num_nodes < self.max_num_ca:
                pad_size = self.max_num_ca - current_num_nodes
                pad_h = torch.zeros((h.shape[0], h.shape[1], pad_size, self.hidden_dim), device=h.device)
                pad_coords = torch.zeros((coords.shape[0], coords.shape[1], pad_size, 3), device=coords.device)
                h = torch.cat([h, pad_h], dim=2)
                coords = torch.cat([coords, pad_coords], dim=2)
                print(f"Padded node count from {current_num_nodes} to {self.max_num_ca}.")

            h = self.bn[i](h)

        return coords, h

class ProAutoMD(torch.nn.Module):
    def __init__(self, layers=1, mp_steps=4, num_types=21, type_dim=32, hidden_dim=16, out_node_dim=32,
                 output_pad_dim=1, output_res_dim=21, pooling=True, noise=False, attn=False, max_num_ca=500):
        super(ProAutoMD, self).__init__()

        self.pooling = pooling
        self.noise = noise

        self.encoder = Encoder(
            n_feat=type_dim,
            hidden_dim=hidden_dim,
            out_node_dim=hidden_dim,
            layers=layers,
            pooling=self.pooling,
            attn=attn,
            max_num_ca=max_num_ca
        )

        self.decoder = DecoderTranspose(
            hidden_dim=hidden_dim,
            layers=layers,
            attn=attn,
            max_num_ca=max_num_ca
        )

        self.residue_type_embedding = torch.nn.Embedding(num_types, hidden_dim)

        self.edge_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, F.relu)
        self.bn_edge = LayerNorm(hidden_dim)
        self.mlp_padding = MLP(hidden_dim, hidden_dim, output_pad_dim, F.relu)
        self.mlp_residue = MLP(hidden_dim, hidden_dim * 4, output_res_dim, F.relu)

        self.sigmoid = nn.Sigmoid()

        self.mlp_mu_h = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_sigma_h = nn.Linear(hidden_dim, hidden_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.kl_h = 0

    def add_noise(self, inputs, noise_factor=2):
        noisy = inputs + torch.randn_like(inputs) * noise_factor
        return noisy

    def forward(self, x, coords_ca, masks, batch_indices):
        batch_size = x.shape[0]
        max_T = coords_ca.shape[1]
        max_num_atoms = x.shape[1]
        F = self.residue_type_embedding.embedding_dim

        h = self.residue_type_embedding(x.squeeze(-1).long())

        h = h.unsqueeze(1).expand(-1, max_T, -1, -1)
        masks = masks.unsqueeze(1).expand(-1, max_T, -1)

        emb_coords_ca, emb_h = self.encoder(coords_ca, h, masks, batch_indices)

        mu_h = self.mlp_mu_h(emb_h)
        sigma_h = self.mlp_sigma_h(emb_h)
        z_h = mu_h + torch.exp(sigma_h / 2) * self.N.sample(mu_h.shape).to(mu_h.device)
        self.kl_h = -0.5 * (1 + sigma_h - mu_h ** 2 - torch.exp(sigma_h)).sum() / batch_size

        coords_ca_pred, h_decoder = self.decoder(emb_coords_ca, z_h, batch_indices)

        pad_pred = self.sigmoid(self.mlp_padding(h_decoder))
        aa_pred = self.mlp_residue(h_decoder)

        return coords_ca_pred, aa_pred, pad_pred, self.kl_h, z_h

