# train.py

import os
import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F
import h5py
import random
import math
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import argparse
import logging
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional, Tuple, List
from seq2fn import ProClipMD, ProClipStudent
from seq2fn import ESMChainAEmbedder, ProteinNodeInitializer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("seq2fn.yaml", "r") as f:
    config = yaml.safe_load(f)

def set_seed(seed: int):
    """
    Set the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#For distillation
def contrastive_token_loss(
    student_emb: torch.Tensor,  
    teacher_emb: torch.Tensor,  
    mask: torch.Tensor,         
    temperature: torch.tensor(config['training']['temperature'], device=device)  
):

    valid = mask.flatten()
    student_flat = student_emb.view(-1,512)[valid]
    teacher_flat = teacher_emb.view(-1,512)[valid]
    return F.mse_loss(student_flat, teacher_flat), None

#Pre_TRAIN
# def contrastive_token_loss(
#     seq_emb: torch.Tensor,   # [B, N, d]
#     geom_emb: torch.Tensor,  # [B, N, d]
#     seq_mask: torch.Tensor,  # [B, N]  True => valid
#     geom_mask: torch.Tensor, # [B, N]  True => valid
#     temperature:torch.tensor(config['training']['temperature'], device=device)       # float scalar, e.g. 1/0.07
# ):
#     """
#     Residue-level CLIP-like contrastive loss:
#       1. Flatten out only positions where both seq_mask & geom_mask are True.
#       2. Normalize embeddings.
#       3. Compute pairwise dot products (shape [M, M]).
#       4. Symmetric cross-entropy loss with row-wise & column-wise classification.

#     Returns:
#       loss (Tensor): single scalar
#       logits (Tensor or None): optional, shape [M, M] of pairwise similarities
#     """
#     device = seq_emb.device
#     B, N, d = seq_emb.shape

#     # 1) Flatten only positions that are valid in *both* seq_mask & geom_mask
#     combined_mask = seq_mask & geom_mask     # [B, N]
#     flatten_valid_idx = combined_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)

#     if flatten_valid_idx.numel() == 0:
#         # If no valid tokens, return zero loss
#         return seq_emb.new_tensor(0.0, requires_grad=True), None

#     # Gather valid embeddings
#     seq_flat  = seq_emb.view(B*N, d)[flatten_valid_idx]   # [M, d]
#     geom_flat = geom_emb.view(B*N, d)[flatten_valid_idx]  # [M, d]
#     M = seq_flat.size(0)

#     # 2) L2 normalize
#     seq_norm  = F.normalize(seq_flat, dim=-1)   # [M, d]
#     geom_norm = F.normalize(geom_flat, dim=-1)  # [M, d]

#     # 3) Dot products * temperature => logits
#     logits = torch.matmul(seq_norm, geom_norm.t())          # [M, M]
#     logits = logits * torch.exp(temperature)

#     # 4) Symmetric cross-entropy
#     labels = torch.arange(M, device=device)
#     loss_i = F.cross_entropy(logits, labels)          # row-wise
#     loss_t = F.cross_entropy(logits.t(), labels)      # column-wise
#     loss = 0.5 * (loss_i + loss_t)

#     return loss, logits

###############################################################################
# Dataset & Collate
###############################################################################
class ProteinHDF5Dataset(Dataset):
    def __init__(self, h5_path: str):
        super().__init__()
        self.h5_path = h5_path
        self.samples = []
        
        with h5py.File(h5_path, 'r') as f:
            for grp_name in f.keys():
                grp = f[grp_name]
                if 'representative_frames' in grp:
                    coords = grp['representative_frames'][:]
                    seq = grp.attrs.get('sequence', '')
                    if len(seq) == coords.shape[0]:
                        self.samples.append((seq, coords))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        seq, coords = self.samples[idx]
        return seq, coords.astype(np.float32)

def protein_collate_fn(batch):
    seq_list, coords_list = [], []
    lengths = [min(len(seq), 1024) for seq, _ in batch]  
    
    max_len = min(1024, max(lengths))
    padded_coords = np.zeros((len(batch), max_len, 3), dtype=np.float32)
    masks = np.zeros((len(batch), max_len), dtype=np.bool_)
    
    for i, (seq, coords) in enumerate(batch):
        seq_len = min(len(seq), max_len)
        coords_trunc = coords[:max_len]
        padded_coords[i, :seq_len] = coords_trunc
        masks[i, :seq_len] = True
        
    return (
        [seq for seq, _ in batch], 
        torch.from_numpy(padded_coords), 
        torch.from_numpy(masks)
    )

###############################################################################
# Training Loop
###############################################################################
def train_epoch(
    model: nn.Module,
    esm_embedder: ESMChainAEmbedder,
    node_init: ProteinNodeInitializer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler(enabled=config['training']['mixed_precision']),
    device: torch.device,
    model_config: dict,
    epoch: int = 200,
    max_seq_len: int = 1024,
    use_amp: bool = True
) -> float:


    total_loss = 0.0
    batch_count = 0
    # Load teacher model
    teacher = ProClipMD(
        seq_dim_in=2560,
        feats_in_dim=26,
        egnn_dim=128,
        d_projection=512,
        egnn_depth=4,
        num_nearest_neighbors=64,
        coors_clamp=2.0
    ).to(device)
    teacher.load_state_dict(torch.load('teacher.pt')['model_state'])
    teacher.eval()
    
    for batch_idx, (seq_list, _, mask) in enumerate(loader):  

        mask = mask.to(device)
        B, N = mask.shape
        
       
        dummy_coords = torch.zeros(B, N, 3, device=device)
        dummy_feats = torch.zeros(B, N, 26, device=device)
        
        
        with torch.no_grad():
            truncated_seqs = [seq[:max_seq_len] for seq in seq_list]
            esm_embs = esm_embedder.embed_batch(truncated_seqs, max_ca=1024)
            
            B_esm, L_esm, D_esm = esm_embs.shape
            if L_esm < N:
                pad = torch.zeros((B_esm, N-L_esm, D_esm), device=device)
                esm_batch = torch.cat([esm_embs, pad], dim=1)
            else:
                esm_batch = esm_embs[:, :N, :]

       
        with torch.no_grad():
            teacher_out = teacher(esm_batch, dummy_feats, dummy_coords, mask)
        
        
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            student_out = model(esm_batch)
            loss, _ = contrastive_token_loss(
                student_out['geom_emb'],
                teacher_out['geom_emb'],
                mask,
                temperature=torch.tensor(config['training']['temperature'], device=device)
            )

       
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        
        total_loss += loss.item() * B
        batch_count += B

    return total_loss / batch_count       

    # model.train()
    # total_loss = 0.0
    # batch_count = 0
    
    # for batch_idx, (seq_list, coords, mask) in enumerate(loader):
    #     coords = coords.to(device)
    #     mask = mask.to(device)
    #     B, N = coords.shape[:2]
        
    #     # 1. Generate ESM embeddings
    #     # esm_embs = []
    #     with torch.no_grad():
    #         truncated_seqs = [seq[:max_seq_len] for seq in seq_list]
    #         esm_embs = esm_embedder.embed_batch(truncated_seqs, max_ca=1024)  # [B, L, 2560]
    #         # Pad to match the max length in the batch (N)
    #         B_esm, L_esm, D_esm = esm_embs.shape
    #         if L_esm < N:
    #             pad = torch.zeros((B_esm, N - L_esm, D_esm), device=device)
    #             esm_batch = torch.cat([esm_embs, pad], dim=1)
    #         else:
    #             esm_batch = esm_embs[:, :N, :]
        
    #     # 2. Generate node features
    #     feats = node_init.encode_sequences(seq_list, max_ca=N)
    #     feats = feats.to(device)
        
    #     # 3. Forward pass
    #     optimizer.zero_grad()
    #     with autocast(enabled=use_amp):
    #         outputs = model(esm_batch, feats, coords, mask=mask)
    #         loss, _ = contrastive_token_loss(
    #             outputs['seq_emb'], 
    #             outputs['geom_emb'],
    #             mask, mask,
    #             temperature=torch.tensor(config['training']['temperature'], device=device)
    #         )
        
    #     # 4. Backward pass
    #     scaler.scale(loss).backward()
    #     scaler.step(optimizer)
    #     scaler.update()

    #     batch_loss = loss.item()
    #     total_loss += batch_loss * B
    #     batch_count += B
        
    #     print_interval = 10  # 10
    #     if (batch_idx + 1) % 10 == 0:
    #         avg_batch_loss = batch_loss
    #         print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(loader)}] Loss: {avg_batch_loss:.4f}")
    
    # epoch_avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    # return epoch_avg_loss

@torch.no_grad()
def validate(
    model: nn.Module,
    esm_embedder: ESMChainAEmbedder,
    node_init: ProteinNodeInitializer,
    loader: DataLoader,
    device: torch.device,
    max_seq_len: int = 1024
) -> float:
    model.eval()
    total_loss = 0.0
    
    for seq_list, coords, mask in loader:
        coords = coords.to(device)
        mask = mask.to(device)
        B, N = coords.shape[:2]
        
        # 1. ESM embeddings
        with torch.no_grad():
            truncated_seqs = [seq[:max_seq_len] for seq in seq_list]
            esm_embs = esm_embedder.embed_batch(truncated_seqs, max_ca=1024)  #  Batch processing
            # Pad to match the max length in the batch (N)
            B_esm, L_esm, D_esm = esm_embs.shape
            if L_esm < N:
                pad = torch.zeros((B_esm, N - L_esm, D_esm), device=device)
                esm_batch = torch.cat([esm_embs, pad], dim=1)
            else:
                esm_batch = esm_embs[:, :N, :]
        
        # 2. Node features
        feats = node_init.encode_sequences(seq_list, max_ca=N)
        feats = feats.to(device)
        
        # 3. Forward
        outputs = model(esm_batch, feats, coords, mask=mask)
        loss, _ = contrastive_token_loss(
            outputs['seq_emb'], outputs['geom_emb'],
            mask, mask, temperature=torch.tensor(config['training']['temperature'], device=device)
        )
        total_loss += loss.item() * B
    
    return total_loss / len(loader.dataset)

###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Train ProClipMD")
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    parser.add_argument('--resume', help="Checkpoint to resume")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (num_gpus := torch.cuda.device_count()) > 1 and config['gpu']['use_multi_gpu']:
        print(f"Using {num_gpus} GPUs!")
    
    # Initialize components
    node_init = ProteinNodeInitializer()
    esm_embedder = ESMChainAEmbedder(device=device)
    model_config = {
        'seq_dim_in': 2560,  
        'feats_in_dim': 26,  
        'egnn_dim': config['model']['geom_dim'],  
        'd_projection': config['model']['d_projection'],  
        'egnn_depth': config['model']['egnn_layers'],  
        'num_nearest_neighbors': 64,  
        'coors_clamp': 2.0  
    }

    #distillation
    model = ProClipStudent(d_projection=config['model']['d_projection']).to(device)
    # model = ProClipMD(**model_config).to(device), for training ProClipMD
    
    if num_gpus > 1 and config['gpu']['use_multi_gpu']:
        model = nn.DataParallel(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'])
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Data
    dataset = ProteinHDF5Dataset(config['data']['path'])
    train_size = int(0.9 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        collate_fn=protein_collate_fn,
        num_workers=16,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config['data']['batch_size'],
        collate_fn=protein_collate_fn,
        num_workers=12,
        pin_memory=True
    )
    
    # Logging
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_dir/'train.log'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    
    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs']):
        train_loss = train_epoch(
            model, esm_embedder, node_init,
            train_loader, optimizer, scaler, device,
            epoch,
            use_amp=config['training']['mixed_precision']
        )
        val_loss = validate(model, esm_embedder, node_init, val_loader, device)
        
        logger.info(
            f"Epoch {epoch+1}/{config['training']['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = log_dir/ "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, ckpt_path)
            logger.info(f"Saved best model to {ckpt_path}")
        
        scheduler.step()

if __name__ == "__main__":
    main()


 