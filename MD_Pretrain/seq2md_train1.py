# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import argparse
import logging
import yaml
from torch.cuda.amp import GradScaler, autocast
from seq2md import ProClipMDModel, ESMChainAEmbedder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# 1) Dataset
###############################################################################
class ProteinHDF5Dataset(torch.utils.data.Dataset):
    """
    Each HDF5 group should have:
      - 'representative_frames' => shape [10, N, 3]
      - attribute 'seq' => the protein sequence string
    """
    def __init__(self, h5_path):
        super().__init__()
        self.samples = []
        with h5py.File(h5_path, 'r') as f:
            for grp_name, grp_data in f.items():
                if isinstance(grp_data, h5py.Group):
                    if 'representative_frames' in grp_data:
                        coords = grp_data['representative_frames'][:]
                        seq = grp_data.attrs.get('seq', None)
                        if seq:
                            self.samples.append({
                                'representative_frames': coords,  # shape [10, N, 3]
                                'seq': seq
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_item = self.samples[idx]
        coords = data_item['representative_frames']
        seq    = data_item['seq']
        return seq, coords


###############################################################################
# 2) Collate Fn with Padding + Mask
###############################################################################
def protein_collate_fn(batch):
    """
    batch: List of (seq_str, coords_10_n_3)
      coords_10_n_3 => shape [10, N_i, 3] with N_i possibly different
    We do:
      1) Find max N in this batch => max_ca
      2) Zero-pad each coords array to [10, max_ca, 3]
      3) Build a Boolean mask => [10, max_ca] (True => valid)
      4) Stack them to get [B, 10, max_ca, 3] and [B, 10, max_ca]
    """
    seq_list = []
    coords_list = []
    mask_list = []

    # 1) Get a list of all N_i
    lengths = []
    for (seq, coords) in batch:
        # coords shape: [10, N_i, 3]
        _, n_i, _ = coords.shape
        lengths.append(n_i)

    max_n = max(lengths) if lengths else 0
    min_ca = 128
    max_ca = max(min_ca, min(max_n, 512))

    # Decide final max_ca
    # if max_n <= 512:
    #     max_ca = max_n
    # else:
    #     max_ca = 512

    # 2) For each sample: (a) Truncate if needed, (b) Pad, (c) Create mask
    for (seq, coords_10_n_3) in batch:
        seq_list.append(seq)
        f, n_i, _ = coords_10_n_3.shape  # f=10

        # (a) If n_i is bigger than max_ca, truncate
        if n_i > max_ca:
            coords_10_n_3 = coords_10_n_3[:, :max_ca, :]
            n_i = max_ca

        # (b) Zero-pad to shape [10, max_ca, 3]
        padded = np.zeros((f, max_ca, 3), dtype=np.float32)
        padded[:, :n_i, :] = coords_10_n_3

        # (c) Create a boolean mask [10, max_ca]
        mask_2d = np.zeros((f, max_ca), dtype=np.bool_)
        mask_2d[:, :n_i] = True

        coords_list.append(torch.tensor(padded, dtype=torch.float32))
        mask_list.append(torch.tensor(mask_2d, dtype=torch.bool))

    coords_batch = torch.stack(coords_list, dim=0)  # [B, 10, max_ca, 3]
    mask_batch   = torch.stack(mask_list,  dim=0)   # [B, 10, max_ca]

    return seq_list, coords_batch, mask_batch


###############################################################################
# 3) Utility Losses
###############################################################################
def contrastive_loss_fn(seq_global, geom_global, temperature=0.1):
    bsz = seq_global.size(0)
    seq_n = F.normalize(seq_global, dim=-1)
    geo_n = F.normalize(geom_global, dim=-1)
    logits = torch.matmul(seq_n, geo_n.t()) / temperature  # [B, B]
    labels = torch.arange(bsz, dtype=torch.long, device=seq_global.device)
    loss_seq2geo = F.cross_entropy(logits, labels)
    loss_geo2seq = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_seq2geo + loss_geo2seq)

def diversity_loss_fn(geom_frames):
    """
    geom_frames => [B, frames, d]
    We penalize frame-to-frame similarity to encourage diversity.
    """
    B, F, D = geom_frames.shape
    
    # Normalize all frames at once for efficiency
    frames_norm = torch.nn.functional.normalize(geom_frames, dim=-1)
    
    # Compute pairwise similarities within each batch
    total_div = 0
    for i in range(B):
        # Get normalized frames for this batch
        batch_frames = frames_norm[i]  # [F, D]
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(batch_frames, batch_frames.t())
        
        # Remove diagonal elements
        mask = ~torch.eye(F, device=sim_matrix.device, dtype=torch.bool)
        total_div += (sim_matrix * mask).sum()
        
    return total_div / B

def reconstruction_loss_fn(coords_pred, coords_true, mask):
    """
    coords_pred => [B, 10, max_ca, 3]
    coords_true => [B, 10, max_ca, 3]
    mask        => [B, 10, max_ca], True => valid
    We only compute MSE over valid positions.
    """
    mask_3d = mask.unsqueeze(-1)  # [B, 10, max_ca, 1]
    diff_sq = (coords_pred - coords_true)**2 * mask_3d
    valid_count = mask_3d.sum().clamp_min(1.0)
    return diff_sq.sum() / valid_count


###############################################################################
# 4) Training Script
###############################################################################
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def validate(model, val_loader, esm_embedder, device):
    """
    Validate the model and compute validation loss.
    """
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for seq_list, coords_batch, mask_batch in val_loader:
            coords_batch = coords_batch.to(device)
            mask_batch = mask_batch.to(device)

            # ESM embedding
            seq_emb_list = []
            max_len = 0
            for seq in seq_list:
                emb = esm_embedder.embed_chain(seq)  # [1, L, 1280]
                seq_emb_list.append(emb)
                if emb.size(1) > max_len:
                    max_len = emb.size(1)

            # pad ESM embeddings => [B, max_len, 1280]
            padded_seq_list = []
            for emb in seq_emb_list:
                L_ = emb.size(1)
                if L_ < max_len:
                    pad_ = torch.zeros((1, max_len - L_, 1280), device=device)
                    emb_ = torch.cat([emb, pad_], dim=1)
                else:
                    emb_ = emb
                padded_seq_list.append(emb_)
            seq_emb = torch.cat(padded_seq_list, dim=0).to(device)

            # Forward pass
            out = model(
                seq_esm=seq_emb,
                coords_ca=coords_batch,
                mask_2d=mask_batch  # crucial
            )
            coords_pred = out["coords_recon"]  # [B, 10, max_ca, 3]

            # Reconstruction loss
            loss = reconstruction_loss_fn(coords_pred, coords_batch, mask_batch)
            val_loss += loss.item()
            val_batches += 1

    return val_loss / max(1, val_batches)

def main():
    parser = argparse.ArgumentParser(description="CLIP-like training with variable N_ca.")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config.")
    parser.add_argument('--eval', action='store_true', help="Run evaluation only.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg  = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    paths_cfg = config['paths']
    gpu_cfg   = config.get('gpu', {'use_multi_gpu': False})

    setup_logging(paths_cfg['log_dir'])
    logger = logging.getLogger()
    logger.info("Starting training with variable Ca count")

    # 1) Dataset
    dataset = ProteinHDF5Dataset(data_cfg['path'])
    logger.info(f"Total samples: {len(dataset)}")
    total_size = len(dataset)
    train_sz = int(0.9 * total_size)
    val_sz   = total_size - train_sz
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz])
    logger.info(f"Train={train_sz}, Val={val_sz}")

    # 2) Dataloaders (using our custom collate_fn)
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        collate_fn=protein_collate_fn,
        num_workers=32,      
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        collate_fn=protein_collate_fn,
        num_workers=24,
        pin_memory=True
    )

    # 3) Model
    model = ProClipMDModel(config).to(device)
    if gpu_cfg.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    # 4) Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get('lr', 1e-3),
        weight_decay=train_cfg.get('weight_decay', 1e-5)
    )
    logger.info("Model & optimizer ready.")

    # 5) ESM embedder
    esm_embedder = ESMChainAEmbedder(device=device)
    if args.eval:  # Add this block
        logger.info("Inference mode...")
        model.eval()
        with torch.no_grad():
            for seq_list, coords_batch, mask_batch in val_loader:
                coords_batch = coords_batch.to(device)
                mask_batch = mask_batch.to(device)

                # Batch embedding
                batch_data = [("seq", seq) for seq in seq_list]
                _, _, tokens = esm_embedder.batch_converter(batch_data)
                tokens = tokens.to(device)

                # Forward pass
                seq_emb = esm_embedder.model(tokens, repr_layers=[33])["representations"][33][:, 1:-1, :]
                geom_global = model.inference_from_seq(seq_emb)
                logger.info(f"Inference result: {geom_global.shape}")

        logger.info("Inference complete.")
        return

    # 6) Training
    scaler = GradScaler(enabled=True)
    n_epochs = train_cfg.get('num_epochs', 10)
    c_w = train_cfg.get('contrastive_weight', 100)
    d_w = train_cfg.get('diversity_weight', 0.5)
    r_w = train_cfg.get('recon_weight', 10.0)
    best_val_loss = float('inf') 

    for epoch in range(1, n_epochs+1):
        model.train()
        total_loss = 0.0
        total_contrast = 0.0
        total_div = 0.0
        total_recon = 0.0
        num_batches = 0

        for b_idx, (seq_list, coords_batch, mask_batch) in enumerate(train_loader, 1):
            # seq_list => list of B strings
            # coords_batch => [B, 10, max_ca, 3]
            # mask_batch   => [B, 10, max_ca] bool
            B = len(seq_list)
            coords_batch = coords_batch.to(device)
            mask_batch   = mask_batch.to(device)

            # (A) ESM embedding
            seq_emb_list = []
            max_len = 0
            for seq in seq_list:
                emb = esm_embedder.embed_chain(seq)  # [1, L, 1280]
                seq_emb_list.append(emb)
                if emb.size(1) > max_len:
                    max_len = emb.size(1)

            # pad ESM embeddings => [B, max_len, 1280]
            padded_seq_list = []
            for emb in seq_emb_list:
                L_ = emb.size(1)
                if L_ < max_len:
                    pad_ = torch.zeros((1, max_len-L_, 1280), device=device)
                    emb_ = torch.cat([emb, pad_], dim=1)
                else:
                    emb_ = emb
                padded_seq_list.append(emb_)
            seq_emb = torch.cat(padded_seq_list, dim=0).to(device)

            optimizer.zero_grad()
            with autocast(enabled=True):
                # (B) Forward pass
                out = model(
                    seq_esm=seq_emb,
                    coords_ca=coords_batch,
                    mask_2d=mask_batch  # crucial
                )
                seq_global  = out["seq_global"]    # [B, d]
                geom_global = out["geom_global"]   # [B, d]
                geom_frames = out["geom_frames"]   # [B, 10, d]
                coords_pred = out["coords_recon"]  # [B, 10, max_ca, 3]

                # (C) Losses
                loss_con = contrastive_loss_fn(seq_global, geom_global, temperature=0.1)
                loss_div = diversity_loss_fn(geom_frames)
                loss_rec = reconstruction_loss_fn(coords_pred, coords_batch, mask_batch)
                total = c_w*loss_con + d_w*loss_div + r_w*loss_rec

            scaler.scale(total).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += total.item()
            total_contrast += loss_con.item()
            total_div += loss_div.item()
            total_recon += loss_rec.item()
            num_batches += 1

            if b_idx % 2 == 0:
                logger.info(
                    f"Epoch={epoch} Batch={b_idx} | "
                    f"loss={total:.4f}, con={loss_con:.4f}, "
                    f"div={loss_div:.4f}, rec={loss_rec:.4f}"
                )

        avg_train_loss = total_loss / max(1, num_batches)
        logger.info(f"[Epoch {epoch}] Training loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader, esm_embedder, device)
        logger.info(f"[Epoch {epoch}] Validation loss: {val_loss:.4f}")

        # Save current epoch model
        ckpt_path = os.path.join(paths_cfg['checkpoint_dir'], f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Checkpoint saved at {ckpt_path}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(paths_cfg['checkpoint_dir'], 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model updated at epoch {epoch} with validation loss {best_val_loss:.4f}")

    logger.info("Training complete.")

if __name__ == '__main__':
    main()
