import os
import h5py
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from torch.cuda.amp import autocast, GradScaler
from esm650_distill import ESMChainAEmbedder, ChainBAutoregressiveModel, CombinedDesign

# Example amino-acid set
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_TOKEN_IDX = len(AMINO_ACIDS)  # e.g. 20 => unknown
PAD_TOKEN_IDX = UNK_TOKEN_IDX + 1 # e.g. 21 => pad

class DoubleChainHDF5Dataset(Dataset):
    def __init__(self, h5_path: str, split: str = "train"):
        super().__init__()
        self.samples = []
        self.h5_path = h5_path
        self.split = split

        with h5py.File(h5_path, 'r') as f:
            if split not in f:
                raise ValueError(f"Group '{split}' not found in the HDF5 file.")
            split_group = f[split]

            for item_name, item_group in split_group.items():
                if isinstance(item_group, h5py.Group):
                    chainA_seq = item_group.attrs.get('sequence_chainA', None)
                    chainB_seq = item_group.attrs.get('sequence_chainB', None)
                    # Possibly coords if needed
                    if chainA_seq and chainB_seq:
                        self.samples.append({
                            'chainA_seq': chainA_seq.strip(),
                            'chainB_seq': chainB_seq.strip()
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['chainA_seq'], s['chainB_seq']

def chain_to_tokens(chain_seq: str) -> torch.Tensor:
    """
    Convert chain B to tokens, ignoring unknown => UNK_TOKEN_IDX
    """
    tokens = [AA_TO_IDX.get(ch, UNK_TOKEN_IDX) for ch in chain_seq]
    return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    chainA_list = []
    chainB_token_list = []
    max_len_b = 0

    for (A_str, B_str) in batch:
        chainA_list.append(A_str)
        b_tokens = chain_to_tokens(B_str)
        if b_tokens.size(0) > max_len_b:
            max_len_b = b_tokens.size(0)
        chainB_token_list.append(b_tokens)

    # Pad chainB
    chainB_batch = []
    for tokens in chainB_token_list:
        pad_len = max_len_b - tokens.size(0)
        if pad_len > 0:
            pad_tokens = torch.full((pad_len,), PAD_TOKEN_IDX, dtype=torch.long)
            tokens = torch.cat([tokens, pad_tokens], dim=0)
        chainB_batch.append(tokens)

    chainB_batch = torch.stack(chainB_batch, dim=0)  # [B, max_len_b]
    return chainA_list, chainB_batch

def setup_dirs(paths_cfg):
    """Create all necessary directories"""
    for dir_path in [
        paths_cfg['log_dir'],
        paths_cfg['checkpoint_dir'],
        paths_cfg['test_output_dir']
    ]:
        os.makedirs(dir_path, exist_ok=True)

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def validate_data_pipeline(model, test_loader, device):
    model.eval()
    try:
        batch = next(iter(test_loader))
        with torch.no_grad():
            logits = model(batch[0], batch[1].to(device))
            print(f"YES: {logits.shape}")
    except Exception as e:
        print(f"NO: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Train ESM + Geometry Distillation")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    paths_cfg = config['paths']
    gpu_cfg = config.get('gpu', {'use_multi_gpu': False})
    setup_dirs(paths_cfg)
    setup_logging(paths_cfg['log_dir'])
    logger = logging.getLogger()
    logger.info("Starting training with ESM-based chain design")

    file_path = data_cfg['path']
    train_ds = DoubleChainHDF5Dataset(file_path, split='train')
    val_ds = DoubleChainHDF5Dataset(file_path, split='val')
    test_ds = DoubleChainHDF5Dataset(file_path, split='test')
    logger.info(f"Train DS: {len(train_ds)}, Val DS: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=data_cfg.get('num_workers', 32),  
        pin_memory=data_cfg.get('pin_memory', True)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_cfg.get('num_workers', 24),  
        pin_memory=data_cfg.get('pin_memory', True)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.get('test_batch_size', data_cfg['batch_size']),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_cfg.get('num_workers', 10),
        pin_memory=data_cfg.get('pin_memory', True)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chainA_embedder = ESMChainAEmbedder(device=device)

    # Build model with penalty & random length logic
    model = CombinedDesign(
        esm_embed_dim=model_cfg.get('esm_embed_dim', 2560),
        md_embed_dim=model_cfg.get('md_embed_dim', 512),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
        # alphabet=chainA_embedder.alphabet,
        vocab_size=len(AMINO_ACIDS)+2,  # UNKPAD
        checkpoint_path=model_cfg['distill_checkpoint'],  # 
        device=device
    ).to(device)

    if gpu_cfg.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get('lr', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 1e-5)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)
    scaler = GradScaler()

    num_epochs = train_cfg.get('num_epochs', 1)
    test_interval = train_cfg.get('test_interval', 1)
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_ckpt_path = os.path.join(paths_cfg['checkpoint_dir'], "esmd_model.pt")

    logger.info("Running data pipeline validation...")
    try:
        test_batch = next(iter(train_loader))
        with torch.no_grad():
            logits = model(test_batch[0], test_batch[1].to(device))
            logger.info(f"Validation success! Logits shape: {logits.shape}")
    except Exception as e:
        logger.error(f"Data pipeline validation failed: {str(e)}")
        raise

    for epoch in range(1, num_epochs+1):
        logger.info(f"\n=== Epoch {epoch}/{num_epochs} ===")
        model.train()
        total_loss = 0.0
        total_samples = 0
        batch_counter = 0

        for batch_idx, (chainA_list, chainB_tokens) in enumerate(train_loader, 1):
            B = len(chainA_list)
            chainB_tokens = chainB_tokens.to(device)

            optimizer.zero_grad()
            with autocast():
                # we do forward => cross-entropy => add penalty
                # chainA_list is needed for the penalty
                logits = model(
                    chainA_list,
                    chainB_tokens
                )

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    chainB_tokens.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * B
            total_samples += B

            if batch_idx % 100 == 0:
                avg_loss = total_loss / total_samples
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                            f"Avg Loss: {avg_loss:.4f} | "
                            f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        avg_train_loss = total_loss / total_samples

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for chainA_list, chainB_tokens in val_loader:
                Bv = len(chainA_list)
                chainB_tokens = chainB_tokens.to(device)
                
                logits = model(chainA_list, chainB_tokens)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    chainB_tokens.view(-1)
                )
                
                val_loss_sum += loss.item() * Bv
                val_samples += Bv

        avg_val_loss = val_loss_sum / val_samples
        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")

        # No special checkpoint logic here, but you can add it
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,  #
                'config': config,
                'vocab_info': {
                    'AMINO_ACIDS': AMINO_ACIDS,
                    'IDX_TO_AA': {v: k for k, v in AA_TO_IDX.items()},
                    'PAD_TOKEN_IDX': PAD_TOKEN_IDX,
                    'UNK_TOKEN_IDX': UNK_TOKEN_IDX
                }
            }, best_ckpt_path)
            logger.info(f"Saved new best model with val loss: {avg_val_loss:.4f}")
        

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_ckpt_path}")

if __name__ == '__main__':
    main()
