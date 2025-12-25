import os
import h5py
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from torch.cuda.amp import autocast, GradScaler

from esm3B import ESMChainAEmbedder, ChainBAutoregressiveModel, CombinedDesign

#
# 1) We will rely on the *ESM* alphabet for token indexing:
#    0: <cls>
#    1: <pad>
#    2: <eos>
#    3: <unk>
#    4: L
#    5: A
#    6: G
#    ... etc.
#
#    We'll call them by name so it's absolutely clear which is which.
#

class DoubleChainHDF5Dataset(Dataset):
    """
    A dataset that has (chainA_seq, chainB_seq) in an HDF5 under e.g.:
      f["train"][item_name].attrs["sequence_chainA"] = "..."
      f["train"][item_name].attrs["sequence_chainB"] = "..."
    """
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


def chainB_to_esm_tokens(seq_str: str, alphabet) -> torch.Tensor:
    """
    Convert chain B from a raw string to a 1D tensor of ESM token indices
    using alphabet.get_batch_converter(). For training, we remove the [CLS] at
    index=0 and [EOS] at index=-1 so that the network can learn to produce them.
    """
    # Put this into a single-sequence batch for ESM's batch_converter
    data = [("chainB", seq_str)]
    labels, strs, tokens = alphabet.get_batch_converter()(data)
    # tokens shape => [1, length+2], e.g. [CLS], ..., [EOS]
    tokens = tokens[0]  # => [L+2] shape
    # Drop the first ([CLS]=0) and last ([EOS]=2) to get just the "middle" tokens
    # that we want to feed as training data. 
    # For example, "ACD" => tokens might be [0, 5, 13, 3, 2]
    # (assuming 'C' is not recognized => <unk>=3, or something),
    # we remove the first and last => [5,13,3].
    # If you want the network to predict the <eos> at the end, you keep it
    # as a label. But commonly, we remove it from the input side.
    # We'll do the typical approach: removing them from the "input" tokens.
    # Then the model can SHIFT them for next-step prediction if you prefer.
    # 
    # For direct cross-entropy with teacher-forcing, you might keep the <eos>
    # at the end. But let's follow the typical "remove <cls>/<eos> from input" approach:
    # 
    # If you want the network to produce <eos>, you can keep the final token
    # as a *target* in the cross entropy. That means you do not remove the <eos>
    # from the *target* side. 
    # 
    # For simplicity here, let's remove them from chainB. 
    tokens = tokens[1:-1]  # remove <cls>, <eos>
    return tokens


def collate_fn(batch, alphabet):
    """
    We gather a batch of (chainA_seq, chainB_seq).
      1) We'll keep chainA_seq as a list of strings (for conditional embedding).
      2) We convert chainB_seq => ESM tokens [L], removing [CLS]/[EOS].
      3) We'll pad chainB_seq to the max length in the batch with <pad>=1.
    """
    chainA_list = []
    chainB_token_list = []
    max_len_b = 0

    for (A_str, B_str) in batch:
        chainA_list.append(A_str)
        b_tokens = chainB_to_esm_tokens(B_str, alphabet)  # 1D [L]
        length_b = b_tokens.size(0)
        if length_b > max_len_b:
            max_len_b = length_b
        chainB_token_list.append(b_tokens)

    # Now we pad with <pad>==1
    padded_batch = []
    for tokens in chainB_token_list:
        pad_len = max_len_b - tokens.size(0)
        if pad_len > 0:
            pad = torch.full((pad_len,), alphabet.padding_idx, dtype=torch.long)
            tokens = torch.cat([tokens, pad], dim=0)
        padded_batch.append(tokens)

    chainB_batch = torch.stack(padded_batch, dim=0)  # [B, max_len_b]
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

    # 1) Build the dataset
    h5_path = data_cfg['path']
    train_ds = DoubleChainHDF5Dataset(h5_path, split='train')
    val_ds   = DoubleChainHDF5Dataset(h5_path, split='val')
    test_ds  = DoubleChainHDF5Dataset(h5_path, split='test')
    logger.info(f"Train DS: {len(train_ds)}, Val DS: {len(val_ds)}, Test DS: {len(test_ds)}")

    # 2) Build your model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedDesign(
        esm_embed_dim=model_cfg.get('esm_embed_dim', 2560),
        md_embed_dim=model_cfg.get('md_embed_dim', 512),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
        checkpoint_path=model_cfg['distill_checkpoint'],
        device=device
    ).to(device)

    # Since CombinedDesign loads self.alphabet from ESM internally, let's store a reference
    alphabet = model.esm_embedder.alphabet   # This is the ESM2_t36_3B_UR50D() alphabet

    # 3) Build DataLoader with a custom collate that uses the same ESM alphabet
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, alphabet),  # pass ESM alphabet
        num_workers=data_cfg.get('num_workers', 32),
        pin_memory=data_cfg.get('pin_memory', True)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, alphabet),
        num_workers=data_cfg.get('num_workers', 32),
        pin_memory=data_cfg.get('pin_memory', True)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.get('test_batch_size', data_cfg['batch_size']),
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, alphabet),
        num_workers=data_cfg.get('num_workers', 24),
        pin_memory=data_cfg.get('pin_memory', True)
    )

    # 4) Multi-GPU if needed
    if gpu_cfg.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    # 5) Optimizer + Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get('lr', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 1e-5)
    )
    #
    # IMPORTANT: we ignore pad=<pad>=1 for the cross-entropy
    #
    criterion = nn.CrossEntropyLoss(ignore_index=alphabet.padding_idx)
    scaler = GradScaler()

    num_epochs     = train_cfg.get('num_epochs', 1)
    test_interval  = train_cfg.get('test_interval', 1)
    best_val_loss  = float('inf')
    best_ckpt_path = os.path.join(paths_cfg['checkpoint_dir'], "esmd_model.pt")

    # Quick pipeline test
    logger.info("Running data pipeline validation (one batch test)...")
    try:
        chainA_list, chainB_tokens = next(iter(train_loader))
        chainB_tokens = chainB_tokens.to(device)
        with torch.no_grad():
            logits = model(chainA_list, chainB_tokens)
            logger.info(f"Check pass! Logits shape: {logits.shape}")
    except Exception as e:
        logger.error(f"Data pipeline validation failed: {str(e)}")
        raise

    for epoch in range(1, num_epochs+1):
        logger.info(f"\n=== Epoch {epoch}/{num_epochs} ===")
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (chainA_list, chainB_tokens) in enumerate(train_loader, 1):
            B = len(chainA_list)
            chainB_tokens = chainB_tokens.to(device)

            optimizer.zero_grad()
            with autocast():
                # 6) Forward pass: we input chainA_list (strings) and chainB_tokens
                #    The model returns [B, Lb, vocab_size].
                logits = model(chainA_list, chainB_tokens)

                # 7) Cross-entropy over chainB_tokens, ignoring <pad>=1
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
                            f"LR: {optimizer.param_groups[0]['lr']:.2e}")

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
                val_samples  += Bv

        avg_val_loss = val_loss_sum / val_samples
        logger.info(f"Epoch {epoch} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                    f"| Best Val Loss: {best_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'config': config,
            }, best_ckpt_path)
            logger.info(f"Saved new best model with val loss: {avg_val_loss:.4f}")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {best_ckpt_path}")

if __name__ == '__main__':
    main()
