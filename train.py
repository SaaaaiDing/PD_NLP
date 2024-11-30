import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data
from PD_LATENT import ProAutoMD  
from typing import List
import argparse
import logging
import yaml
import h5py

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(file_path):
    """
    Load protein data from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        List[Data]: List of PyTorch Geometric Data objects.
    """
    data_list = []
    with h5py.File(file_path, 'r') as f:
        for name, obj in f.items():
            if isinstance(obj, h5py.Group):
                if 'ca_coordinates' in obj:
                    coords = obj['ca_coordinates'][:]
                    sequence = obj.attrs.get('sequence', None)
                    if sequence:
                        # Encode residues: A=0, B=1, ..., Z=25
                        sequence_tensor = torch.tensor([ord(residue) - ord('A') for residue in sequence], dtype=torch.long)

                        # Validate residue indices
                        if torch.any(sequence_tensor < 0) or torch.any(sequence_tensor >= 27):
                            print(f"Invalid residue encoding in group {name}, skipping.")
                            continue

                        coords_tensor = torch.tensor(coords, dtype=torch.float32)
                        # coords_tensor has shape (T, N, 3)
                        # sequence_tensor has shape (N,)
                        data = Data(x=sequence_tensor.unsqueeze(1), coords_ca=coords_tensor)
                        data_list.append(data)
                    else:
                        print(f"Warning: 'sequence' attribute not found in group {name}, skipping.")
    return data_list


def custom_collate_fn(batch, max_num_ca=500, max_T=1000):
    """
    Custom collate function to pad sequences and coordinates.

    Args:
        batch (List[Data]): List of Data objects.
        max_num_ca (int): Maximum number of amino acids.
        max_T (int): Maximum number of time steps.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Padded x, coords_ca, masks, batch_indices.
    """
    max_num_atoms = max_num_ca  # Directly set to max_num_ca
    current_max_T = min(max(data.coords_ca.shape[0] for data in batch), max_T)

    x_list = []
    coords_ca_list = []
    mask_list = []
    batch_indices_list = []

    for i, data in enumerate(batch):
        N = data.x.shape[0]
        T = data.coords_ca.shape[0]

        if N > max_num_ca:
            print(f"Length {N} exceeds max_num_ca {max_num_ca}. Truncating.")
            data.x = data.x[:max_num_ca]
            data.coords_ca = data.coords_ca[:, :max_num_ca, :]
            N = max_num_ca

        if T > current_max_T:
            print(f"Time steps {T} exceed current_max_T {current_max_T}. Truncating to {current_max_T}.")
            data.coords_ca = data.coords_ca[:current_max_T, :, :]
            T = current_max_T

        # Pad x
        x_padded = torch.zeros((max_num_ca, 1), dtype=torch.long)
        x_padded[:N] = data.x
        x_list.append(x_padded)

        # Pad coords_ca
        coords_padded = torch.zeros((current_max_T, max_num_ca, 3), dtype=torch.float32)
        coords_padded[:T, :N, :] = data.coords_ca
        coords_ca_list.append(coords_padded)

        # Create mask for valid amino acids
        mask = torch.zeros((max_num_ca,), dtype=torch.bool)
        mask[:N] = 1
        mask_list.append(mask)

        # Create batch indices for valid nodes
        # For each valid node, assign the batch index `i`
        batch_indices = torch.full((N,), i, dtype=torch.long)
        batch_indices_list.append(batch_indices)

    # Stack tensors
    x = torch.stack(x_list, dim=0)  # Shape: (batch_size, max_num_atoms, 1)
    coords_ca = torch.stack(coords_ca_list, dim=0)  # Shape: (batch_size, current_max_T, max_num_atoms, 3)
    masks = torch.stack(mask_list, dim=0)  # Shape: (batch_size, max_num_atoms)
    batch_indices = torch.cat(batch_indices_list, dim=0)  # Shape: (total_valid_nodes,)
    assert x.shape[1] <= max_num_ca, f"Batch size exceeds max_num_ca: {x.shape[1]} > {max_num_ca}"
    assert coords_ca.shape[2] <= max_num_ca, f"Coords_CA nodes exceed max_num_ca: {coords_ca.shape[2]} > {max_num_ca}"

    return x, coords_ca, masks, batch_indices


def setup_logging(log_dir):
    """
    Setup logging.

    Args:
        log_dir (str): Directory to save logs.
    """
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


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='ProAutoMD Training Script with Config File')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configurations with error handling
    try:
        data_config = config['data']
        model_config = config['model']
        training_config = config['training']
        paths_config = config['paths']
        gpu_config = config.get('gpu', {'use_multi_gpu': False})
    except KeyError as e:
        print(f"Missing configuration section: {e}")
        raise

    # Setup logging
    setup_logging(paths_config['log_dir'])
    logging.info("Starting training process")

    # Log loaded configurations for debugging
    logging.info(f"Data Config: {data_config}")
    logging.info(f"Model Config: {model_config}")
    logging.info(f"Training Config: {training_config}")
    logging.info(f"Paths Config: {paths_config}")
    logging.info(f"GPU Config: {gpu_config}")

    # Load dataset
    logging.info(f"Loading dataset from {data_config['path']}")
    dataset = load_dataset(data_config['path'])
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    logging.info(f"Dataset split into train: {train_size}, val: {val_size}, test: {test_size}")

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(
            batch,
            max_num_ca=data_config['max_num_ca'],
            max_T=data_config['max_T']
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(
            batch,
            max_num_ca=data_config['max_num_ca'],
            max_T=data_config['max_T']
        )
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(
            batch,
            max_num_ca=data_config['max_num_ca'],
            max_T=data_config['max_T']
        )
    )
    logging.info("DataLoaders prepared")

    # Initialize model
    model = ProAutoMD(
        layers=model_config['layers'],
        hidden_dim=model_config['hidden_dim'],
        max_num_ca=data_config['max_num_ca'],
        attn=model_config['attn']
    )
    model.to(device)

    # Use multiple GPUs if available and specified in config
    if gpu_config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        if gpu_config.get('use_multi_gpu', False):
            logging.warning("Multiple GPUs requested but not available. Using single GPU or CPU.")
        else:
            logging.info("Using single GPU or CPU")

    # Initialize optimizer with type casting
    try:
        lr = float(training_config['lr'])
        weight_decay = float(training_config['weight_decay'])
        kl_weight = float(training_config['kl_weight'])
    except ValueError as e:
        logging.error(f"Invalid type for training hyperparameters: {e}")
        raise

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )

    # Define loss function
    criterion_aa = nn.CrossEntropyLoss()

    # Create checkpoint directory
    os.makedirs(paths_config['checkpoint_dir'], exist_ok=True)

    # Initialize a list to store embeddings
    all_embeddings = []

    # Training loop
    for epoch in range(1, int(training_config['num_epochs']) + 1):
        model.train()
        total_loss = 0
        total_loss_aa = 0
        total_loss_kl = 0
        logging.info(f"\nStarting Epoch {epoch}/{int(training_config['num_epochs'])}")

        for batch_idx, batch in enumerate(train_loader, 1):
            logging.info(f"Processing Batch {batch_idx}")
            # Unpack the returned values, including batch_indices
            x, coords_ca, masks, batch_indices = batch

            # Move data to device
            x, coords_ca, masks, batch_indices = (
                x.to(device),
                coords_ca.to(device),
                masks.to(device),
                batch_indices.to(device),
            )
            logging.info("Data moved to device")

            optimizer.zero_grad()

            # Forward pass
            coords_ca_pred, aa_pred, pad_pred, kl_h, z_h = model(x, coords_ca, masks, batch_indices)
            logging.info("Forward pass completed")

            # Expand masks to match coords_ca_pred shape
            masks_expanded = masks.unsqueeze(1).expand(-1, coords_ca.shape[1], -1)  # Shape: [B, T, 500]

            # Expand target x to match aa_pred shape
            x_expanded = x.squeeze(-1).unsqueeze(1).expand(-1, coords_ca.shape[1], -1)  # Shape: [B, T, N]

            # Apply mask to select valid amino acids
            aa_pred_valid = aa_pred[masks_expanded]        # Shape: [B * T * 500_valid, 26]
            x_valid = x_expanded[masks_expanded]          # Shape: [B * T * 500_valid]

            logging.info(f"aa_pred_valid shape: {aa_pred_valid.shape}")
            logging.info(f"x_valid shape: {x_valid.shape}")

            try:
                # Compute amino acid type prediction loss
                loss_aa = criterion_aa(aa_pred_valid, x_valid)
                logging.info(f"AA Prediction Loss: {loss_aa.item():.4f}")

                # Compute KL divergence loss
                loss_kl = kl_weight * kl_h
                logging.info(f"KL Loss: {loss_kl.item():.4f}")

                # Compute total loss
                loss = loss_aa + loss_kl
                logging.info(f"Total Loss: {loss.item():.4f}")

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error("Loss is NaN or Inf. Stopping training.")
                    break

                # Backward pass
                loss.backward()
                logging.info("Backward pass completed")

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                logging.info("Gradient clipping applied")

                # Optimizer step
                optimizer.step()
                logging.info("Optimizer step completed")

                # Accumulate losses
                total_loss += loss.item()
                total_loss_aa += loss_aa.item()
                total_loss_kl += loss_kl.item()

                # Detach and store embeddings
                all_embeddings.append(z_h.detach().cpu())
            except Exception as e:
                logging.error(f"Error during loss computation or backpropagation: {e}")
                break

        # Calculate average losses for the epoch
        avg_loss = total_loss / batch_idx
        avg_loss_aa = total_loss_aa / batch_idx
        avg_loss_kl = total_loss_kl / batch_idx
        logging.info(f"Epoch {epoch} completed - Average Total Loss: {avg_loss:.4f}, "
                     f"Average AA Loss: {avg_loss_aa:.4f}, Average KL Loss: {avg_loss_kl:.4f}")

        # Validate the model
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_loss_aa = 0
            val_loss_kl = 0
            for val_batch_idx, val_batch in enumerate(val_loader, 1):
                x, coords_ca, masks, batch_indices = val_batch
                x, coords_ca, masks, batch_indices = (
                    x.to(device),
                    coords_ca.to(device),
                    masks.to(device),
                    batch_indices.to(device),
                )

                coords_ca_pred, aa_pred, pad_pred, kl_h, z_h = model(x, coords_ca, masks, batch_indices)

                masks_expanded = masks.unsqueeze(1).expand(-1, coords_ca.shape[1], -1)
                x_expanded = x.squeeze(-1).unsqueeze(1).expand(-1, coords_ca.shape[1], -1)

                aa_pred_valid = aa_pred[masks_expanded]
                x_valid = x_expanded[masks_expanded]

                loss_aa = criterion_aa(aa_pred_valid, x_valid)
                loss_kl = kl_weight * kl_h
                loss = loss_aa + loss_kl

                val_loss += loss.item()
                val_loss_aa += loss_aa.item()
                val_loss_kl += loss_kl.item()

            avg_val_loss = val_loss / val_batch_idx
            avg_val_loss_aa = val_loss_aa / val_batch_idx
            avg_val_loss_kl = val_loss_kl / val_batch_idx
            logging.info(f"Validation - Average Total Loss: {avg_val_loss:.4f}, "
                         f"Average AA Loss: {avg_val_loss_aa:.4f}, Average KL Loss: {avg_val_loss_kl:.4f}")

        # Save model checkpoints every 2 epochs and at the end of training
        if epoch % 2 == 0 or epoch == int(training_config['num_epochs']):
            checkpoint_path = os.path.join(paths_config['checkpoint_dir'], f'ProAutoMD_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model checkpoint saved at {checkpoint_path}")

    if __name__ == '__main__':
        main()

