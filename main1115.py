import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data
from MD_LATENT import ProAutoMD  
import h5py
import numpy as np
from typing import List
import torch_geometric
from torch_geometric.utils import from_networkx, to_dense_batch
from torch_cluster import knn_graph
import argparse
import logging
import yaml
from torch.cuda.amp import GradScaler, autocast  

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define pdb_protein with aa2idx starting from 1
pdb_protein = {
    'amino_acid_names': [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ],
    'NON_STANDARD_SUBSTITUTIONS': {
        '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA',
        'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG', 'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP',
        'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA', 'BUC':'CYS',
        'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU',
        'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS', 'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS',
        'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS', 'CYM':'CYS',
        'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU',
        'DGN':'GLN', 'DHA':'ALA', 'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS',
        'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP', 'DTH':'THR', 'DTR':'TRP',
        'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY',
        'GLZ':'GLY', 'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS',
        'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO', 'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR',
        'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA',
        'MEN':'ASN', 'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET',
        'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU', 'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY',
        'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
        'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY',
        'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS', 'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS',
        'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA', 'TPL':'TRP',
        'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR',
        'TYS':'TYR', 'TYY':'TYR'
    },

    'amino_acid_abbr': {
        'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLU':'E', 'GLN':'Q', 'GLY':'G',
        'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S',
        'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
    },

    'aa2idx': {
        'ALA': 1,
        'ARG': 2,
        'ASN': 3,
        'ASP': 4,
        'CYS': 5,
        'GLU': 6,
        'GLN': 7,
        'GLY': 8,
        'HIS': 9,
        'ILE': 10,
        'LEU': 11,
        'LYS': 12,
        'MET': 13,
        'PHE': 14,
        'PRO': 15,
        'SER': 16,
        'THR': 17,
        'TRP': 18,
        'TYR': 19,
        'VAL': 20
    }

}

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
                        # Convert sequence to uppercase
                        sequence = sequence.upper()
                        # Encode amino acids using pdb_protein['aa2idx']
                        sequence_indices = []
                        for residue in sequence:
                            # Check if residue is a standard amino acid abbreviation
                            if residue in pdb_protein['amino_acid_abbr'].values():
                                # Get full name from abbreviation
                                full_name = next(
                                    (k for k, v in pdb_protein['amino_acid_abbr'].items() if v == residue), None
                                )
                                if full_name and full_name in pdb_protein['aa2idx']:
                                    sequence_indices.append(pdb_protein['aa2idx'][full_name])
                                else:
                                    sequence_indices.append(0)  # Unknown amino acid
                            else:
                                # Check if residue is a non-standard substitution
                                if residue in pdb_protein['NON_STANDARD_SUBSTITUTIONS']:
                                    standard_aa = pdb_protein['NON_STANDARD_SUBSTITUTIONS'][residue]
                                    if standard_aa in pdb_protein['aa2idx']:
                                        sequence_indices.append(pdb_protein['aa2idx'][standard_aa])
                                    else:
                                        sequence_indices.append(0)  # Unknown after substitution
                                else:
                                    sequence_indices.append(0)  # Unknown amino acid

                        sequence_tensor = torch.tensor(sequence_indices, dtype=torch.long)

                        # Validate indices
                        if torch.any(sequence_tensor < 0) or torch.any(sequence_tensor > 20):
                            print(f"Invalid residue encoding in group {name}, skipping.")
                            continue

                        coords_tensor = torch.tensor(coords, dtype=torch.float32)
                        # coords_tensor shape: (T, N, 3)
                        # sequence_tensor shape: (N,)
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
        # Assign the batch index `i` to each valid node
        batch_indices = torch.full((N,), i, dtype=torch.long)
        batch_indices_list.append(batch_indices)

    # Stack tensors
    x = torch.stack(x_list, dim=0)  # Shape: (batch_size, max_num_atoms, 1)
    coords_ca = torch.stack(coords_ca_list, dim=0)  # Shape: (batch_size, current_max_T, max_num_ca, 3)
    masks = torch.stack(mask_list, dim=0)  # Shape: (batch_size, max_num_ca)
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


def save_embeddings(all_embeddings, embeddings_path, logger):
    """
    Concatenate all embeddings and save them to a file.

    Args:
        all_embeddings (List[Tensor]): List of embedding tensors.
        embeddings_path (str): Path to save the concatenated embeddings.
        logger (logging.Logger): Logger for logging information.
    """
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)  # Shape: [Total Batches, hidden_dim]
        # Save embeddings
        torch.save(all_embeddings, embeddings_path)
        logger.info(f"\nSaved latent embeddings to '{embeddings_path}'")
        print(f"\nSaved latent embeddings to '{embeddings_path}'")
    else:
        logger.warning("\nNo embeddings were collected.")
        print("\nNo embeddings were collected.")


def evaluate_test_set(model, test_loader, criterion_aa, kl_weight, device, logger):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion_aa (nn.Module): Loss function for amino acid prediction.
        kl_weight (float): Weight for KL divergence in loss.
        device (torch.device): Device to run the evaluation on.
        logger (logging.Logger): Logger for logging information.
    """
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_loss_aa = 0
        test_loss_kl = 0
        for test_batch_idx, test_batch in enumerate(test_loader, 1):
            x, coords_ca, masks, batch_indices = test_batch
            x, coords_ca, masks, batch_indices = (
                x.to(device),
                coords_ca.to(device),
                masks.to(device),
                batch_indices.to(device),
            )

            try:
                coords_ca_pred, aa_pred, pad_pred, kl_h, z_h = model(x, coords_ca, masks, batch_indices)
                logger.info("Forward pass for test completed")
                print("Forward pass for test completed")
            except Exception as e:
                print(f"Error during forward pass in test: {e}")
                logger.error(f"Error during forward pass in test: {e}")
                break

            # Expand masks to match aa_pred shape
            masks_expanded = masks.unsqueeze(1).expand(-1, coords_ca.shape[1], -1)
            x_expanded = x.squeeze(-1).unsqueeze(1).expand(-1, coords_ca.shape[1], -1)

            # Apply mask to select valid amino acids
            aa_pred_valid = aa_pred[masks_expanded]
            x_valid = x_expanded[masks_expanded]

            loss_aa = criterion_aa(aa_pred_valid, x_valid)
            loss = loss_aa + kl_weight * kl_h

            test_loss += loss.item()
            test_loss_aa += loss_aa.item()
            test_loss_kl += kl_weight * kl_h.item()

        if test_batch_idx > 0:
            avg_test_loss = test_loss / test_batch_idx
            avg_test_loss_aa = test_loss_aa / test_batch_idx
            avg_test_loss_kl = test_loss_kl / test_batch_idx
            logger.info(f"Test - Average Total Loss: {avg_test_loss:.4f}, "
                        f"Average AA Loss: {avg_test_loss_aa:.4f}, Average KL Loss: {avg_test_loss_kl:.4f}")
            print(f"Test - Average Total Loss: {avg_test_loss:.4f}, "
                  f"Average AA Loss: {avg_test_loss_aa:.4f}, Average KL Loss: {avg_test_loss_kl:.4f}")
        else:
            logger.warning("No batches were processed in the test set.")
            print("No batches were processed in the test set.")


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='ProAutoMD Training Script with Config File')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configurations
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    paths_config = config['paths']
    gpu_config = config.get('gpu', {'use_multi_gpu': False})

    try:
        training_config['lr'] = float(training_config['lr'])
        training_config['weight_decay'] = float(training_config['weight_decay'])
        training_config['kl_weight'] = float(training_config['kl_weight'])
    except ValueError as e:
        print(f"Error in configuration file: {e}")
        logging.error(f"Error in configuration file: {e}")
        return

    # Setup logging
    setup_logging(paths_config['log_dir'])
    logger = logging.getLogger()
    logger.info("Starting training process")
    print("Logging setup completed")

    # Load dataset
    logger.info(f"Loading dataset from {data_config['path']}")
    try:
        dataset = load_dataset(data_config['path'])
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        logger.error(f"Failed to load dataset: {e}")
        return

    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    logger.info(f"Dataset split into train: {train_size}, val: {val_size}, test: {test_size}")
    print(f"Dataset split into train: {train_size}, val: {val_size}, test: {test_size}")

    # Prepare DataLoaders
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            pin_memory=True,  # For faster data transfer
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
            pin_memory=True,
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
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(
                batch,
                max_num_ca=data_config['max_num_ca'],
                max_T=data_config['max_T']
            )
        )
        logger.info("DataLoaders prepared")
        print("DataLoaders prepared")
    except Exception as e:
        print(f"Failed to prepare DataLoaders: {e}")
        logger.error(f"Failed to prepare DataLoaders: {e}")
        return

    # Initialize model
    try:
        model = ProAutoMD(
            layers=model_config['layers'],
            hidden_dim=model_config['hidden_dim'],
            max_num_ca=data_config['max_num_ca'],
            attn=model_config['attn'],
            num_types=21  # Ensure this matches your actual number of classes
        )
        model.to(device)
        logger.info("Model initialized and moved to device")
        print("Model initialized and moved to device")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        logger.error(f"Failed to initialize model: {e}")
        return

    # Use multiple GPUs if available and specified in config
    if gpu_config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        if gpu_config.get('use_multi_gpu', False):
            logger.warning("Multiple GPUs requested but not available. Using single GPU or CPU.")
            print("Multiple GPUs requested but not available. Using single GPU or CPU.")
        else:
            logger.info("Using single GPU or CPU")
            print("Using single GPU or CPU")

    # Initialize optimizer
    try:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config['lr'],
            betas=(0.9, 0.999),
            weight_decay=training_config['weight_decay']
        )
        logger.info("Optimizer initialized")
        print("Optimizer initialized")
    except Exception as e:
        print(f"Failed to initialize optimizer: {e}")
        logger.error(f"Failed to initialize optimizer: {e}")
        return

    # Define loss functions and kl_weight
    criterion_aa = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    kl_weight = training_config['kl_weight']
    logger.info("Loss function and kl_weight defined")
    print("Loss function and kl_weight defined")

    # Create checkpoint directory
    os.makedirs(paths_config['checkpoint_dir'], exist_ok=True)
    logger.info(f"Checkpoint directory set to {paths_config['checkpoint_dir']}")
    print(f"Checkpoint directory set to {paths_config['checkpoint_dir']}")

    # Initialize a list to store embeddings
    all_embeddings = []

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Training loop
    for epoch in range(1, training_config['num_epochs'] + 1):
        print(f"\nStarting Epoch {epoch}")
        logger.info(f"\nStarting Epoch {epoch}/{training_config['num_epochs']}")
        model.train()
        total_loss = 0
        total_loss_aa = 0
        total_loss_kl = 0

        for batch_idx, batch in enumerate(train_loader, 1):
            logger.info(f"Processing Batch {batch_idx}")
            print(f"Processing Batch {batch_idx}")
            # Unpack the returned values, including batch_indices
            x, coords_ca, masks, batch_indices = batch

            # Move data to device
            x, coords_ca, masks, batch_indices = (
                x.to(device),
                coords_ca.to(device),
                masks.to(device),
                batch_indices.to(device),
            )
            logger.info("Data moved to device")
            print("Data moved to device")

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                try:
                    coords_ca_pred, aa_pred, pad_pred, kl_h, z_h = model(x, coords_ca, masks, batch_indices)
                    logger.info("Forward pass completed")
                    print("Forward pass completed")
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    logger.error(f"Error during forward pass: {e}")
                    break

                # Expand masks to match aa_pred shape
                masks_expanded = masks.unsqueeze(1).expand(-1, coords_ca.shape[1], -1)  # Shape: [B, T, max_num_ca]

                # Expand target x to match aa_pred shape
                x_expanded = x.squeeze(-1).unsqueeze(1).expand(-1, coords_ca.shape[1], -1)  # Shape: [B, T, max_num_ca]

                # Apply mask to select valid amino acids
                aa_pred_valid = aa_pred[masks_expanded]        # Shape: [B * T * valid_nodes, 21]
                x_valid = x_expanded[masks_expanded]          # Shape: [B * T * valid_nodes]

                logger.info(f"aa_pred_valid shape: {aa_pred_valid.shape}")
                logger.info(f"x_valid shape: {x_valid.shape}")
                print(f"aa_pred_valid shape: {aa_pred_valid.shape}")
                print(f"x_valid shape: {x_valid.shape}")

                try:
                    # Compute amino acid type prediction loss
                    loss_aa = criterion_aa(aa_pred_valid, x_valid)
                    logger.info(f"AA Prediction Loss: {loss_aa.item():.4f}")
                    print(f"AA Prediction Loss: {loss_aa.item():.4f}")

                    # Compute total loss
                    loss = loss_aa + kl_weight * kl_h
                    logger.info(f"Total Loss: {loss.item():.4f}")
                    print(f"Total Loss: {loss.item():.4f}")

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error("Loss is NaN or Inf. Stopping training.")
                        print("Loss is NaN or Inf. Stopping training.")
                        break

                except Exception as e:
                    print(f"Error during loss computation: {e}")
                    logger.error(f"Error during loss computation: {e}")
                    break

            # Backward pass with mixed precision
            try:
                scaler.scale(loss).backward()
                logger.info("Backward pass completed")
                print("Backward pass completed")

                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                logger.info("Gradient clipping applied")
                print("Gradient clipping applied")

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                logger.info("Optimizer step completed")
                print("Optimizer step completed")

                # Accumulate losses
                total_loss += loss.item()
                total_loss_aa += loss_aa.item()
                total_loss_kl += kl_weight * kl_h.item()

                # Detach and store embeddings
                all_embeddings.append(z_h.detach().cpu())

                # Optionally, clear CUDA cache
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error during loss computation or backpropagation: {e}")
                logger.error(f"Error during loss computation or backpropagation: {e}")
                break

        # Calculate average losses for the epoch
        avg_loss = total_loss / batch_idx
        avg_loss_aa = total_loss_aa / batch_idx
        avg_loss_kl = total_loss_kl / batch_idx
        logger.info(f"Epoch {epoch} completed - Average Total Loss: {avg_loss:.4f}, "
                    f"Average AA Loss: {avg_loss_aa:.4f}, Average KL Loss: {avg_loss_kl:.4f}")
        print(f"Epoch {epoch} completed - Average Total Loss: {avg_loss:.4f}, "
              f"Average AA Loss: {avg_loss_aa:.4f}, Average KL Loss: {avg_loss_kl:.4f}")

        # Validate the model
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_loss_aa = 0
            val_loss_kl = 0
            for val_batch_idx, val_batch in enumerate(val_loader, 1):
                logger.info(f"Validating Batch {val_batch_idx}")
                print(f"Validating Batch {val_batch_idx}")
                x, coords_ca, masks, batch_indices = val_batch
                x, coords_ca, masks, batch_indices = (
                    x.to(device),
                    coords_ca.to(device),
                    masks.to(device),
                    batch_indices.to(device),
                )

                try:
                    coords_ca_pred, aa_pred, pad_pred, kl_h, z_h = model(x, coords_ca, masks, batch_indices)
                    logger.info("Forward pass for validation completed")
                    print("Forward pass for validation completed")
                except Exception as e:
                    print(f"Error during forward pass in validation: {e}")
                    logger.error(f"Error during forward pass in validation: {e}")
                    break

                # Expand masks to match aa_pred shape
                masks_expanded = masks.unsqueeze(1).expand(-1, coords_ca.shape[1], -1)
                x_expanded = x.squeeze(-1).unsqueeze(1).expand(-1, coords_ca.shape[1], -1)

                # Apply mask to select valid amino acids
                aa_pred_valid = aa_pred[masks_expanded]
                x_valid = x_expanded[masks_expanded]

                loss_aa = criterion_aa(aa_pred_valid, x_valid)
                loss = loss_aa + kl_weight * kl_h

                val_loss += loss.item()
                val_loss_aa += loss_aa.item()
                val_loss_kl += kl_weight * kl_h.item()

            if val_batch_idx > 0:
                avg_val_loss = val_loss / val_batch_idx
                avg_val_loss_aa = val_loss_aa / val_batch_idx
                avg_val_loss_kl = val_loss_kl / val_batch_idx
                logger.info(f"Validation - Average Total Loss: {avg_val_loss:.4f}, "
                            f"Average AA Loss: {avg_val_loss_aa:.4f}, Average KL Loss: {avg_val_loss_kl:.4f}")
                print(f"Validation - Average Total Loss: {avg_val_loss:.4f}, "
                      f"Average AA Loss: {avg_val_loss_aa:.4f}, Average KL Loss: {avg_val_loss_kl:.4f}")
            else:
                logger.warning("No batches were processed in the validation set.")
                print("No batches were processed in the validation set.")

        # Save model checkpoints every 2 epochs and at the end of training
        if epoch % 2 == 0 or epoch == training_config['num_epochs']:
            if isinstance(model, nn.DataParallel):
                checkpoint_path = os.path.join(paths_config['checkpoint_dir'], f'ProAutoMD_epoch_{epoch}.pt')
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                checkpoint_path = os.path.join(paths_config['checkpoint_dir'], f'ProAutoMD_epoch_{epoch}.pt')
                torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model checkpoint saved at {checkpoint_path}")
            print(f"Model checkpoint saved at {checkpoint_path}")

        # Save embeddings
    save_embeddings(all_embeddings, paths_config['embeddings_path'], logger)

        # Evaluate on test set
    evaluate_test_set(model, test_loader, nn.CrossEntropyLoss(ignore_index=0), training_config['kl_weight'], device, logger)

    # Define the entry point of the script
if __name__ == '__main__':
    main()

