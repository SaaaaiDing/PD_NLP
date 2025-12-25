import torch
import yaml
import os
from esm3B_distill import ESMChainAEmbedder, MDPretrainedEmbedder, ChainBAutoregressiveModel, CombinedDesign  

# 
with open('esm3B_distill.yaml', 'r') as f:
    config = yaml.safe_load(f)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_TOKEN_IDX = len(AMINO_ACIDS)  # 20
PAD_TOKEN_IDX = UNK_TOKEN_IDX + 1 # 21
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}
IDX_TO_AA[UNK_TOKEN_IDX] = 'X'
IDX_TO_AA[PAD_TOKEN_IDX] = ''

# If you have the definitions for ESMChainAEmbedder, MDPretrainedEmbedder,
# ChainBAutoregressiveModel, CombinedDesign in a separate file (e.g., model_defs.py),
# you could do:
# from model_defs import ESMChainAEmbedder, MDPretrainedEmbedder, ChainBAutoregressiveModel, CombinedDesign

# -------------------------------------------------------------------
# 1. Define a function to load the trained model from checkpoint
# -------------------------------------------------------------------
def load_trained_model(checkpoint_path: str, device: str = 'cuda') -> CombinedDesign:
    """
    Loads the CombinedDesign model from the specified checkpoint.
    
    :param checkpoint_path: Path to 'esmd_model.pt' (the best model checkpoint).
    :param device: 'cuda' or 'cpu'
    :return: An instance of CombinedDesign with pretrained weights.
    """
    # Load the checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device)

    # Extract config if it was saved, or use the relevant parameters
    # the training script saved 'config' and 'vocab_info' in checkpoint_data; 
    # you can adapt as needed. For demonstration, let's assume typical defaults:

    config = checkpoint_data.get('config', {})  # or fallback if needed
    model_cfg = config.get('model', {})
    
    # Initialize the model. 
    # Make sure the arguments here match those used in your training script. 
    # If you manually know your embed dims, checkpoint paths, etc., you can just hardcode them.
    model = CombinedDesign(
        esm_embed_dim=model_cfg.get('esm_embed_dim', 2560),
        md_embed_dim=model_cfg.get('md_embed_dim', 512),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
        vocab_size=len("AMINO_ACIDS") + 2,  #
        checkpoint_path=config['model']['distill_checkpoint'],
        device=device
     )

    # Load the trained model parameters
    model.load_state_dict(checkpoint_data['model_state'], strict=True)
    model.to(device)
    model.eval()

    return model


# -------------------------------------------------------------------
# 2. Provide a function that uses the model for Chain B generation
# -------------------------------------------------------------------
def design_proteins_from_chainA(
    model: CombinedDesign, 
    chainA_seqs: list,
    max_length: int = 256
) -> list:
    """
    Given a set of Chain A sequences, use the model to generate Chain B designs.
    
    :param model: An instance of CombinedDesign (with loaded weights).
    :param chainA_seqs: A list of Chain A sequences (strings).
    :param max_length: Maximum length for the generated chain B.
    :return: A list of designed chain B sequences (strings).
    """
    device = next(model.parameters()).device
    chainB_designed = []

    # The model's 'generate' function in the provided code expects 
    # a list for embedding: we can adapt to batch calls or do them one at a time.
    for seq in chainA_seqs:
        # Our model.generate(...) was written to handle a single sequence at a time, 
        # but it expects a list [seq]. Check your final version; if it needs retooling, do so.
        # For demonstration, we do:
        with torch.no_grad():
            # Generate tokens
            tokens_2d = model.generate(seq, max_length=max_length)  # shape [1, L]
            # Detokenize
            chainB_str = model.detokenize(tokens_2d[0])
        chainB_list.append(chainB_str)
    return chainB_list


# -------------------------------------------------------------------
# 3. Main execution / usage example
# -------------------------------------------------------------------
if __name__ == "__main__":
    # The path to your best model checkpoint
    BEST_MODEL_CKPT = "esmd_model.pt"  # Adapt if needed
    
    # 3.1 Load the trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(checkpoint_path=BEST_MODEL_CKPT, device=device)

    # 3.2 Suppose we have a batch of 2-3 Chain A sequences
    chainA_batch = [
        "EQLLFLYIIYTVGYALSFSALVIASAILLGFRHLHCTRNYIHLNWIFRLYR",    
        #"QRASFGTVLDAIKNIQSTH",   # Another short chain
        # ... add as many as you need
    ]

    # 3.3 Generate designs (Chain B) for each
    chainB_output = design_proteins_from_chainA(model, chainA_batch, max_length=128)

    # 3.4 Print results
    for i, cb in enumerate(chainB_output):
        print(f"Chain A {i}: {chainA_batch[i]}")
        print(f"Designed Chain B {i}: {cb}")
        print("--------------------------------------------------------")
