import yaml
import torch

# Import the classes from your seq2md.py (adjust if necessary)
from seq2md import ProClipMDModel, ESMChainAEmbedder

def generate_geometry_embedding_from_sequence(
    sequence: str,
    config_path: str = "/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MD_Pretrain/seq2md.yaml",
    model_ckpt: str = "/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MD_Pretrain/seq2md_eval2_out/model_epoch_5.pt",
    device: str = "cuda"
) -> torch.Tensor:
    """
    Given an amino-acid sequence string, load the trained ProClipMDModel in eval mode,
    embed the sequence with ESM, and produce a geometry-based embedding vector.
    
    Args:
        sequence   (str): Protein sequence, e.g. "MKVL..." 
        config_path(str): Path to the YAML config file used for training.
        model_ckpt (str): Path to the saved model checkpoint (best_model.pt).
        device     (str): "cuda" or "cpu" depending on your setup.

    Returns:
        embedding (torch.Tensor): A 1D geometry embedding tensor of shape [geom_dim],
                                  typically [256] by default.
    """
    # -------------------------------------------------------------------------
    # 1) Load training config
    # -------------------------------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # 2) Instantiate the model and load checkpoint
    # -------------------------------------------------------------------------
    model = ProClipMDModel(config).to(device)
    state_dict = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # set to evaluation mode

    # -------------------------------------------------------------------------
    # 3) Create an ESM embedder and embed the input sequence
    # -------------------------------------------------------------------------
    esm_embedder = ESMChainAEmbedder(device=device)
    seq_esm = esm_embedder.embed_chain(sequence)  # shape [1, L, 1280]
    # This returns a single-batch embedding (B=1). 
    # If you want to handle multiple sequences, you could batch them.

    # -------------------------------------------------------------------------
    # 4) Use the model's built-in inference method to get a geometry embedding
    # -------------------------------------------------------------------------
    with torch.no_grad():
        # inference_from_seq will generate a geometry embedding (geom_global)
        # by creating dummy coordinates internally.
        geom_global = model.inference_from_seq(seq_esm)  # shape [1, 256] by default

    # geom_global[0] => the embedding vector (since B=1)
    return geom_global[0]


if __name__ == "__main__":
    # Example usage
    config_path = "/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MD_Pretrain/seq2md.yaml"
    model_ckpt = "/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MD_Pretrain/seq2md_eval2_out/model_epoch_5.pt"
    sample_sequence = "MKTAYIAKQRQISFVKSHFSRQDIL"  # Replace with your protein sequence
    embedding = generate_geometry_embedding_from_sequence(
        sequence=sample_sequence,
        config_path="/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MD_Pretrain/seq2md.yaml",
        model_ckpt="/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MD_Pretrain/seq2md_eval2_out/model_epoch_5.pt",
        device="cuda"
    )
    print("Output embedding shape:", embedding.shape)  # e.g., torch.Size([256])
    print("Embedding vector (truncated):", embedding[:10])  # Just peek at first 10 dims
