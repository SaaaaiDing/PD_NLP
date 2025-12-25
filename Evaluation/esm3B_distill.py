import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import random
from typing import Optional
from seq2fn import ProClipStudent

class ESMChainAEmbedder:
    """
    Uses a pretrained ESM model (esm2_t33_650M_UR50D) to embed chain A.
    """
    def __init__(self,max_seq_len=1024, device='cuda'):
        self.max_seq_len = max_seq_len
        self.device = device
        self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.embed_dim = 2560
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed_chain(self, chain_seqs: list) -> (torch.Tensor, torch.Tensor):
        """
        chain_seq: a string for chain A, e.g. "MKVL..."
        returns: a [1, L, d_esm] embedding (L = length of chain A)
        """
        data = [("chainA_{}".format(i), seq) for i, seq in enumerate(chain_seqs)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.model(tokens, repr_layers=[33], need_head_weights=False)

        embeddings = results["representations"][33][:, 1:-1, :]          # remove [CLS], [EOS]

        batch_size, seq_len, feat_dim = embeddings.shape
        padded_embeddings = torch.zeros(
            (batch_size, self.max_seq_len, feat_dim),
            device=self.device
        )

        masks = torch.zeros(
            (batch_size, self.max_seq_len),
            dtype=torch.bool,
            device=self.device
        )

        valid_lengths = [min(len(s), self.max_seq_len) for s in chain_seqs]
        for i, valid_len in enumerate(valid_lengths):
            padded_embeddings[i, :valid_len] = embeddings[i, :valid_len]
            masks[i, :valid_len] = True

        return padded_embeddings, masks  # shape [1, L, d_esm]

class MDPretrainedEmbedder(nn.Module):
    """
    ProClipStudent
    """
    def __init__(self, checkpoint_path: str, max_seq_len=1024, device='cuda', freeze=True):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        
        # ProClipStudent
        self.student = ProClipStudent(d_projection=512)  #  d_projection=512
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.student.load_state_dict(ckpt['model_state'])
        self.student.to(device)
        self.student.eval()
        
        if freeze:
            for param in self.student.parameters():
                param.requires_grad = False

    def forward(self, esm_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        esm_emb: [B, max_seq_len, 1280]
        mask:    [B, max_seq_len]
        returns: [B, max_seq_len, 512]
        """
        with torch.no_grad():
            student_out = self.student(esm_emb)
            geom_emb = student_out['geom_emb']  # [B, max_seq_len, 512]
        
        # 
        geom_emb = geom_emb * mask.unsqueeze(-1)
        return geom_emb


class ChainBAutoregressiveModel(nn.Module):
    """
    A Transformer-based autoregressive decoder that:
      - Generates chain B token-by-token, conditioned on chain A's ESM embedding.
      - Random chain B length in [min_len..max_len].
      - Similarity penalty to discourage chain B ~ chain A.
    """
    def __init__(
        self,
        alphabet,
        d_model=2560,          # ESM dimension
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        min_len=40,
        max_len=256,
        similarity_weight=0.1
    ):
        super().__init__()
        self.alphabet = alphabet
        self.vocab_size = len(alphabet)
        self.d_model = d_model
        self.min_len = min_len
        self.max_len = max_len
        self.similarity_weight = similarity_weight

        # Embedding for chain B
        self.embed_b = nn.Embedding(self.vocab_size, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.ln_out = nn.Linear(d_model, self.vocab_size)

    def forward(
        self,
        chainB_tokens: torch.Tensor,       # [B, Lb], tokenized
        chainA_emb: torch.Tensor,          # [B, La, d_model], memory
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, Lb = chainB_tokens.shape
        b_embed = self.embed_b(chainB_tokens)        # [B, Lb, d_model]
        b_embed = b_embed.transpose(0, 1)            # [Lb, B, d_model]
        memory = chainA_emb.transpose(0, 1)          # [La, B, d_model]

        dec_out = self.decoder(
            b_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [Lb, B, d_model]
        dec_out = dec_out.transpose(0, 1)            # [B, Lb, d_model]
        logits = self.ln_out(dec_out)                # [B, Lb, vocab_size]
        return logits

    def sample_chain_length(self) -> int:
        """
        Randomly pick chain B length in [min_len..max_len].
        """
        return random.randint(self.min_len, self.max_len)

    def tokenize(self, chain_seq: str) -> torch.Tensor:
        """
        Tokenizes chain B using ESM's alphabet (without [CLS]/[EOS]).
        """
        data = [("chainB", chain_seq)]
        _, _, tokens = self.alphabet.get_batch_converter()(data)
        return tokens[:, 1:-1]  # remove [CLS], [EOS]

    def generate(
        self,
        chainA_emb: torch.Tensor,
        max_length: Optional[int] = None,
        start_token: int = 2,  # for demonstration
        end_token: int = 3     # for demonstration
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation, up to random length or user-defined max_length.
        """
        B, _, _ = chainA_emb.shape
        device = chainA_emb.device
        length = max_length if max_length else self.sample_chain_length()

        generated = torch.full((B, 1), start_token, dtype=torch.long, device=device)
        for _ in range(length):
            logits = self.forward(generated, chainA_emb)  # [B, L, vocab_size]
            next_token_logits = logits[:, -1, :]          # [B, vocab_size]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_token).all():
                break

        return generated

class CombinedDesign(nn.Module):
    """ """
    def __init__(
        self,
        esm_embed_dim=2560,
        md_embed_dim=512,
        max_seq_len=1024,
        vocab_size=33,
        checkpoint_path="./distill_model.pt",
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()[1]  # 
        
        # 
        self.esm_embedder = ESMChainAEmbedder(max_seq_len=max_seq_len, device=device)
        self.md_embedder = MDPretrainedEmbedder(
            checkpoint_path=checkpoint_path,
            max_seq_len=max_seq_len,
            device=device,
            freeze=True
        )
        self.md_proj = nn.Linear(md_embed_dim, esm_embed_dim).to(device)
        self.chainB_decoder = ChainBAutoregressiveModel(
            alphabet=self.alphabet,  # 
            d_model=esm_embed_dim,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        ).to(device)

    def forward(self, chainA_seqs: list, chainB_tokens: torch.Tensor) -> torch.Tensor:
        esm_emb, mask = self.esm_embedder.embed_chain(chainA_seqs)
        md_emb = self.md_embedder(esm_emb, mask)
        fused_emb = esm_emb + self.md_proj(md_emb)
        return self.chainB_decoder(chainB_tokens, fused_emb)

    def generate(self, chainA_seq: str, max_length=256) -> torch.Tensor:
        esm_emb, mask = self.esm_embedder.embed_chain(chainA_seq)
        md_emb = self.md_embedder(esm_emb, mask)
        fused_emb = esm_emb + self.md_proj(md_emb)
        
        generated = torch.full((1, 1), self.alphabet.cls_idx, dtype=torch.long, device=self.device)
        for _ in range(max_length):
            logits = self.chainB_decoder(generated, fused_emb)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == self.alphabet.eos_idx:
                break
        return generated[:, 1:-1]  #

    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Converts token IDs to a string."""
        return ''.join([self.alphabet.get_tok(p) for p in token_ids.squeeze().cpu().numpy()])  

    # def tokenize(self, chain_seq: str) -> torch.Tensor:
    #     """Tokenizes chain B using ESM's alphabet."""
    #     data = [("chainB", chain_seq)]
    #     _, _, tokens = self.alphabet.get_batch_converter()(data)
    #     return tokens[:, 1:-1] 

