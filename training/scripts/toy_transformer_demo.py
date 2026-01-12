"""
toy_transformer_demo.py

This creates a tiny 2-layer transformer and runs a dummy forward pass
on your batch-encoded foundational sequences.

This is NOT the real 150M model.
It is only a sanity check for shapes + plumbing.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from sequence_builder import build_qa_sequence
from batch_encoding_demo import sample_sequences_from_foundational, pad_batch

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "training" / "configs" / "model_150m.yaml"

# ----- Tiny transformer block -----
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=36000, d_model=256, n_heads=4, n_layers=2, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection to vocab
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape

        # Token embeddings
        x = self.embed(input_ids)

        # Positional embeddings
        positions = torch.arange(0, L, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        # Transformer expects attention mask as True for masked, False for valid
        attn_mask = (attention_mask == 0)

        # Transformer forward
        x = self.encoder(x, src_key_padding_mask=attn_mask)

        # Predict next token
        logits = self.lm_head(x)   # [B, L, vocab]
        return logits

# ----- Main demo -----
def main():
    import yaml

    # Load tokenizer path from config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tok_path = cfg["tokenizer"]["path"]
    tokenizer = Tokenizer.from_file(tok_path)

    print("Tokenizer loaded.")

    # Sample sequences
    seqs = sample_sequences_from_foundational(n=4)
    encoded = [tokenizer.encode(s).ids for s in seqs]

    # Pad to 128 tokens
    padded_ids, masks, used_len = pad_batch(encoded, max_len=128, pad_id=0)

    # Convert to tensors
    input_ids = torch.tensor(padded_ids, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)

    print("Input tensor shape:", input_ids.shape)
    print("Attention mask shape:", attention_mask.shape)

    # Initialize tiny model
    model = TinyTransformer(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_len=128
    )

    # Run forward pass
    logits = model(input_ids, attention_mask)

    print("Output logits shape:", logits.shape)
    # Expected: [batch_size, seq_len, vocab_size]

if __name__ == "__main__":
    main()
