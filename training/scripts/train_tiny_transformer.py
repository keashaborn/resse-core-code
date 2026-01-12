"""
train_tiny_transformer.py

Tiny training loop for a small transformer on a subset of the foundational dataset.
This is a practice run to understand shapes, loss, training dynamics, dev eval, and
simple text generation. It does NOT train the real 150M model.
"""

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from sequence_builder import build_qa_sequence

ROOT = Path(__file__).resolve().parents[2]  # /home/ubuntu/resse-core
DATASETS = ROOT / "datasets"
DEVICE = torch.device("cpu")  # later we can switch to cuda if available

# ----------- FM-style prompts for probing generalization -----------

FM_PROMPTS = [
    "Q: How does Fractal Monism conceptualize time?\nA:",
    "Q: What is the role of distinction in Fractal Monism?\nA:",
    "Q: How does perception relate to reality in Fractal Monism?\nA:",
    "Q: Why does Fractal Monism treat identity as scale-dependent?\nA:",
    "Q: Does consciousness scale continuously in Fractal Monism?\nA:",
    "Q: How do oscillation and differentiation give rise to fractal structure?\nA:",
    "Q: Is duality a truth or a tool in Fractal Monism?\nA:",
    "Q: How does Fractal Monism view causation?\nA:",
]


# ----------- Tiny Transformer Model -----------

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=36000, d_model=256, n_heads=4, n_layers=2, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        """Forward pass.

        Args:
            input_ids: LongTensor [B, L]
            attention_mask: LongTensor [B, L] with 1=real tokens, 0=pad
        """
        B, L = input_ids.shape

        # token + positional embeddings
        x = self.embed(input_ids)  # [B, L, d_model]
        positions = torch.arange(0, L, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(positions)  # [B, L, d_model]

        # Transformer expects True for positions to ignore
        src_key_padding_mask = (attention_mask == 0)  # [B, L], bool

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, L, d_model]

        logits = self.lm_head(x)  # [B, L, vocab_size]
        return logits


# ----------- Data Helpers -----------

def iter_jsonl(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_foundational_sequences(n_max: int = 512):
    """Load up to n_max Q/A sequences from foundational/train split."""
    src = DATASETS / "foundational" / "train_foundational_train.jsonl"
    seqs = []
    for obj in iter_jsonl(src):
        q = obj.get("input", "")
        a = obj.get("output", "")
        seqs.append(build_qa_sequence(q, a))
        if len(seqs) >= n_max:
            break
    return seqs


def encode_batch(tokenizer, sequences, max_len=128, pad_id=0):
    """Tokenize and pad/truncate a list of strings.

    Returns:
        input_ids: LongTensor [B, L]
        attention_mask: LongTensor [B, L]
    """
    all_ids = [tokenizer.encode(s).ids for s in sequences]
    # Truncate long ones and find max length actually used
    true_max = max(len(ids) for ids in all_ids) if all_ids else 0
    max_len = min(max_len, true_max if true_max > 0 else max_len)

    padded = []
    masks = []
    for ids in all_ids:
        ids = ids[:max_len]
        attn = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(pad_id)
            attn.append(0)
        padded.append(ids)
        masks.append(attn)

    input_ids = torch.tensor(padded, dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=DEVICE)
    return input_ids, attention_mask


# ----------- Loss Function (language modeling) -----------

def compute_lm_loss(logits, input_ids, attention_mask):
    """Standard language modeling loss.

    We:
      - shift logits and targets by one position
      - ignore padding positions

    Args:
        logits: FloatTensor [B, L, V]
        input_ids: LongTensor [B, L]
        attention_mask: LongTensor [B, L]
    """
    B, L, V = logits.shape

    # Shift: predict token t+1 from position t
    logits_shifted = logits[:, :-1, :].contiguous()  # [B, L-1, V]
    targets = input_ids[:, 1:].contiguous()          # [B, L-1]
    mask = attention_mask[:, 1:].contiguous()        # [B, L-1]

    # Flatten
    logits_flat = logits_shifted.view(-1, V)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1).bool()

    # Only compute loss on non-pad positions
    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    loss = F.cross_entropy(logits_sel, targets_sel)
    return loss


# ----------- Dev Evaluation -----------

def evaluate_dev(model, dev_input_ids, dev_attention_mask):
    """Evaluate model on a dev split and return loss + perplexity."""
    model.eval()
    with torch.no_grad():
        logits = model(dev_input_ids, dev_attention_mask)
        loss = compute_lm_loss(logits, dev_input_ids, dev_attention_mask)
    ppl = math.exp(loss.item())
    return loss.item(), ppl


# ----------- Simple Text Generation (greedy) -----------

def generate_text(model, tokenizer, prompt, max_new_tokens=50, max_len=128):
    """Greedy decode: start from a prompt and let the tiny model continue."""
    model.eval()
    with torch.no_grad():
        ids = tokenizer.encode(prompt).ids

        # Keep only the last max_len tokens if prompt is long
        if len(ids) > max_len:
            ids = ids[-max_len:]

        # Try to find EOS token id if it exists
        try:
            eos_id = tokenizer.token_to_id("[EOS]")
        except Exception:
            eos_id = None

        for _ in range(max_new_tokens):
            # Build input tensors
            input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)  # [1, L]
            attn_mask = torch.ones_like(input_ids)  # no padding in generation prompt

            logits = model(input_ids, attn_mask)  # [1, L, V]
            next_logits = logits[0, -1, :]       # last position
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.argmax(probs).item())  # greedy

            ids.append(next_id)
            # Keep window at most max_len long
            if len(ids) > max_len:
                ids = ids[-max_len:]

            if eos_id is not None and next_id == eos_id:
                break

        text = tokenizer.decode(ids)
        return text


def generate_for_prompts(model, tokenizer, prompts, step, max_new_tokens=40, max_len=128):
    """Generate answers for a set of FM prompts at a given training step."""
    print(f"\n=== Sample generations at step {step} ===")
    for i, prompt in enumerate(prompts, start=1):
        sample = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, max_len=max_len)
        print(f"\n[{i}] Prompt:")
        print(prompt)
        print("Output:")
        print(sample)
    print("=========================================\n")


# ----------- Training Loop -----------

def main():
    # Load tokenizer
    tok_path = "/home/ubuntu/resse-core/tokenizer/resse-bpe-36k-v0b.json"
    tokenizer = Tokenizer.from_file(tok_path)

    print("Loading sequences...")
    sequences = load_foundational_sequences(n_max=512)
    print(f"Loaded {len(sequences)} sequences from foundational.")

    if not sequences:
        print("No sequences found. Aborting.")
        return

    # Encode all sequences once
    input_ids, attention_mask = encode_batch(tokenizer, sequences, max_len=128, pad_id=0)
    print("Encoded batch shape:", input_ids.shape)

    # Train/dev split (simple 90/10)
    num_examples = input_ids.size(0)
    split_idx = int(num_examples * 0.9)
    train_input_ids = input_ids[:split_idx]
    train_attention_mask = attention_mask[:split_idx]
    dev_input_ids = input_ids[split_idx:]
    dev_attention_mask = attention_mask[split_idx:]

    print(f"Train sequences: {train_input_ids.size(0)}")
    print(f"Dev sequences:   {dev_input_ids.size(0)}")

    # Initialize tiny model
    model = TinyTransformer(vocab_size=tokenizer.get_vocab_size()).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training hyperparams for the toy run
    n_steps = 300          # was 50
    batch_size = 8
    num_train = train_input_ids.size(0)
    train_indices = list(range(num_train))

    log_path = "tiny_foundational_log.txt"

    print("Starting tiny training loop...")

    for step in range(1, n_steps + 1):
        model.train()

        # sample a random mini-batch from TRAIN only
        batch_idx = random.sample(train_indices, k=min(batch_size, num_train))
        batch_input = train_input_ids[batch_idx]
        batch_mask = train_attention_mask[batch_idx]

        optimizer.zero_grad()
        logits = model(batch_input, batch_mask)
        loss = compute_lm_loss(logits, batch_input, batch_mask)
        loss.backward()
        optimizer.step()

        # Print train loss at key steps
        if step % 10 == 0 or step == 1 or step == 5:
            print(f"Step {step:03d} - train_loss: {loss.item():.4f}")

        # Every 10 steps, evaluate on DEV and log
        if step % 10 == 0 and dev_input_ids.size(0) > 0:
            dev_loss, dev_ppl = evaluate_dev(model, dev_input_ids, dev_attention_mask)
            print(f"          dev_loss: {dev_loss:.4f}  dev_ppl: {dev_ppl:.2f}")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"step={step}  train_loss={loss.item():.6f}  "
                    f"dev_loss={dev_loss:.6f}  dev_ppl={dev_ppl:.4f}\n"
                )

        # Every 100 steps, generate samples for multiple FM prompts
        if step % 100 == 0:
            generate_for_prompts(model, tokenizer, FM_PROMPTS, step, max_new_tokens=40, max_len=128)

    # Save a tiny checkpoint at the end
    ckpt_path = "tiny_resse_step_300.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Tiny training run complete. Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
