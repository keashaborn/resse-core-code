"""
batch_encoding_demo.py

Demo:
  - load a few foundational Q/A sequences
  - encode with the RESSE tokenizer
  - pad/truncate to a fixed length
  - print shapes and first row
"""

import json
from pathlib import Path
from tokenizers import Tokenizer
from sequence_builder import build_qa_sequence

ROOT = Path(__file__).resolve().parents[2]  # /home/ubuntu/resse-core
DATASETS = ROOT / "datasets"
CONFIG_PATH = ROOT / "training" / "configs" / "model_150m.yaml"

def load_config():
    import yaml
    if not CONFIG_PATH.exists():
        print("❌ Config not found:", CONFIG_PATH)
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def iter_jsonl(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def sample_sequences_from_foundational(n=4):
    src = DATASETS / "foundational" / "train_foundational_train.jsonl"
    if not src.exists():
        print("❌ File not found:", src)
        return []

    seqs = []
    for i, obj in enumerate(iter_jsonl(src)):
        q = obj.get("input", "")
        a = obj.get("output", "")
        seqs.append(build_qa_sequence(q, a))
        if len(seqs) >= n:
            break
    return seqs

def pad_batch(encoded_ids, max_len=128, pad_id=0):
    """
    encoded_ids: list[list[int]]
    Returns:
      - padded_ids: list[list[int]]
      - attention_masks: list[list[int]]
    """
    # Option: auto-determine max_len from data, then clip to a cap
    true_max = max(len(ids) for ids in encoded_ids) if encoded_ids else 0
    max_len = min(max_len, true_max if true_max > 0 else max_len)

    padded = []
    masks = []
    for ids in encoded_ids:
        ids = ids[:max_len]
        attn = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(pad_id)
            attn.append(0)
        padded.append(ids)
        masks.append(attn)
    return padded, masks, max_len

def main():
    cfg = load_config()
    if cfg is None:
        return

    tok_path = cfg["tokenizer"]["path"]
    tokenizer = Tokenizer.from_file(tok_path)
    print("Tokenizer loaded. Vocab size:", tokenizer.get_vocab_size())

    seqs = sample_sequences_from_foundational(n=4)
    if not seqs:
        print("No sequences loaded.")
        return

    print(f"\nLoaded {len(seqs)} sequences from foundational.")
    for i, s in enumerate(seqs, 1):
        print(f"\n--- Sequence {i} (truncated preview) ---")
        print(s[:200])

    # Encode all sequences
    encoded = [tokenizer.encode(s).ids for s in seqs]
    print("\nRaw token lengths:", [len(ids) for ids in encoded])

    # Pad/truncate to fixed length
    padded_ids, masks, used_len = pad_batch(encoded, max_len=128, pad_id=0)

    print(f"\nPadded batch shape: {len(padded_ids)} x {used_len}")
    print("First row token IDs (first 40):", padded_ids[0][:40])
    print("First row attention mask (first 40):", masks[0][:40])

if __name__ == "__main__":
    main()
