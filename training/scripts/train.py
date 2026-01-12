"""
train.py (skeleton)

First step:
  - load model_150m.yaml
  - print config summary
  - (optionally) load a few sequences using dataset_loader
No actual training yet.
"""

import yaml
from pathlib import Path
from dataset_loader import demo_load_foundational
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]  # /home/ubuntu/resse-core
CONFIG_PATH = ROOT / "training" / "configs" / "model_150m.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        print("❌ Config not found:", CONFIG_PATH)
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    print("ROOT:", ROOT)
    print("CONFIG:", CONFIG_PATH)

    cfg = load_config()
    if cfg is None:
        return

    print("\n--- Loading tokenizer ---")
    tok_path = cfg["tokenizer"]["path"]
    tokenizer = Tokenizer.from_file(tok_path)
    print("Tokenizer loaded! Vocab size:", tokenizer.get_vocab_size())

    # 🔹 NEW: run the tokenization demo
    demo_tokenization(tokenizer)

    print("\nModel name:", cfg["model"]["name"])
    print("d_model:", cfg["model"]["d_model"])
    print("n_layers:", cfg["model"]["n_layers"])
    print("n_heads:", cfg["model"]["n_heads"])
    print("vocab_size:", cfg["model"]["vocab_size"])
    print("max_seq_len:", cfg["model"]["max_seq_len"])

    print("\nTrain files:")
    for p in cfg["data"]["train_files"]:
        print("  -", p)

    print("\nDev files:")
    for p in cfg["data"]["dev_files"]:
        print("  -", p)

    print("\n--- Demo: load a few foundational sequences ---")
    demo_load_foundational(n_examples=3)

def demo_tokenization(tokenizer):
    test_q = "How does Fractal Monism conceptualize time?"
    test_a = "Time emerges as an effect of recursive perception, not a physical dimension."
    from sequence_builder import build_qa_sequence
    seq = build_qa_sequence(test_q, test_a)

    print("\n--- Demo tokenization ---")
    print("Sequence:\n", seq)

    enc = tokenizer.encode(seq)
    print("Token count:", len(enc.ids))
    print("First 40 token IDs:", enc.ids[:40])
    print("Decoded round-trip:", tokenizer.decode(enc.ids[:80]))


if __name__ == "__main__":
    main()
