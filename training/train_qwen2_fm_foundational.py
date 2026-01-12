"""
train_qwen2_fm_foundational.py

Tiny fine-tune of Qwen2-1.5B-Instruct on your foundational Fractal Monism JSONL.

- Uses train_foundational_train.jsonl for training
- Uses train_foundational_dev.jsonl for eval
- Each example is formatted as: "Q: ...\\nA: ..."

This is a SMALL first run meant to see whether Qwen starts to answer
more like your FM corpus on core questions (e.g. time, duality, etc.).
"""

import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]  # /home/ubuntu/resse-core
DATA_DIR = ROOT / "datasets" / "foundational"

TRAIN_JSONL = DATA_DIR / "train_foundational_train.jsonl"
DEV_JSONL = DATA_DIR / "train_foundational_dev.jsonl"

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = ROOT / "models" / "qwen2-1.5b-fm-foundational-tiny"


class FMDataset(Dataset):
    """
    Simple PyTorch dataset that:
    - reads your JSONL (with 'input' and 'output' fields)
    - turns each into "Q: {input}\\nA: {output}\\n"
    - tokenizes to fixed length
    """

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 512, max_examples: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("input", "").strip()
                a = obj.get("output", "").strip()

                if not q or not a:
                    continue

                text = f"Q: {q}\nA: {a}\n"
                self.texts.append(text)

        # Optional: shuffle and take a subset for a tiny experiment
        random.shuffle(self.texts)
        if max_examples is not None:
            self.texts = self.texts[:max_examples]

        print(f"Loaded {len(self.texts)} examples from {jsonl_path.name}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # squeeze batch dim
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # causal LM: labels are input_ids (Trainer will handle shifting)
        item["labels"] = item["input_ids"].clone()
        return item


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. Load tokenizer & model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )

    # 2. Build datasets (small subset for first run)
    max_train = 256  # you can increase later
    max_dev = 128

    train_dataset = FMDataset(TRAIN_JSONL, tokenizer, max_length=512, max_examples=max_train)
    eval_dataset = FMDataset(DEV_JSONL, tokenizer, max_length=512, max_examples=max_dev)

    # 3. Training configuration (tiny run)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,

        # Updated argument names for newer transformers
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,

        fp16=False,    # turn off AMP; we already loaded the model in float16
        report_to=[],
    )
    def compute_metrics(eval_pred):
        # simple perplexity metric
        import math
        from transformers.trainer_utils import EvalPrediction

        if isinstance(eval_pred, tuple):
            logits, labels = eval_pred
        elif isinstance(eval_pred, EvalPrediction):
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            logits, labels = eval_pred

        # loss is already handled by Trainer; we just compute perplexity from it if needed
        # Here we just return an empty dict to avoid complexity.
        return {}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 4. Train
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 5. Save model + tokenizer
    print("Saving model to", OUTPUT_DIR)
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
