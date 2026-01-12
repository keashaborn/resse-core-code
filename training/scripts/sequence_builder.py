"""
sequence_builder.py

Helpers for turning (input, output) pairs from the JSONL datasets
into text sequences the model will be trained on.
"""

def build_qa_sequence(question: str, answer: str) -> str:
    """
    Build a simple Q/A style sequence.
    We can change this later (e.g. add tags, roles, etc.).
    """
    question = question.strip()
    answer = answer.strip()
    return f"Q: {question}\nA: {answer}"

if __name__ == "__main__":
    # Simple sanity check
    q = "How does Fractal Monism conceptualize time?"
    a = "Fractal Monism describes time as an emergent property of perception..."
    seq = build_qa_sequence(q, a)
    print(seq)
