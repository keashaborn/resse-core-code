#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import hdbscan
import umap
import matplotlib.pyplot as plt

# ----- Paths -----

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

INPUT = RESULTS / "generated_qna_all_v3.jsonl"
EMB_PATH = RESULTS / "qna_v3_embeddings.npy"
CLUSTER_PATH = RESULTS / "qna_v3_clusters.jsonl"
SUMMARY_PATH = RESULTS / "qna_v3_cluster_summary.json"
FIG_PATH = RESULTS / "qna_v3_umap.png"

client = OpenAI()

# ----- Load Q&A -----

def load_items():
    items = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

# ----- Embed text -----

def embed_items(items):
    texts = [f"Q: {it['question']}\nA: {it['answer']}" for it in items]
    vectors = []

    print("Embedding items…")
    for i in tqdm(range(0, len(texts), 250)):
        batch = texts[i:i+250]
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )
        for r in resp.data:
            vectors.append(r.embedding)

    arr = np.array(vectors, dtype=np.float32)
    np.save(EMB_PATH, arr)
    return arr

# ----- Reduce -----

def reduce_umap(vecs):
    print("Running UMAP reduction…")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    emb2d = reducer.fit_transform(vecs)
    return emb2d

# ----- Cluster -----

def cluster_points(vecs):
    print("Clustering with HDBSCAN…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=35,
        min_samples=10,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(vecs)
    return labels

# ----- Main -----

def main():
    items = load_items()
    print(f"Loaded {len(items)} Q&A")

    # embed
        # embed or load cached
    if EMB_PATH.exists():
        print("Loading cached embeddings…")
        vecs = np.load(EMB_PATH)
    else:
        vecs = embed_items(items)

    # UMAP
    emb2d = reduce_umap(vecs)

    # Cluster
    labels = cluster_points(emb2d)

    # write clusters file
    with CLUSTER_PATH.open("w", encoding="utf-8") as f:
        for it, label in zip(items, labels):
            out = dict(it)
            out["cluster"] = int(label)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

        # summary (convert numpy.int64 keys to plain ints/strings for JSON)
    summary = {}
    for lbl in labels:
        key = int(lbl)  # ensure Python int, not numpy.int64
        summary[key] = summary.get(key, 0) + 1

    # JSON keys must be str/int; we'll use strings for readability
    summary_json = {str(k): v for k, v in summary.items()}

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # visualization
    plt.figure(figsize=(12,8))
    sc = plt.scatter(emb2d[:,0], emb2d[:,1], c=labels, s=3, cmap="Spectral")
    plt.colorbar(sc)
    plt.title("QnA v3 UMAP Clusters")
    plt.savefig(FIG_PATH, dpi=300)

    print("\nDone.")
    print(f"Wrote: {CLUSTER_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Plot: {FIG_PATH}")


if __name__ == "__main__":
    main()