#!/usr/bin/env python3
"""
review_server.py

Tiny FastAPI backend to review generated Q&A.

- Reads from: results/generated_qas_all.jsonl
- Writes reviews to: results/generated_qas_reviews.jsonl

Endpoints:
- GET  /api/next?bucket=<FM_CANON|FM_CONTEXT|NON_FM|ANY>&status=unreviewed
- POST /api/review/{item_id}
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
DATA_PATH = RESULTS_DIR / "generated_qna_all_v3.jsonl"
REVIEWS_PATH = RESULTS_DIR / "generated_qas_reviews.jsonl"

app = FastAPI(title="RESSE Q&A Curator")

# ---- Data structures in memory ----
items: List[Dict[str, Any]] = []
reviews: Dict[str, Dict[str, Any]] = {}


class ReviewPayload(BaseModel):
    status: str  # "canon", "context", "practice", "non_fm", "reject"
    comment: Optional[str] = None


def load_items():
    global items
    if not DATA_PATH.exists():
        raise RuntimeError(f"DATA file not found: {DATA_PATH}")
    items = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    print(f"Loaded {len(items)} items from {DATA_PATH}")


def load_reviews():
    global reviews
    reviews = {}
    if not REVIEWS_PATH.exists():
        return
    with REVIEWS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj.get("id")
            if rid:
                reviews[rid] = obj


def save_review(review_obj: Dict[str, Any]):
    """Append a single review line to the reviews file."""
    with REVIEWS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(review_obj, ensure_ascii=False) + "\n")


@app.on_event("startup")
def startup_event():
    load_items()
    load_reviews()
    print(f"Loaded {len(reviews)} existing reviews from {REVIEWS_PATH}")


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the front-end HTML."""
    html_path = ROOT / "review_ui" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/next")
def get_next(
    bucket: str = "ANY",
    status: str = "unreviewed",
    fit_min: float = 0.0,
    fit_max: float = 1.0
):
    """
    Get the next item that:
    - matches bucket filter (or ANY)
    - matches fit-score range
    - has no review yet if status='unreviewed'
    """

    bucket = bucket.upper()

    for obj in items:
        item_id = obj.get("id")
        if not item_id:
            continue

        # Skip already-reviewed
        rv = reviews.get(item_id)
        if status == "unreviewed" and rv is not None:
            continue

        # Bucket filter
        item_bucket = (obj.get("tags", {}) or {}).get("bucket", "UNKNOWN")
        if bucket != "ANY" and item_bucket != bucket:
            continue

        # Extract fit score
        score = None

        # 1) From metadata
        meta = obj.get("metadata") or {}
        if "fit_score" in meta:
            score = float(meta["fit_score"])

        # 2) From notes (fallback)
        if score is None:
            notes = obj.get("notes") or []
            for n in notes:
                if "sim_to_canon_centroid" in n:
                    try:
                        score = float(n.split("=")[1])
                    except:
                        pass

        # If score still missing → skip (or treat as 0)
        if score is None:
            continue

        # Fit-score filter
        if not (fit_min <= score <= fit_max):
            continue

        # Return item + existing review
        return {"item": obj, "review": rv}

    raise HTTPException(status_code=404, detail="No matching items left.")


@app.post("/api/review/{item_id}")
def post_review(item_id: str, payload: ReviewPayload):
    """
    Save/update review for item_id.
    status: canon | context | practice | non_fm | reject
    """

    valid_status = {"canon", "context", "practice", "non_fm", "reject"}
    if payload.status not in valid_status:
        raise HTTPException(status_code=400, detail=f"Invalid status: {payload.status}")

    status = payload.status
    if status == "canon":
        review_bucket = "FM_CANON"
    elif status == "context":
        review_bucket = "FM_CONTEXT"
    elif status == "practice":
        review_bucket = "FM_PRACTICE"
    elif status == "non_fm":
        review_bucket = "NON_FM"
    else:  # "reject"
        review_bucket = "REJECT"

    # Make sure item exists
    target = None
    for obj in items:
        if obj.get("id") == item_id:
            target = obj
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Item not found")

    review_obj = {
        "id": item_id,
        "status": status,
        "review_bucket": review_bucket,
        "comment": payload.comment,
        "original_bucket": (target.get("tags", {}) or {}).get("bucket", "UNKNOWN"),
    }

    reviews[item_id] = review_obj
    save_review(review_obj)

    return {"ok": True, "review": review_obj}

from collections import defaultdict

@app.get("/api/stats")
def get_stats():
    """
    Return simple stats for the UI.
    """
    # Count original buckets
    original_counts = defaultdict(int)
    for obj in items:
        bucket = (obj.get("tags", {}) or {}).get("bucket", "UNKNOWN")
        original_counts[bucket] += 1

    # Count reviewed buckets
    review_counts = defaultdict(int)
    for rv in reviews.values():
        rb = rv.get("review_bucket", "UNLABELED")
        review_counts[rb] += 1

    return {
        "total_items": len(items),
        "reviewed_items": len(reviews),
        "original_buckets": original_counts,
        "review_buckets": review_counts,
    }