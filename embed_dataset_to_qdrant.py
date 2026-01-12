#!/usr/bin/env python3
import os, sys, uuid, yaml, requests
from typing import Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI, PermissionDeniedError

# --- config ---
QDRANT_URL  = "http://localhost:6333"
EMBED_MODEL = "text-embedding-3-large"

client = OpenAI()

# ---------- content normalization ----------

from typing import Dict, Any, Optional  # make sure this import exists near the top

def build_content(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Normalize many schema variants to a unified {title, body} dict for embedding.
    Returns {"title": str, "body": str} or None if nothing embeddable is found.
    """

    def _to_text(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, list):
            return "\n- " + "\n- ".join(str(x).strip() for x in v if str(x).strip())
        if isinstance(v, dict):
            return "\n" + "\n".join(f"{k}: {str(v[k]).strip()}" for k in v if str(v[k]).strip())
        return str(v).strip()

    # 1) Classic Q/A
    q = str(item.get("question", "") or "").strip()
    a = str(item.get("answer", "") or "").strip()
    if q or a:
        body = f"Q: {q}\nA: {a}".strip()
        if body:
            return {"title": "", "body": body}

    # 2) Title / Statement
    title = str(item.get("title", "") or "").strip()
    stmt  = str(item.get("statement", "") or "").strip()
    if title or stmt:
        body = (title + "\n\n" + stmt).strip()
        if body:
            return {"title": title, "body": body}

    # 3) Term / Definitions
    term = str(item.get("term", "") or "").strip()
    defs = item.get("definitions")
    if term or defs:
        if isinstance(defs, list):
            defs_txt = "\n- " + "\n- ".join(str(d).strip() for d in defs if str(d).strip())
        elif isinstance(defs, str):
            defs_txt = "\n" + defs.strip()
        else:
            defs_txt = ""
        body = (term + defs_txt).strip()
        if body:
            return {"title": term, "body": body}

    # 4) Directive / text / rule / method / body / content
    for k in ("directive", "text", "rule", "guideline", "guidelines",
              "method", "method_step", "body", "content"):
        if k in item and item[k]:
            v_txt = _to_text(item[k])
            if v_txt:
                name = str(item.get("name", "") or item.get("label", "") or "").strip()
                return {"title": name, "body": v_txt}

    # 5) Prompt / Response (explicit prompt/response keys)
    p = str(item.get("prompt", "") or "").strip()
    r2 = str(item.get("response", "") or "").strip()
    if p or r2:
        body = (f"Prompt: {p}\nResponse: {r2}").strip()
        if body:
            return {"title": "", "body": body}

    # 6) Persona / Archetype definitions (type: persona/archetype/identity)
    if item.get("type") in ("persona", "archetype", "identity"):
        name = _to_text(item.get("name"))
        desc = _to_text(item.get("description"))
        pd   = item.get("primary_directive")
        imps = item.get("imperatives")
        ex   = item.get("example_prompt")

        parts = []
        if desc:
            parts.append("Description:\n" + desc)
        if pd:
            parts.append("Primary Directive:\n" + _to_text(pd))
        if imps:
            parts.append("Imperatives:\n" + _to_text(imps))
        if ex:
            parts.append("Example Prompt:\n" + _to_text(ex))

        body = "\n\n".join(p for p in parts if p.strip()).strip()
        if body:
            return {"title": name, "body": body}

    # 7) Stimulus / Response examples (e.g., Michael, Sage entries)
    stim  = item.get("stimulus")
    resp2 = item.get("response")
    ctx   = item.get("context")
    style = item.get("style_reference")

    if stim or resp2:
        parts = []
        if ctx:
            parts.append("Context:\n" + _to_text(ctx))
        if stim:
            parts.append("Stimulus:\n" + _to_text(stim))
        if resp2:
            parts.append("Response:\n" + _to_text(resp2))

        body = "\n\n".join(p for p in parts if p.strip()).strip()
        if body:
            title2 = _to_text(style)
            return {"title": title2, "body": body}

    # Fallback: nothing embeddable found
    return None

def validate_item(it: Dict[str, Any], path: str, idx: int) -> None:
    """
    Lightweight schema guard:
    - item must resolve to embeddable content via build_content
    - tags/metadata, if present, must be mappings
    - block common tail-glue signatures in Q/A strings
    """
    content = build_content(it)
    if not content:
        keys = ", ".join(sorted(it.keys()))
        raise ValueError(f"{path}[{idx}]: no embeddable content (keys: {keys})")

    if "tags" in it and not isinstance(it["tags"], dict):
        raise ValueError(f"{path}[{idx}]: tags must be a mapping")
    if "metadata" in it and not isinstance(it["metadata"], dict):
        raise ValueError(f"{path}[{idx}]: metadata must be a mapping")

    q = str(it.get("question","") or "")
    a = str(it.get("answer","")   or "")
    if any(sig in q or sig in a for sig in (" tags:", " metadata:", " notes:")):
        raise ValueError(f"{path}[{idx}]: detected inline 'tags/metadata/notes' inside Q/A text; fix source YAML")



# ---------- qdrant helpers ----------

def ensure_collection(collection: str, dim: int = 3072):
    """Create collection if it doesn't exist (cosine distance)."""
    requests.put(
        f"{QDRANT_URL}/collections/{collection}",
        json={
            "vectors": {"size": dim, "distance": "Cosine"},
            "optimizers_config": {"default_segment_number": 2}
        }
    )

def stable_id(source_file: str, idx: int, item: Dict[str, Any]) -> str:
    """
    Deterministic UUIDv5 so re-embeds update instead of duplicating.
    Namespace by source_file; name by (index | item_id/content).
    """
    item_id = str(item.get("id") or item.get("item_id") or "").strip()
    if not item_id:
        q  = str(item.get("question","")  or "")
        a  = str(item.get("answer","")    or "")
        t  = str(item.get("title","")     or "")
        st = str(item.get("statement","") or "")
        content_key = (q + "\n" + a + "\n" + t + "\n" + st)[:512]
    else:
        content_key = item_id
    ns = uuid.uuid5(uuid.NAMESPACE_URL, source_file)      # stable per file
    return str(uuid.uuid5(ns, f"{idx}|{content_key}"))     # stable per item

def embed(txt: str):
    """
    Call OpenAI embeddings once, return vector.
    If the API denies the request (403), log and skip this item.
    """
    try:
        r = client.embeddings.create(input=[txt], model=EMBED_MODEL)
        return r.data[0].embedding
    except PermissionDeniedError as e:
        print(f"⚠️  PermissionDeniedError while embedding, skipping this item: {e}")
        return None

# ---------- main ----------

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 embed_dataset_to_qdrant.py <yaml_path> <collection_name>")
        sys.exit(1)

    yaml_path  = sys.argv[1]
    collection = sys.argv[2]

    # Skip backups / non-sources
    if yaml_path.endswith((".bak",".fixed",".rtf",".rtf.txt",".zip")):
        print(f"⏩ Skipping backup/non-source file: {yaml_path}")
        sys.exit(0)

    ensure_collection(collection, dim=3072)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        print("No items found in YAML (expected a list or dict with 'items').")
        sys.exit(1)

    points = []
    for i, item in enumerate(tqdm(items, desc=os.path.basename(yaml_path))):
        # validate + normalize
        validate_item(item, yaml_path, i)
        content_dict = build_content(item)
        body = content_dict["body"].strip()
        if not body:
            continue

        vec = embed(body)
        if vec is None:
            # skip items the API refused to embed
            continue


        # payload
        payload: Dict[str, Any] = {
            "item_id": str(item.get("id") or item.get("item_id") or "").strip(),
            "source_file": yaml_path,
            "index": i,
        }
        # optional fields
        for k in (
            "question","answer","title","statement","term","definitions",
            "directive","text","rule","guideline","guidelines","method","method_step",
            "prompt","response","tags","metadata","status","evidence_count","notes",
            "name","label","type"
        ):
            if k in item:
                payload[k] = item[k]

        pid = stable_id(yaml_path, i, item)
        points.append({"id": pid, "vector": vec, "payload": payload})

        if len(points) >= 200:
            resp = requests.put(
                f"{QDRANT_URL}/collections/{collection}/points?wait=true",
                json={"points": points}
            )
            try:
                j = resp.json()
            except Exception:
                j = {"status": f"http {resp.status_code}", "text": resp.text[:300]}
            print("upload:", j.get("status"), j.get("result", j))
            points = []

    if points:
        resp = requests.put(
            f"{QDRANT_URL}/collections/{collection}/points?wait=true",
            json={"points": points}
        )
        try:
            j = resp.json()
        except Exception:
            j = {"status": f"http {resp.status_code}", "text": resp.text[:300]}
        print("upload:", j.get("status"), j.get("result", j))

    print(f"✅ Ingest complete → collection: {collection}")

if __name__ == "__main__":
    main()
