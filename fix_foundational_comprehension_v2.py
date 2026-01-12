import re, yaml

SRC = "datasets/foundational_comprehension_batch_01.yaml.txt"
BAK = SRC + ".bak"

data = yaml.safe_load(open(SRC, encoding="utf-8"))
items = data.get("items", data)
if not isinstance(items, list):
    raise SystemExit("Expected a list of items in the YAML")

def parse_tail_map(tail):
    """Parse tail text like 'tags: ... metadata: ...' to a dict; tolerate one-liners."""
    if not tail.strip():
        return {}
    text = tail
    # put each key on its own line for YAML to be lenient
    text = re.sub(r'\b(tags|metadata|notes):', r'\n\1:', text)
    try:
        obj = yaml.safe_load(text) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def split_from_markers(text):
    """
    Given a text (question or answer), split at the first of tags:/metadata:/notes:
    Returns (before, tail_text)
    """
    m = re.split(r'\b(tags:|metadata:|notes:)\b', text, maxsplit=1)
    if len(m) > 1:
        before = m[0].strip()
        tail = (m[1] + m[2]).strip()
        return before, tail
    return text.strip(), ""

fixed = 0
for it in items:
    # Case A: dirty QUESTION (already handled previously)
    q = str(it.get("question",""))
    if "answer:" in q and "tags:" in q:
        parts = q.split("answer:", 1)
        it["question"] = parts[0].strip()
        ans_dirty = parts[1]
        ans_clean, tail = split_from_markers(ans_dirty)
        if not it.get("answer"):
            it["answer"] = ans_clean
        tail_map = parse_tail_map(tail)
        # merge tail_map
        if "tags" in tail_map:
            existing = it.get("tags") or {}
            if isinstance(existing, list):
                existing = {"misc": existing}
            for k,v in tail_map["tags"].items():
                existing[k] = v
            it["tags"] = existing
        if "notes" in tail_map and not it.get("notes"):
            it["notes"] = tail_map["notes"]
        if "metadata" in tail_map:
            meta = it.get("metadata") or {}
            for k,v in tail_map["metadata"].items():
                meta[k] = v
            it["metadata"] = meta
        fixed += 1

    # Case B: clean QUESTION but dirty ANSWER (answer contains 'tags:' or 'metadata:')
    a = str(it.get("answer",""))
    if ("tags:" in a or "metadata:" in a or "notes:" in a):
        ans_clean, tail = split_from_markers(a)
        it["answer"] = ans_clean
        tail_map = parse_tail_map(tail)
        if "tags" in tail_map:
            existing = it.get("tags") or {}
            if isinstance(existing, list):
                existing = {"misc": existing}
            for k,v in tail_map["tags"].items():
                existing[k] = v
            it["tags"] = existing
        if "notes" in tail_map and not it.get("notes"):
            it["notes"] = tail_map["notes"]
        if "metadata" in tail_map:
            meta = it.get("metadata") or {}
            for k,v in tail_map["metadata"].items():
                meta[k] = v
            it["metadata"] = meta
        fixed += 1

# write back with backup
open(BAK, "w", encoding="utf-8").write(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
yaml.safe_dump(data, open(SRC, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print("fixed items:", fixed)
