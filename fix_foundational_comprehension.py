import re, yaml

SRC = "datasets/foundational_comprehension_batch_01.yaml.txt"
BAK = SRC + ".bak"

data = yaml.safe_load(open(SRC, encoding="utf-8"))
items = data.get("items", data)
if not isinstance(items, list):
    raise SystemExit("Expected a list of items in the YAML")

def split_blob(blob):
    """
    Given a big 'question' string that wrongly contains 'answer:' and 'tags:'
    split into (question_text, answer_text, rest_text).
    """
    # Normalize line breaks and collapse multiple spaces around keys
    t = blob.replace("\r","")
    # Find 'answer:' marker
    parts = t.split("answer:", 1)
    qtxt = parts[0].strip()
    atxt = ""
    rest = ""
    if len(parts) == 2:
        tail = parts[1]
        # Cut off at first of 'tags:' | 'metadata:' | 'notes:' if present
        m = re.split(r'\b(tags:|metadata:|notes:)\b', tail, maxsplit=1)
        atxt = m[0].strip()
        if len(m) > 1:
            # include the split token again
            rest = (m[1] + m[2]).strip()
    return qtxt, atxt, rest

def parse_inline_mapping(text):
    """
    Try to parse a mini YAML mapping from the tail text (e.g. 'tags: ... metadata: ...').
    Returns dict or {}.
    """
    # Ensure each key is on its own line for YAML
    text = re.sub(r'\b(tags|metadata|notes):', r'\n\1:', text)
    try:
        obj = yaml.safe_load(text) or {}
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

fixed_count = 0
for it in items:
    q = it.get("question")
    a = it.get("answer")
    if not isinstance(q, str): 
        continue
    if "answer:" in q and "tags:" in q:
        qtxt, atxt, tail = split_blob(q)
        # update question/answer
        it["question"] = qtxt
        if not it.get("answer"):  # only fill if missing
            it["answer"] = atxt

        # parse any trailing inline mappings for tags/metadata/notes
        tail_map = parse_inline_mapping(tail) if tail else {}

        # merge tags
        if "tags" in tail_map:
            existing = it.get("tags") or {}
            # if existing is a list by accident, normalize
            if isinstance(existing, list):
                existing = {"misc": existing}
            # deep-merge
            for k,v in tail_map["tags"].items():
                existing[k] = v
            it["tags"] = existing

        # merge notes
        if "notes" in tail_map and not it.get("notes"):
            it["notes"] = tail_map["notes"]

        # merge metadata
        if "metadata" in tail_map:
            existing = it.get("metadata") or {}
            for k,v in tail_map["metadata"].items():
                existing[k] = v
            it["metadata"] = existing

        fixed_count += 1

# write back safely
open(BAK, "w", encoding="utf-8").write(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
yaml.safe_dump(data, open(SRC, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print("fixed items:", fixed_count)
