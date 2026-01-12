import sys, yaml, glob, os

SKIP_SUFFIXES = (".bak", ".rtf", ".rtf.txt", ".zip")

def tag_item(it, authority=None, scale=None):
    it.setdefault("tags", {})
    it.setdefault("metadata", {})
    if authority:
        auth = it["tags"].get("authority", [])
        if authority not in auth:
            it["tags"]["authority"] = auth + [authority]
    if scale and not it["metadata"].get("scale"):
        it["metadata"]["scale"] = scale
    return it

def process_file(path, authority=None, scale=None):
    # skip backups / non-sources
    if path.endswith(SKIP_SUFFIXES):
        print("skip (backup or non-source):", path)
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print("skip (parse error):", path, "→", e)
        return

    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        print("skip (no items):", path)
        return

    for i, it in enumerate(items):
        items[i] = tag_item(dict(it), authority=authority, scale=scale)

    out = {"items": items} if isinstance(data, dict) and "items" in data else items
    bak = path + ".bak"
    os.replace(path, bak)  # backup original
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(out, f, allow_unicode=True)
    print("tagged:", path, "→", authority, scale, "(backup:", bak, ")")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 tag_yaml_bulk.py <glob> <authority> <scale>")
        sys.exit(1)
    pattern, authority, scale = sys.argv[1], sys.argv[2], sys.argv[3]
    for path in glob.glob(pattern):
        process_file(path, authority=authority, scale=scale)
