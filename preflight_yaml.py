#!/usr/bin/env python3
import yaml, glob, os, sys, re

ROOT = sys.argv[1] if len(sys.argv) > 1 else "uploads"

print(f"🔍 Preflighting YAML files in: {ROOT}")

issues = []

def check_file(path):
    try:
        text = open(path, encoding="utf-8").read()
    except Exception as e:
        issues.append((path, f"read_error: {e}"))
        return

    # Try parsing
    try:
        data = yaml.safe_load(text)
    except Exception as e:
        issues.append((path, f"parse_error: {e}"))
        return

    # Must contain items or be a list
    if isinstance(data, dict):
        items = data.get("items")
        if items is None:
            issues.append((path, "missing 'items' key"))
            return
        if not isinstance(items, list):
            issues.append((path, "'items' is not a list"))
            return
    elif isinstance(data, list):
        items = data
    else:
        issues.append((path, "YAML top-level is not list or mapping"))
        return

    # Check each item
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            issues.append((path, f"item[{idx}] not a mapping"))
            continue

        # Check for tail-glue signatures
        q = str(it.get("question",""))
        a = str(it.get("answer",""))
        if any(s in q or s in a for s in (" tags:", " metadata:", " notes:")):
            issues.append((path, f"item[{idx}] has tail-glue (‘tags/metadata/notes’ inside question/answer)"))

        # Check content presence
        if not any(
            k in it for k in
            ("question","answer","title","statement","term","definitions",
             "directive","method","method_step","prompt","response",
             "primary_directive","imperatives","description")
        ):
            issues.append((path, f"item[{idx}] has no embeddable fields"))

    # Passed validation
    return


# Scan all YAML files in the upload directory
paths = sorted(glob.glob(os.path.join(ROOT, "**/*.yaml"), recursive=True) +
               glob.glob(os.path.join(ROOT, "**/*.yaml.txt"), recursive=True))

print(f"Found {len(paths)} YAML files.")
for p in paths:
    check_file(p)

if issues:
    print("\n❌ Preflight found issues:")
    for p, err in issues:
        print(f" - {p}: {err}")
    sys.exit(1)

print("\n✅ All YAML files passed preflight validation!")

