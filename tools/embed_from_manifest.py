#!/usr/bin/env python3
import yaml, os, subprocess, sys

if len(sys.argv) < 2:
    print("Usage: embed_from_manifest.py <manifest.yaml>")
    sys.exit(1)

manifest = sys.argv[1]
data = yaml.safe_load(open(manifest, "r"))
collection = data.get("collection", "unknown_collection")
files = data.get("files", [])

print(f"=== Embedding manifest → {collection} ===")
failures = 0
for path in files:
    path = path.strip()
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        failures += 1
        continue
    print(f"Embedding → {path}")
    try:
        subprocess.run(
            ["python3", "embed_dataset_to_qdrant.py", path, collection],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {path} → {e}")
        failures += 1
        continue

print(f"\n✅ Finished embedding all listed files → {collection} (failures: {failures})")
