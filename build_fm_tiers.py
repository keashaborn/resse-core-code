import json
import csv
from pathlib import Path

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
NTA_ROOT = Path("resse_nta_clean")
ROLES_FILE = Path("results/nta_roles.tsv")
OUTPUT_DIR = Path("fm_tiers")

OUTPUT_DIR.mkdir(exist_ok=True)

CANON_FILE = OUTPUT_DIR / "fm_canon_v1.jsonl"
CONTEXT_FILE = OUTPUT_DIR / "fm_context_v1.jsonl"
NONFM_FILE = OUTPUT_DIR / "external_context_v1.jsonl"

# ----------------------------------------------------
# Load role classifications
# ----------------------------------------------------
print("Loading roles...")

roles_by_id = {}  # id -> role

with ROLES_FILE.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        item_id = (row.get("id") or "").strip()
        role = (row.get("role") or "").strip()

        # skip malformed / header / summary rows
        if not item_id or not role:
            continue

        roles_by_id[item_id] = role

print(f"Loaded {len(roles_by_id)} classified items.")


# ----------------------------------------------------
# Open output JSONL writers
# ----------------------------------------------------
canon_f = CANON_FILE.open("w", encoding="utf-8")
context_f = CONTEXT_FILE.open("w", encoding="utf-8")
nonfm_f = NONFM_FILE.open("w", encoding="utf-8")


# ----------------------------------------------------
# Walk all NTA JSONL files
# ----------------------------------------------------
print("Scanning NTA corpus...")

count = 0
miss_role = 0

for jsonl_file in NTA_ROOT.rglob("*.nta.jsonl"):
    with jsonl_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            item_id = item.get("id")

            if not item_id:
                miss_role += 1
                continue

            role = roles_by_id.get(item_id)
            if not role:
                miss_role += 1
                continue

            if role == "FM_CANON":
                canon_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif role == "FM_CONTEXT":
                context_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                nonfm_f.write(json.dumps(item, ensure_ascii=False) + "\n")

            count += 1


canon_f.close()
context_f.close()
nonfm_f.close()

print("\n--------------------------------------")
print("DONE building FM tiers!")
print(f"Total items written: {count}")
print(f"Items missing role:  {miss_role}")
print("Output directory:", OUTPUT_DIR)
print("--------------------------------------")