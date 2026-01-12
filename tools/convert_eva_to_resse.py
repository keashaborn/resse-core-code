import re, yaml, glob, os, sys, shutil

# Toggle pronoun neutralization here (False keeps pronouns unchanged)
NEUTRALIZE_PRONOUNS = False   # set to True if you want 'she/her' → 'RESSE' (safer than 'they')

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
BACKUP_DIR = os.path.join(SRC_DIR, "_backup_eva_to_resse")
os.makedirs(BACKUP_DIR, exist_ok=True)

# Which files to consider
candidates = set(glob.glob(os.path.join(SRC_DIR, "eva_*")))
for p in glob.glob(os.path.join(SRC_DIR, "*.yaml*")):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            if re.search(r'\b[Ee]va\b', f.read()):
                candidates.add(p)
    except:
        pass

def convert_text(txt):
    # 1) Handle "Eva RESSE" / "Eva Resse" → "RESSE"
    txt = re.sub(r'\b[Ee]va\s+[Rr]esse\b', "RESSE", txt)

    # 2) Possessive Eva’s/Eva's → RESSE’s/RESSE's
    txt = re.sub(r"\b[Ee]va[’']s\b", "RESSE’s", txt)

    # 3) Standalone Eva → RESSE (word-boundary so 'evaluate' is safe)
    txt = re.sub(r'\b[Ee]va\b', "RESSE", txt)

    # 4) Collapse accidental doubles: "RESSE  RESSE" → "RESSE"
    txt = re.sub(r'\bRESSE\s+RESSE\b', "RESSE", txt)

    # 5) Optional: neutralize pronouns near old Eva mentions (best effort, conservative)
    if NEUTRALIZE_PRONOUNS:
        # Replace 'She/Her' when referring to the agent; keep case where possible
        txt = re.sub(r'\b[Ss]he\b', "RESSE", txt)
        txt = re.sub(r'\b[Hh]er\b', "RESSE", txt)
        txt = re.sub(r'\b[Hh]ers\b', "RESSE’s", txt)

    return txt

def safe_write(path, new_txt):
    # ensure YAML parses
    try:
        obj = yaml.safe_load(new_txt)
    except Exception as e:
        raise RuntimeError(f"YAML parse error: {e}")

    # write with backup
    base = os.path.basename(path)
    shutil.copy2(path, os.path.join(BACKUP_DIR, base))
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_txt)

def main():
    changed = 0
    renamed = 0
    for p in sorted(candidates):
        # read
        try:
            old = open(p, "r", encoding="utf-8", errors="ignore").read()
        except:
            continue

        new = convert_text(old)
        if new == old:
            continue

        # If filename starts with eva_, rename to resse_*
        dirname, fname = os.path.dirname(p), os.path.basename(p)
        target_path = p
        if fname.startswith("eva_"):
            new_name = "resse_" + fname[len("eva_"):]
            target_path = os.path.join(dirname, new_name)
            # move original to target name before write (so downstream globs match)
            os.rename(p, target_path)
            p = target_path
            renamed += 1

        try:
            safe_write(p, new)
            changed += 1
            print("✓ updated:", os.path.basename(p))
        except Exception as e:
            print("✗ skipped (parse failure):", os.path.basename(p), "→", e)

    print(f"\nDone. Changed files: {changed}, renamed: {renamed}, backups in {BACKUP_DIR}")

if __name__ == "__main__":
    main()
