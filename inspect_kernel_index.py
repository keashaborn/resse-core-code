"""
inspect_kernel_index.py

Builds a human-readable index of the kernel datasets, grouped by subfolder.

This does NOT produce training data and does NOT modify any files.
It just prints an organized list so Eric can see what's in kernel at a glance.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "datasets" / "resse_json" / "kernel"

def main():
    if not KERNEL_DIR.is_dir():
        print(f"[!] Kernel directory not found at {KERNEL_DIR}")
        return

    print(f"[+] Kernel index for {KERNEL_DIR}\n")

    # Top-level subfolders like: d_identity, imp_archetype, imp_reasoning, qa_*, ps_sarcasm, sr_archetype, etc.
    for subdir in sorted(KERNEL_DIR.iterdir()):
        if not subdir.is_dir():
            continue

        print(f"{subdir.name}:")
        json_files = sorted(subdir.glob("*.json"))
        if not json_files:
            print("  (no JSON files)")
            print()
            continue

        for jf in json_files:
            print(f"  - {jf.name}")
        print()

if __name__ == "__main__":
    main()
