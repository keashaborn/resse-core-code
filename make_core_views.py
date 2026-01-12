#!/usr/bin/env python3
"""
make_core_views.py

Create conceptual views of the corpus:

- primary_core/ : copies of the 5 foundational core NTA files
- secondary_core/ : copies of all core/** NTA files

These are views only; the original foundational/ and core/ trees remain intact.
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).parent
NTA = ROOT / "resse_nta_clean"

FOUNDATIONAL = NTA / "foundational"
PRIMARY_CORE = NTA / "primary_core"

CORE = NTA / "core"
SECONDARY_CORE = NTA / "secondary_core"


def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        print(f"SKIP (missing): {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def build_primary_core():
    print("\nBuilding primary_core view...")

    # These are the canonical foundational NTA files
    names = [
        "foundational_core.nta.json",
        "foundational_clarifications.nta.json",
        "foundational_clarifications_extended.nta.json",
        "foundational_comprehension.nta.json",
        "foundational_expansion.nta.json",
    ]

    for name in names:
        src = FOUNDATIONAL / name
        dst = PRIMARY_CORE / name
        copy_if_exists(src, dst)


def build_secondary_core():
    print("\nBuilding secondary_core view...")

    if not CORE.exists():
        print(f"SKIP: core dir does not exist at {CORE}")
        return

    # Copy all core/**.nta.json into secondary_core/**.nta.json
    for src in CORE.rglob("*.nta.json"):
        rel = src.relative_to(CORE)  # e.g. ai/[core]_ai_fractal...
        dst = SECONDARY_CORE / rel
        copy_if_exists(src, dst)


def main():
    build_primary_core()
    build_secondary_core()
    print("\nDone building primary_core and secondary_core views.")


if __name__ == "__main__":
    main()
