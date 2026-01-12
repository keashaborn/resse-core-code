import os, glob, yaml

ROOT = "uploads"

# Collect all yaml files in uploads/
all_paths = sorted(
    glob.glob(os.path.join(ROOT, "*.yaml")) +
    glob.glob(os.path.join(ROOT, "*yaml.txt"))
)

groups = {
    "resse_foundational_v2": [],
    "resse_kernel_v2": [],
    "resse_core_v2": [],
    "resse_applied_v2": [],
    "resse_ref_psy_v2": [],
    "resse_ref_phi_v2": [],
    "resse_ref_misc_v2": [],
}

# Special foundational extras (FM commentary)
foundational_extras = {
    "core_fractal_monism_scientific_commentary.yaml",
    "core_fractal_monistic_inquiry_framework.yaml",
    "core_functional_ai_guidance.yaml",
}

for path in all_paths:
    base = os.path.basename(path)

    # 1) Foundational
    if base.startswith("foundational_") or base in foundational_extras:
        groups["resse_foundational_v2"].append(path)
        continue

    # 2) RESSE kernel (identity, archetypes, frameworks)
    if base.startswith("resse_"):
        groups["resse_kernel_v2"].append(path)
        continue

    # 3) CORE extrapolations
    if base.startswith("[core]_"):
        groups["resse_core_v2"].append(path)
        continue

    # 4) Applied / PFI
    if base.startswith("[applied]_") or base.startswith("[pfi]_"):
        groups["resse_applied_v2"].append(path)
        continue

    # 5) REF-PSY (ABA / RFT)
    if base.startswith("[ref-psy]_"):
        groups["resse_ref_psy_v2"].append(path)
        continue

    # 6) REF-PHI (comparative philosophy)
    if base.startswith("[ref-phi]_"):
        groups["resse_ref_phi_v2"].append(path)
        continue

    # 7) REF-MISC (peripheral works)
    if base.startswith("[ref-misc]_"):
        groups["resse_ref_misc_v2"].append(path)
        continue

    # Anything not caught above can be inspected later if needed.
    # For now, we just print it.
    print("⚠️ Unassigned file (not matched by any pattern):", base)

os.makedirs("datasets", exist_ok=True)

for coll, files in groups.items():
    manifest_path = f"datasets/{coll}_manifest.yaml"
    data = {
        "collection": coll,
        "files": files,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"✅ {coll}: {len(files)} files → {manifest_path}")

