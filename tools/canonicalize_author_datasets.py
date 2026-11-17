"""Convert author-repo CSVs (DDCL/RealData/ecols) to the canonical release format.

Output per dataset: ``data/<id>.csv`` = ``row_id, feature_1..feature_d, label`` plus
``provenance/<id>.json`` and a manifest entry. Raw row order is preserved; nothing is
scaled, imputed, deduplicated or resampled. Existing manifest entries are left untouched.

Entries are written with status ``unverified_author_copy``: the loader (src/dddcl/data/hf.py)
only accepts ``verified_primary_source`` / ``verified_hosted_mirror``, so these datasets stay
blocked until someone verifies them against the primary source and changes the status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

AUTHOR_REPO = "https://github.com/ECOLS-research-group/DDCL"
AUTHOR_COMMIT = "c72c34478f3cf523eb43016f19057bd7ba64c2f7"
README_MARKER = "## Author-copy datasets (unverified)"

PIMA_FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                 "insulin", "bmi", "diabetes_pedigree_function", "age"]

# source: file name inside ecols/; drop: raw column names removed (identifiers);
# features: explicit names, else feature_1..feature_d.
SPECS = {
    "breast_cancer_diagnostic": {"source": "Breast Cancer Diagnostic.csv", "drop": ["C1"],
        "note": "Raw column C1 is a patient/sample ID (569 unique values) and was dropped; 30 features remain."},
    "cancer": {"source": "Cancer.csv", "drop": [],
        "note": "Origin of this 457-row subset is NOT verified (see DATA_REGISTRY.md: BLOCKED)."},
    "clean1": {"source": "Clean1.csv", "drop": [],
        "note": "Musk v1 has molecule groups; the author CSV carries no group column, so none is available."},
    "liver": {"source": "Liver Disorder.csv", "drop": [],
        "note": "BUPA selector column recoded by the author file as class 0/1; do not read it as disease status."},
    "optical_digits": {"source": "Optical_Recognition_Handwritten_Digits.csv", "drop": [],
        "note": "3823-row training portion only; the 1797-row UCI test portion is not included."},
    "pima": {"source": "Pima Indians Diabetes.csv", "drop": [], "features": PIMA_FEATURES,
        "note": "Feature names follow the standard Pima column order (assumption). Zeros are preserved, not treated as missing."},
}


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def convert(name: str, spec: dict, source_dir: Path, release: Path) -> dict:
    """Write canonical CSV + provenance for one dataset and return its manifest entry."""
    src = source_dir / spec["source"]
    blob = src.read_bytes()
    with src.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = [h.strip() for h in next(reader)]
        raw_rows = [[f.strip() for f in row] for row in reader if row and any(f.strip() for f in row)]
    if header[-1] != "class":
        raise ValueError(f"{name}: expected last raw column 'class', got {header[-1]!r}")
    missing = [c for c in spec["drop"] if c not in header]
    if missing:
        raise ValueError(f"{name}: columns to drop not found: {missing}")
    keep = [i for i, h in enumerate(header[:-1]) if h not in spec["drop"]]
    d = len(keep)
    features = spec.get("features") or [f"feature_{i + 1}" for i in range(d)]
    if len(features) != d:
        raise ValueError(f"{name}: {len(features)} feature names for {d} columns")

    records, counts = [], Counter()
    for n, row in enumerate(raw_rows):
        if len(row) != len(header):
            raise ValueError(f"{name}: wrong column count at row {n}")
        values = [row[i] for i in keep]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"{name}: non-finite feature at row {n}")
        label = row[-1]
        if not label.isdigit():
            raise ValueError(f"{name}: non-integer label {label!r} at row {n}")
        counts[label] += 1
        records.append([f"{name}:{n:06d}", *values, label])
    classes = sorted(counts, key=int)
    if classes != [str(i) for i in range(len(classes))]:
        raise ValueError(f"{name}: labels are not 0..C-1: {classes}")
    class_counts = {c: counts[c] for c in classes}

    out = release / "data" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["row_id", *features, "label"])
        writer.writerows(records)

    source_rel = f"RealData/ecols/{spec['source']}"
    label_mapping = {c: int(c) for c in classes}
    label_meanings = {c: f"author_csv_class_{c}" for c in classes}
    provenance = {
        "source_repo": AUTHOR_REPO, "source_commit": AUTHOR_COMMIT, "source_file": source_rel,
        "source_file_sha256": sha256_bytes(blob), "raw_columns": len(header),
        "dropped_columns": spec["drop"], "features": features, "n_rows": len(records),
        "class_counts": class_counts, "label_mapping": label_mapping, "label_meanings": label_meanings,
        "note": spec["note"], "license": "unspecified", "status": "unverified_author_copy",
        "transforms": ["strip CSV whitespace", "drop identifier columns" if spec["drop"] else "no column dropped",
                       "rename class->label (values unchanged)", "stable row_id", "generic feature names"
                       if "features" not in spec else "semantic feature names"],
        "no_scaling_no_splitting_no_resampling": True,
        "primary_source_verified": False,
    }
    provenance_file = f"provenance/{name}.json"
    write_json(release / provenance_file, provenance)
    return {
        "filename": f"data/{name}.csv", "sha256": sha256_bytes(out.read_bytes()), "n_rows": len(records),
        "feature_columns": features, "label_column": "label", "row_id_column": "row_id",
        "class_counts": class_counts, "label_mapping": label_mapping, "label_meanings": label_meanings,
        "source_url": f"{AUTHOR_REPO}/blob/{AUTHOR_COMMIT}/{source_rel.replace(' ', '%20')}",
        "source_sha256": sha256_bytes(blob), "provenance_file": provenance_file,
        "license": "unspecified", "status": "unverified_author_copy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("../DDCL/RealData/ecols"))
    parser.add_argument("--release", type=Path, default=Path("hf_release"))
    args = parser.parse_args()

    manifest_path = args.release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {}
    for name, spec in SPECS.items():
        existing = manifest["datasets"].get(name)
        if existing and existing.get("status") in {"verified_primary_source", "verified_hosted_mirror"}:
            raise ValueError(f"{name}: already approved in manifest; refusing to overwrite")
        entries[name] = convert(name, spec, args.source_dir, args.release)
    manifest["datasets"].update(entries)
    write_json(manifest_path, manifest)

    readme = args.release / "README.md"
    text = readme.read_text(encoding="utf-8")
    if README_MARKER not in text:
        text += (f"\n{README_MARKER}\n\n"
                 f"The datasets {', '.join(SPECS)} are canonicalised copies of CSV files from {AUTHOR_REPO} "
                 f"(commit {AUTHOR_COMMIT}). They are NOT verified against the primary UCI sources "
                 "(status `unverified_author_copy`); see provenance/*.json for per-dataset notes.\n")
        readme.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({k: {"rows": v["n_rows"], "features": len(v["feature_columns"]), "classes": len(v["class_counts"])}
                      for k, v in entries.items()}, indent=2))
    print("manifest_sha256", sha256_bytes(manifest_path.read_bytes()))


if __name__ == "__main__":
    main()
