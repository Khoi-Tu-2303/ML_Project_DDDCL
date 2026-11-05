"""Build an unscaled, unsplit pilot HF dataset release from pinned UCI archives.

This is a one-time data publication utility, not the experimental data loader.
Use only after checking whether an existing public HF/Kaggle copy is suitable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

SPECS = {
    "haberman": {
        "url": "https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip",
        "page": "https://archive.ics.uci.edu/dataset/43/haberman+s+survival",
        "archive_sha256": "1001bac29022350e279b064003c426c8ed60f4d76ab4916fb05baf886c8e3ac7",
        "member": "haberman.data", "n_rows": 306, "label_index": 3,
        "features": ["age", "operation_year", "positive_axillary_nodes"],
        "label_mapping": {"1": 0, "2": 1}, "class_counts": {"0": 225, "1": 81},
        "label_meanings": {"0": "survived_5_years_or_longer", "1": "died_within_5_years"},
    },
    "new_thyroid": {
        "url": "https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip",
        "page": "https://archive.ics.uci.edu/dataset/102/thyroid+disease",
        "archive_sha256": "a0982569a7442c03a20815db58f271245e7a111b10ac46f6c6b5fa6feee4c1f4",
        "member": "new-thyroid.data", "n_rows": 215, "label_index": 0,
        "features": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        "label_mapping": {"1": 0, "2": 1, "3": 2},
        "class_counts": {"0": 150, "1": 35, "2": 30},
        "label_meanings": {"0": "raw_class_1", "1": "raw_class_2", "2": "raw_class_3"},
    },
    "banknote": {
        "url": "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        "page": "https://archive.ics.uci.edu/dataset/267/banknote+authentication",
        "archive_sha256": "1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227",
        "member": "data_banknote_authentication.txt", "n_rows": 1372, "label_index": 4,
        "features": ["variance", "skewness", "curtosis", "entropy"],
        "label_mapping": {"0": 0, "1": 1}, "class_counts": {"0": 762, "1": 610},
        "label_meanings": {"0": "raw_class_0", "1": "raw_class_1"},
    },
}


def sha256_bytes(blob: bytes) -> str:
    """Return SHA256 hexadecimal digest for bytes."""
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    """Write stable UTF-8 JSON; callers provide a staged destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def parse_records(name: str, blob: bytes, spec: dict) -> list[list[str]]:
    """Validate raw CSV records and return canonical, unscaled rows with IDs."""
    records = []
    counts: Counter[str] = Counter()
    for raw in csv.reader(io.StringIO(blob.decode("utf-8-sig"))):
        if not raw or all(not field.strip() for field in raw):
            continue
        fields = [field.strip() for field in raw]
        if len(fields) != len(spec["features"]) + 1:
            raise ValueError(f"{name}: wrong raw column count at row {len(records)}")
        raw_label = fields.pop(spec["label_index"])
        if raw_label not in spec["label_mapping"]:
            raise ValueError(f"{name}: unexpected label {raw_label!r}")
        if not all(math.isfinite(float(value)) for value in fields):
            raise ValueError(f"{name}: nonfinite feature")
        label = str(spec["label_mapping"][raw_label])
        counts[label] += 1
        records.append([f"{name}:{len(records):06d}", *fields, label])
    if len(records) != spec["n_rows"] or dict(counts) != spec["class_counts"]:
        raise ValueError(f"{name}: wrong row/class counts: {len(records)}, {dict(counts)}")
    return records


def get_archive(name: str, spec: dict, raw_dir: Path) -> bytes:
    """Download/cache a pinned archive, rejecting changed or corrupted bytes."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}.zip"
    if path.exists():
        blob = path.read_bytes()
    else:
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            blob = response.read(20_000_001)
        if len(blob) > 20_000_000:
            raise ValueError(f"{name}: archive unexpectedly large")
    if sha256_bytes(blob) != spec["archive_sha256"]:
        raise ValueError(f"{name}: archive SHA256 differs; investigate source/version, do not silently accept")
    if not path.exists():
        temporary = path.with_suffix(".zip.part")
        temporary.write_bytes(blob)
        temporary.replace(path)
    return blob


def validate_release(folder: Path) -> dict:
    """Check local manifest, canonical file hashes and basic schema/counts."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("datasets"):
        raise ValueError("Invalid release manifest")
    for name, entry in manifest["datasets"].items():
        file = folder / entry["filename"]
        if sha256_bytes(file.read_bytes()) != entry["sha256"]:
            raise ValueError(f"{name}: file SHA mismatch")
        with file.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            expected = [entry["row_id_column"], *entry["feature_columns"], entry["label_column"]]
            if reader.fieldnames != expected:
                raise ValueError(f"{name}: schema/order mismatch")
            rows = list(reader)
        ids = [row[entry["row_id_column"]] for row in rows]
        if len(rows) != entry["n_rows"] or len(ids) != len(set(ids)):
            raise ValueError(f"{name}: row count or IDs invalid")
        if dict(Counter(row[entry["label_column"]] for row in rows)) != entry["class_counts"]:
            raise ValueError(f"{name}: class counts invalid")
        if not all(math.isfinite(float(row[col])) for row in rows for col in entry["feature_columns"]):
            raise ValueError(f"{name}: nonfinite feature")
    return manifest


def build_release(out: Path, raw_dir: Path, overwrite: bool = False) -> dict:
    """Build staged files, preserve raw order and publish the local release directory."""
    if out.exists() and not overwrite:
        manifest = validate_release(out)
        if set(manifest["datasets"]) != set(SPECS):
            raise ValueError("Existing release uses different dataset set; choose a new output directory")
        return manifest
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ddcl-build-", dir=out.parent) as temporary:
        stage = Path(temporary) / "release"
        (stage / "data").mkdir(parents=True)
        entries = {}
        for name, spec in SPECS.items():
            archive = get_archive(name, spec, raw_dir)
            with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
                member = zipped.read(spec["member"])
            rows = parse_records(name, member, spec)
            file = stage / "data" / f"{name}.csv"
            with file.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["row_id", *spec["features"], "label"])
                writer.writerows(rows)
            provenance = {**spec, "raw_member_sha256": sha256_bytes(member), "license": "CC-BY-4.0",
                          "transforms": ["strip CSV whitespace", "stable row_id", "explicit integer label map"],
                          "no_scaling_no_splitting_no_resampling": True}
            provenance_file = f"provenance/{name}.json"
            write_json(stage / provenance_file, provenance)
            entries[name] = {
                "filename": f"data/{name}.csv", "sha256": sha256_bytes(file.read_bytes()),
                "n_rows": len(rows), "feature_columns": spec["features"], "label_column": "label",
                "row_id_column": "row_id", "class_counts": spec["class_counts"],
                "label_mapping": spec["label_mapping"], "label_meanings": spec["label_meanings"],
                "source_url": spec["url"], "source_sha256": spec["archive_sha256"],
                "provenance_file": provenance_file, "license": "CC-BY-4.0", "status": "verified_primary_source",
            }
        manifest = {"schema_version": 1, "release_name": "dynamic-ddcl-pilot-v1", "datasets": entries}
        write_json(stage / "manifest.json", manifest)
        card = """---
license: cc-by-4.0
task_categories:
- tabular-classification
language:
- en
---
# Dynamic DDCL pilot data

Independent canonical preparation of UCI Haberman, New-Thyroid and Banknote.
Not a claim of identical author CSV bytes or official author splits.
See manifest.json and provenance/*.json for sources, archive hashes, column order,
label mappings and UCI attribution. Raw file order is preserved, no scaling,
imputation, resampling, splitting or training has been performed.

## Attribution

- Haberman, S. (1976). Haberman's Survival. UCI. https://doi.org/10.24432/C5XK51
- Thyroid Disease collection, UCI (new-thyroid.data only). https://archive.ics.uci.edu/dataset/102/thyroid+disease
- Banknote Authentication, UCI. https://archive.ics.uci.edu/dataset/267/banknote+authentication

Dataset licenses are attributed to the primary UCI sources.
row_id is metadata and must never be included in model inputs.

## Usage

Use huggingface_hub.hf_hub_download with repo_type='dataset' and a full commit
revision. Check manifest and file SHA256 before reading CSV. Split before
fitting scalers, curriculum statistics or SMOTE. Training runs locally.
"""
        (stage / "README.md").write_text(card, encoding="utf-8", newline="\n")
        validate_release(stage)
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(stage), str(out))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf_release"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/bootstrap"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_release(args.out, args.raw_dir, args.overwrite)
    print(json.dumps({"release": str(args.out), "datasets": {k: v["n_rows"] for k, v in manifest["datasets"].items()},
                      "manifest_sha256": sha256_bytes((args.out / "manifest.json").read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
