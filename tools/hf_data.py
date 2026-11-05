"""Verified HF file acquisition for canonical CSVs; imports SDK only when called."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
import re


def checked_config(path: Path) -> dict:
    """Read configuration and reject placeholder or floating revisions."""
    config = json.loads(path.read_text(encoding="utf-8"))
    repo = config.get("repo_id", "")
    if repo.count("/") != 1 or "YOUR_" in repo or " " in repo:
        raise ValueError("Set a real HF dataset repo_id in the data config")
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("revision", "")):
        raise ValueError("HF revision must be a full 40-character commit SHA, not main/latest")
    if not re.fullmatch(r"[0-9a-f]{64}", config.get("manifest_sha256", "")):
        raise ValueError("Set manifest_sha256 from the built/verified release")
    return config


def validate_csv(blob: bytes, entry: dict) -> list[dict]:
    """Validate bytes independently of HF transport and return CSV records."""
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8")))
    if reader.fieldnames != [entry["row_id_column"], *entry["feature_columns"], entry["label_column"]]:
        raise ValueError("Dataset schema/column order mismatch")
    rows = list(reader)
    ids = [row[entry["row_id_column"]] for row in rows]
    if len(rows) != entry["n_rows"] or len(set(ids)) != len(ids):
        raise ValueError("Dataset row count or unique row_id constraint failed")
    if dict(Counter(row[entry["label_column"]] for row in rows)) != entry["class_counts"]:
        raise ValueError("Dataset class counts differ from manifest")
    if not all(math.isfinite(float(row[col])) for row in rows for col in entry["feature_columns"]):
        raise ValueError("Dataset contains nonfinite features")
    labels = sorted(int(v) for v in entry["class_counts"])
    if labels != list(range(len(labels))):
        raise ValueError("Canonical labels must be contiguous integers 0..C-1")
    return rows


def verified_csv_path(config_path: Path, dataset_id: str) -> tuple[Path, dict]:
    """Download pinned manifest/CSV, verify their bytes/schema, return cache path and entry."""
    from huggingface_hub import hf_hub_download

    config = checked_config(config_path)
    kwargs = {"repo_id": config["repo_id"], "repo_type": "dataset", "revision": config["revision"],
              "cache_dir": config.get("cache_dir")}
    manifest_path = Path(hf_hub_download(filename=config.get("manifest_filename", "manifest.json"), **kwargs))
    raw_manifest = manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(raw_manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    entry = manifest["datasets"][dataset_id]
    if entry.get("status") not in {"verified_primary_source", "verified_hosted_mirror"}:
        raise ValueError("Dataset provenance not approved")
    path = Path(hf_hub_download(filename=entry["filename"], **kwargs))
    validate_csv(path.read_bytes(), entry)
    return path, entry
