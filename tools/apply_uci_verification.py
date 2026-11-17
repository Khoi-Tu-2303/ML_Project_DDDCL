"""Record the outcome of the 2026-11-16 UCI cross-check in manifest.json and provenance/*.json.

Evidence came from ``ucimlrepo.fetch_ucirepo`` (ucimlrepo 0.0.7) plus the raw UCI archives, compared
cell-by-cell with hf_release/data/*.csv. Only datasets whose content is identical to the UCI data are
promoted to ``verified_primary_source``; the rest keep ``unverified_author_copy`` with findings noted.
Re-running is idempotent. Does not touch data CSVs or already-verified pilot datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CHECKED_ON = "2026-11-16"
UCI = "https://archive.ics.uci.edu/static/public"
LICENSE = "CC-BY-4.0"

# Datasets whose release CSV is identical (features + labels) to UCI -> promoted.
VERIFIED = {
    "breast_cancer_diagnostic": {
        "uci_id": 17, "url": f"{UCI}/17/breast+cancer+wisconsin+diagnostic.zip",
        "sha256": "bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
        "label_mapping": {"B": 0, "M": 1}, "label_meanings": {"0": "benign", "1": "malignant"},
        "evidence": "569x30 features identical in order to ucimlrepo id=17 (UCI ID column dropped); Diagnosis B/M == label 0/1 on every row.",
    },
    "liver": {
        "uci_id": 60, "url": f"{UCI}/60/liver+disorders.zip",
        "sha256": "839c8b8c3cd0383e2106afa9b97be01305a905c854813de054e87e9efaa52a1c",
        "label_mapping": {"1": 0, "2": 1},
        "label_meanings": {"0": "bupa_selector_1", "1": "bupa_selector_2"},
        "evidence": "feature_1..5 == ucimlrepo id=60 features (mcv, alkphos, sgpt, sgot, gammagt) and feature_6 == drinks, all in order; "
                    "label == bupa.data selector-1 on every row (counts 145/200). Selector is not a disease diagnosis. "
                    "UCI notes duplicate rows in this data; none removed.",
    },
    "clean1": {
        "uci_id": 74, "url": f"{UCI}/74/musk+version+1.zip",
        "sha256": None,  # hashed from --clean1-zip at runtime
        "label_mapping": {"0": 0, "1": 1}, "label_meanings": {"0": "uci_class_0_non_musk", "1": "uci_class_1_musk"},
        "evidence": "476x166 features and class identical in order to ucimlrepo id=74. UCI also has molecule_name (92 distinct) and "
                    "conformation_name columns that are NOT in this release, so no group split is possible from the release as is.",
    },
    "optical_digits": {
        "uci_id": 80, "url": f"{UCI}/80/optical+recognition+of+handwritten+digits.zip",
        "sha256": "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0",
        "label_mapping": {str(i): i for i in range(10)}, "label_meanings": {str(i): f"digit_{i}" for i in range(10)},
        "evidence": "3823x64 features and class identical, in order, to the first 3823 rows of ucimlrepo id=80 (= optdigits.tra). "
                    "The 1797-row optdigits.tes is not included.",
    },
}

# Datasets that stay unverified, with what the cross-check found.
FINDINGS = {
    "cancer": {
        "label_mapping": {"4": 0, "2": 1},
        "label_meanings": {"0": "malignant (UCI Wisconsin Original class 4)", "1": "benign (UCI Wisconsin Original class 2)"},
        "reference": {"uci_id": 15, "url": f"{UCI}/15/breast+cancer+wisconsin+original.zip",
                      "sha256": "3f91e49bceb30c0de8ea988344357236f26ed8bd536415ac594226015b6fd84d"},
        "finding": "All 457 rows occur in UCI Breast Cancer Wisconsin (Original, id=15, 699 rows / 683 complete) and labels agree on every "
                   "row (label 1 = benign/class 2: 285, label 0 = malignant/class 4: 172), the file is a sub-multiset of the complete rows "
                   "but NOT in UCI order, contains 137 duplicate feature rows, and omits 226 complete UCI rows. The selection procedure "
                   "is unknown, so it cannot be called a verified copy.",
    },
    "pima": {
        "finding": "UCI id=34 is 'Diabetes' (AIM'94 time series), NOT Pima Indians Diabetes, and ucimlrepo refuses to import it; the old "
                   "UCI Pima path returns 404. No primary UCI copy could be fetched. Content is identical (768x8 features + labels, in "
                   "order) to the third-party mirror github.com/jbrownlee/Datasets pima-indians-diabetes.data.csv "
                   "(sha256 recorded in 'mirror_sha256'), which is not a primary source.",
        "mirror_sha256": None,
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("hf_release"))
    parser.add_argument("--clean1-zip", type=Path, required=True, help="UCI musk+version+1.zip (id 74) to hash")
    parser.add_argument("--pima-mirror", type=Path, required=True, help="jbrownlee pima-indians-diabetes.data.csv to hash")
    args = parser.parse_args()
    VERIFIED["clean1"]["sha256"] = sha256_file(args.clean1_zip)
    FINDINGS["pima"]["mirror_sha256"] = sha256_file(args.pima_mirror)

    manifest_path = args.release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, v in VERIFIED.items():
        entry = manifest["datasets"][name]
        entry.update(source_url=v["url"], source_sha256=v["sha256"], status="verified_primary_source", license=LICENSE,
                     label_mapping=v["label_mapping"], label_meanings=v["label_meanings"])
        prov_path = args.release / entry["provenance_file"]
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        prov.update(status="verified_primary_source", primary_source_verified=True, license=LICENSE,
                    uci_verification={"checked_on": CHECKED_ON, "method": "ucimlrepo 0.0.7 fetch_ucirepo + raw UCI archive",
                                      "uci_id": v["uci_id"], "archive_url": v["url"], "archive_sha256": v["sha256"], "result": v["evidence"]},
                    label_mapping=v["label_mapping"], label_meanings=v["label_meanings"])
        write_json(prov_path, prov)
    for name, f in FINDINGS.items():
        entry = manifest["datasets"][name]
        prov_path = args.release / entry["provenance_file"]
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        prov["uci_verification"] = {"checked_on": CHECKED_ON, "result": "NOT VERIFIED", **{k: v for k, v in f.items() if k not in ("label_mapping", "label_meanings")}}
        prov["primary_source_verified"] = False
        if "label_meanings" in f:
            entry["label_meanings"] = prov["label_meanings"] = f["label_meanings"]
            entry["label_mapping"] = prov["label_mapping"] = f["label_mapping"]
        write_json(prov_path, prov)
    write_json(manifest_path, manifest)
    print(json.dumps({k: v["status"] for k, v in sorted(manifest["datasets"].items())}, indent=2))
    print("manifest_sha256", sha256_file(manifest_path))


if __name__ == "__main__":
    main()
