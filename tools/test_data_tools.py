"""Standard-library tests for the shipped data helpers, not future training code."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from hf_data import checked_config, validate_csv, verified_csv_path
from prepare_pilot_release import parse_records, write_json


def fixture() -> tuple[bytes, dict]:
    """Return independent tiny CSV and explicit schema/count expectations."""
    blob = b"row_id,x,label\na,1.5,0\nb,3.0,1\n"
    entry = {"filename": "data/tiny.csv", "sha256": hashlib.sha256(blob).hexdigest(),
             "row_id_column": "row_id", "label_column": "label", "feature_columns": ["x"],
             "n_rows": 2, "class_counts": {"0": 1, "1": 1}, "status": "verified_primary_source"}
    return blob, entry


class DataTests(unittest.TestCase):
    def test_valid_csv_and_corruption(self):
        blob, entry = fixture()
        self.assertEqual(validate_csv(blob, entry)[0]["x"], "1.5")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_csv(blob.replace(b"1.5", b"1.6"), entry)

    def test_nonfinite_even_when_checksum_matches(self):
        blob, entry = fixture()
        blob = blob.replace(b"1.5", b"nan")
        entry["sha256"] = hashlib.sha256(blob).hexdigest()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_csv(blob, entry)

    def test_duplicate_ids(self):
        blob, entry = fixture()
        blob = blob.replace(b"b,3.0", b"a,3.0")
        entry["sha256"] = hashlib.sha256(blob).hexdigest()
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_csv(blob, entry)

    def test_schema_not_only_shape(self):
        blob, entry = fixture()
        entry["feature_columns"] = ["wrong_name"]
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_csv(blob, entry)

    def test_class_count_oracle(self):
        blob, entry = fixture()
        entry["class_counts"] = {"0": 2}
        with self.assertRaisesRegex(ValueError, "class counts"):
            validate_csv(blob, entry)

    def test_placeholder_and_floating_revision_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, {"repo_id": "YOUR_HF_USERNAME/data"})
            with self.assertRaises(ValueError):
                checked_config(path)
            write_json(path, {"repo_id": "team/data", "revision": "main", "manifest_sha256": "b" * 64})
            with self.assertRaisesRegex(ValueError, "40-character"):
                checked_config(path)

    def test_hf_transport_pins_same_revision_and_checks_manifest(self):
        blob, entry = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = folder / "tiny.csv"
            data.write_bytes(blob)
            manifest = folder / "manifest.json"
            write_json(manifest, {"schema_version": 1, "datasets": {"tiny": entry}})
            config = folder / "config.json"
            write_json(config, {"repo_id": "team/data", "revision": "a" * 40,
                                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                return str(manifest if kwargs["filename"] == "manifest.json" else data)

            fake = types.ModuleType("huggingface_hub")
            fake.hf_hub_download = download
            with patch.dict(sys.modules, {"huggingface_hub": fake}):
                actual, _ = verified_csv_path(config, "tiny")
                self.assertEqual(actual, data)
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(x["revision"] == "a" * 40 and x["repo_type"] == "dataset" for x in calls))
                manifest.write_bytes(manifest.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "Manifest checksum"):
                    verified_csv_path(config, "tiny")

    def test_raw_label_first_or_last_no_scaling(self):
        base = {"features": ["x", "z"], "n_rows": 2, "label_mapping": {"1": 0, "2": 1},
                "class_counts": {"0": 1, "1": 1}}
        a = parse_records("tiny", b"1,12,7\n2,31,8\n", {**base, "label_index": 0})
        b = parse_records("tiny", b"12,7,1\n31,8,2\n", {**base, "label_index": 2})
        self.assertEqual(a, b)
        self.assertEqual(a[0], ["tiny:000000", "12", "7", "0"])

    def test_raw_unknown_label_and_wrong_shape_reject(self):
        spec = {"features": ["x"], "label_index": 1, "label_mapping": {"1": 0},
                "class_counts": {"0": 1}, "n_rows": 1}
        with self.assertRaisesRegex(ValueError, "unexpected label"):
            parse_records("tiny", b"12,3\n", spec)
        with self.assertRaisesRegex(ValueError, "column count"):
            parse_records("tiny", b"12,1,0\n", spec)


if __name__ == "__main__":
    unittest.main()
