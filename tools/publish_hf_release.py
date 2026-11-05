"""Explicitly publish an allowlisted local data release to a PUBLIC HF dataset repo."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from prepare_pilot_release import validate_release


def main() -> None:
    """Validate release, publish it, and save the exact resulting revision config."""
    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_hf.json"))
    args = parser.parse_args()
    if args.repo_id.count("/") != 1 or "YOUR_" in args.repo_id:
        raise ValueError("Use your actual HF username/dataset-name")
    validate_release(args.folder)
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    # Do not silently change visibility of an already existing private repository.
    if api.repo_info(repo_id=args.repo_id, repo_type="dataset").private:
        raise ValueError("Existing repo is private; select a public dataset repo or explicitly change its visibility yourself")
    commit = api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=args.folder,
                               allow_patterns=["manifest.json", "README.md", "data/*.csv", "provenance/*.json"],
                               commit_message="Publish verified Dynamic DDCL pilot data release")
    revision = commit.oid
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Upload returned no valid commit SHA; do not substitute main")
    config = {"schema_version": 1, "repo_id": args.repo_id, "revision": revision,
              "manifest_filename": "manifest.json", "manifest_sha256": hashlib.sha256((args.folder / "manifest.json").read_bytes()).hexdigest(),
              "cache_dir": ".cache/huggingface"}
    args.data_config.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.data_config.with_suffix(".json.part")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.data_config)
    print(json.dumps({"dataset_url": f"https://huggingface.co/datasets/{args.repo_id}", "revision": revision,
                      "config_saved": str(args.data_config)}, indent=2))


if __name__ == "__main__":
    main()
