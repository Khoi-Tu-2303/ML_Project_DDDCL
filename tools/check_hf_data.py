"""Read and summarize a pinned, verified HF canonical dataset (no training)."""
import argparse
import json
from pathlib import Path
from hf_data import verified_csv_path


def main() -> None:
    """Parse local config, read HF snapshot and show non-sensitive summary."""
    from dotenv import load_dotenv
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    path, entry = verified_csv_path(args.config, args.dataset)
    print(json.dumps({"dataset": args.dataset, "rows": entry["n_rows"],
                      "features": entry["feature_columns"], "class_counts": entry["class_counts"],
                      "checksum_verified": True, "cached_file": str(path)}, indent=2))


if __name__ == "__main__":
    main()
