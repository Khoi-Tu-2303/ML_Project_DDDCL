"""Dedicated CLI script to execute the official R1 Final Benchmark Matrix (30 seeds, 200 epochs).

Usage:
  python tools/run_final_matrix.py --dataset haberman
  python tools/run_final_matrix.py --dataset new_thyroid
  python tools/run_final_matrix.py --dataset banknote
  python tools/run_final_matrix.py --dataset all
"""

import argparse
from pathlib import Path
import sys
import time

from dddcl.runner import load_yaml_or_json_config, execute_single_run


def run_dataset_final(
    config_path: Path,
    target_dataset: str,
    resume: bool = True,
) -> None:
    cfg = load_yaml_or_json_config(config_path)
    protocol_id = cfg.get("protocol_id", "paper_interpretation_clean_v1")
    runs_dir = Path(cfg.get("runs_dir", "outputs/runs_final"))
    splits_dir = Path("outputs/splits")
    release_dir = Path("hf_release")

    datasets = [target_dataset] if target_dataset != "all" else cfg.get("datasets", [])
    strategies = cfg.get("strategies", [])
    seeds = cfg.get("seeds", list(range(30)))

    total_runs = len(datasets) * len(strategies) * len(seeds)
    print("=" * 65)
    print(f"Executing Final R1 Benchmark for {datasets}")
    print(f"Total fits: {total_runs} (Epochs: {cfg.get('epochs', 200)}, Seeds: 0..{len(seeds)-1})")
    print(f"Target runs directory: {runs_dir}")
    print("=" * 65)

    completed_count = 0
    skipped_count = 0
    t_start = time.time()

    run_index = 0
    for d in datasets:
        for s in strategies:
            for seed in seeds:
                run_index += 1
                run_id = f"{d}_{s}_s{seed}"
                status_file = runs_dir / run_id / "status.json"
                if status_file.is_file():
                    skipped_count += 1
                    continue

                t0 = time.time()
                res = execute_single_run(
                    raw_config=cfg,
                    dataset_id=d,
                    strategy=s,
                    seed=seed,
                    runs_dir=runs_dir,
                    splits_dir=splits_dir,
                    release_dir=release_dir,
                    resume=resume,
                )
                dur = time.time() - t0
                completed_count += 1
                acc = res.get("test_accuracy", 0.0) * 100.0
                f1 = res.get("test_macro_f1", 0.0) * 100.0
                print(f"[{run_index:3d}/{total_runs:3d}] {run_id:<32} -> Acc: {acc:5.2f}% | F1: {f1:5.2f}% | {dur:4.1f}s")

    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"[+] DATASET '{target_dataset}' RUNS COMPLETE:")
    print(f"    - Newly completed: {completed_count}")
    print(f"    - Previously completed: {skipped_count}")
    print(f"    - Total duration: {elapsed:.1f}s ({elapsed/60:.1f} mins)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Run Final R1 Benchmark Matrix")
    parser.add_argument("--config", type=Path, default=Path("configs/final_r1.yaml"), help="Config file")
    parser.add_argument("--dataset", type=str, default="all", choices=["haberman", "new_thyroid", "banknote", "all"])
    parser.add_argument("--no-resume", action="store_true", help="Force re-run")
    args = parser.parse_args()

    run_dataset_final(
        config_path=args.config,
        target_dataset=args.dataset,
        resume=(not args.no_resume),
    )


if __name__ == "__main__":
    main()

