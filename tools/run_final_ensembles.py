"""Dedicated script to execute the official R2 Final Ensemble Experiments (30 seeds).

Evaluates three ensemble paradigms:
1. DDCL-6 Strategy Ensemble (Hard voting among 6 curriculum strategies)
2. Standard Paper Baseline Ensemble (1 NN no_curriculum + 1 SVM RBF)
3. Controlled 6-NN Ensemble (6 independent random initializations, no curriculum, 200 epochs)

Outputs saved to outputs/ensembles_final/<ensemble_type>/<dataset_id>_s<seed>.json
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np

from dddcl.data.hf import load_dataset_from_release
from dddcl.data.splits import load_split
from dddcl.data.preprocess import prepare_processed_split
from dddcl.ensemble import (
    evaluate_ddcl_6strategy_ensemble,
    evaluate_standard_paper_baseline,
    evaluate_controlled_6nn_ensemble,
    save_ensemble_result,
)


def run_final_ensembles(
    datasets: list[str],
    seeds: list[int],
    methods: list[str],
    runs_dir: Path = Path("outputs/runs_final"),
    out_dir: Path = Path("outputs/ensembles_final"),
    splits_dir: Path = Path("outputs/splits"),
    rel_dir: Path = Path("hf_release"),
    protocol_id: str = "paper_interpretation_clean_v1",
    epochs: int = 200,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    total_tasks = len(datasets) * len(seeds) * len(methods)
    print("=" * 65)
    print("Executing Official R2 Final Ensemble Experiments (30 seeds)")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {min(seeds)}..{max(seeds)} ({len(seeds)} total)")
    print(f"Methods: {methods}")
    print(f"Runs directory: {runs_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Controlled 6-NN epochs: {epochs}")
    print(f"Total ensemble evaluations: {total_tasks}")
    print("=" * 65)

    completed_count = 0
    skipped_count = 0
    t_start = time.time()

    for ds_id in datasets:
        ds_bundle = load_dataset_from_release(rel_dir, ds_id)

        for s in seeds:
            split_file = splits_dir / protocol_id / ds_id / f"seed_{s}.json"
            split_bundle = load_split(split_file, bundle=ds_bundle)
            proc = prepare_processed_split(ds_bundle, split_bundle)

            # 1. DDCL-6 Strategy Ensemble
            if "ddcl_6strategy" in methods:
                target_file = out_dir / "ddcl_6strategy" / f"{ds_id}_s{s}.json"
                if target_file.is_file() and target_file.stat().st_size > 0:
                    skipped_count += 1
                else:
                    t0 = time.time()
                    res_ddcl = evaluate_ddcl_6strategy_ensemble(
                        dataset_id=ds_id,
                        seed=s,
                        runs_dir=runs_dir,
                        splits_dir=splits_dir,
                        protocol_id=protocol_id,
                    )
                    save_ensemble_result(target_file, res_ddcl)
                    dur = time.time() - t0
                    completed_count += 1
                    print(f"  [+] [{ds_id} s{s:2d}] DDCL-6 Ensemble   : Acc={res_ddcl.metrics['accuracy']*100:5.2f}% | F1={res_ddcl.metrics['macro_f1']*100:5.2f}% | Ties={res_ddcl.tie_count:2d} | {dur:4.2f}s")

            # 2. Standard Paper Baseline (NN + SVM)
            if "standard_nn_svm" in methods:
                target_file = out_dir / "standard_nn_svm" / f"{ds_id}_s{s}.json"
                if target_file.is_file() and target_file.stat().st_size > 0:
                    skipped_count += 1
                else:
                    t0 = time.time()
                    nn_pred_file = runs_dir / f"{ds_id}_no_curriculum_s{s}" / "predictions.csv"
                    res_std = evaluate_standard_paper_baseline(
                        processed_X_train=proc.X_train_scaled,
                        y_train=proc.y_train,
                        processed_X_test=proc.X_test_scaled,
                        y_test=proc.y_test,
                        test_ids=proc.test_ids,
                        dataset_id=ds_id,
                        seed=s,
                        nn_prediction_file=nn_pred_file,
                    )
                    save_ensemble_result(target_file, res_std)
                    dur = time.time() - t0
                    completed_count += 1
                    print(f"  [+] [{ds_id} s{s:2d}] Standard (NN+SVM) : Acc={res_std.metrics['accuracy']*100:5.2f}% | F1={res_std.metrics['macro_f1']*100:5.2f}% | Ties={res_std.tie_count:2d} | {dur:4.2f}s")

            # 3. Controlled 6-NN Ensemble
            if "controlled_6nn" in methods:
                target_file = out_dir / "controlled_6nn" / f"{ds_id}_s{s}.json"
                if target_file.is_file() and target_file.stat().st_size > 0:
                    skipped_count += 1
                else:
                    t0 = time.time()
                    res_ctrl = evaluate_controlled_6nn_ensemble(
                        processed_X_train=proc.X_train_scaled,
                        y_train=proc.y_train,
                        train_ids=proc.train_ids,
                        processed_X_test=proc.X_test_scaled,
                        y_test=proc.y_test,
                        test_ids=proc.test_ids,
                        dataset_id=ds_id,
                        base_seed=s,
                        epochs=epochs,
                    )
                    save_ensemble_result(target_file, res_ctrl)
                    dur = time.time() - t0
                    completed_count += 1
                    dis = res_ctrl.diversity_stats.get('pairwise_disagreement_rate', 0.0)
                    print(f"  [+] [{ds_id} s{s:2d}] Controlled 6-NN   : Acc={res_ctrl.metrics['accuracy']*100:5.2f}% | F1={res_ctrl.metrics['macro_f1']*100:5.2f}% | Disagree={dis:.4f} | {dur:4.2f}s")

    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("[+] FINAL ENSEMBLE EVALUATIONS COMPLETE:")
    print(f"    - Newly evaluated: {completed_count}")
    print(f"    - Previously completed: {skipped_count}")
    print(f"    - Total duration: {elapsed:.1f}s ({elapsed/60:.1f} mins)")
    print("=" * 65)
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Run Final R2 Ensemble Matrix")
    parser.add_argument("--dataset", type=str, default="all", choices=["haberman", "new_thyroid", "banknote", "all"])
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated methods or 'all'")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs for controlled 6-NN")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds (0..N-1)")
    args = parser.parse_args()

    datasets = ["haberman", "new_thyroid", "banknote"] if args.dataset == "all" else [args.dataset]
    all_methods = ["ddcl_6strategy", "standard_nn_svm", "controlled_6nn"]
    if args.methods == "all":
        methods = all_methods
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip() in all_methods]

    seeds = list(range(args.seeds))

    run_final_ensembles(
        datasets=datasets,
        seeds=seeds,
        methods=methods,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

