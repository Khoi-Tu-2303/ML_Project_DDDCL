"""Script to execute and evaluate all three Ensemble configurations across pilot datasets and seeds.

Ensembles evaluated:
1. DDCL 6-Strategy Ensemble (DDCL-6)
2. Standard Paper Baseline Ensemble (NN + SVM RBF)
3. Controlled 6-NN Ensemble (6 independent seeds, no curriculum)

Outputs saved to outputs/ensembles/<ensemble_type>/<dataset_id>_s<seed>.json
"""

from pathlib import Path
import json
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


def run_all_pilot_ensembles() -> list[dict]:
    rel_dir = Path("hf_release")
    splits_dir = Path("outputs/splits")
    runs_dir = Path("outputs/runs")
    out_dir = Path("outputs/ensembles")

    datasets = ["haberman", "new_thyroid", "banknote"]
    seeds = [0, 1, 2]
    all_summaries = []

    print("=" * 60)
    print("Executing Pilot Ensemble Experiments (DDCL-6, Standard NN+SVM, Controlled 6-NN)")
    print("=" * 60)

    for ds_id in datasets:
        ds_bundle = load_dataset_from_release(rel_dir, ds_id)
        for s in seeds:
            split_file = splits_dir / "paper_interpretation_clean_v1" / ds_id / f"seed_{s}.json"
            split_bundle = load_split(split_file, bundle=ds_bundle)
            proc = prepare_processed_split(ds_bundle, split_bundle)

            print(f"\n[*] Dataset: {ds_id:<12} | Seed: {s}")

            # 1. DDCL 6-Strategy Ensemble
            res_ddcl = evaluate_ddcl_6strategy_ensemble(
                dataset_id=ds_id,
                seed=s,
                runs_dir=runs_dir,
                splits_dir=splits_dir,
            )
            save_ensemble_result(
                out_dir / "ddcl_6strategy" / f"{ds_id}_s{s}.json",
                res_ddcl,
            )
            print(f"  [+] DDCL-6 Ensemble     : Acc = {res_ddcl.metrics['accuracy']:.4f}, Macro-F1 = {res_ddcl.metrics['macro_f1']:.4f}, Ties = {res_ddcl.tie_count}")

            # 2. Standard Paper Baseline (NN + SVM)
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
            save_ensemble_result(
                out_dir / "standard_nn_svm" / f"{ds_id}_s{s}.json",
                res_std,
            )
            print(f"  [+] Standard (NN+SVM)   : Acc = {res_std.metrics['accuracy']:.4f}, Macro-F1 = {res_std.metrics['macro_f1']:.4f}, Ties = {res_std.tie_count}")

            # 3. Controlled 6-NN Ensemble
            res_ctrl = evaluate_controlled_6nn_ensemble(
                processed_X_train=proc.X_train_scaled,
                y_train=proc.y_train,
                train_ids=proc.train_ids,
                processed_X_test=proc.X_test_scaled,
                y_test=proc.y_test,
                test_ids=proc.test_ids,
                dataset_id=ds_id,
                base_seed=s,
                epochs=5,
            )
            save_ensemble_result(
                out_dir / "controlled_6nn" / f"{ds_id}_s{s}.json",
                res_ctrl,
            )
            print(f"  [+] Controlled 6-NN     : Acc = {res_ctrl.metrics['accuracy']:.4f}, Macro-F1 = {res_ctrl.metrics['macro_f1']:.4f}, Disagreement = {res_ctrl.diversity_stats['pairwise_disagreement_rate']:.4f}")

            all_summaries.append({
                "dataset_id": ds_id,
                "seed": s,
                "ddcl_6_acc": res_ddcl.metrics["accuracy"],
                "standard_acc": res_std.metrics["accuracy"],
                "controlled_acc": res_ctrl.metrics["accuracy"],
            })

    print("\n" + "=" * 60)
    print(f"[+] ALL {len(all_summaries)} ENSEMBLE EXPERIMENTS COMPLETED!")
    print("=" * 60)
    return all_summaries


if __name__ == "__main__":
    run_all_pilot_ensembles()

