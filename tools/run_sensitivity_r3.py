"""Script to execute R3 Sensitivity Analysis on initial lambda values (0.2, 0.4, 0.6).

Analyzes the impact of the SPL starting threshold on convergence and final test accuracy
across two benchmark datasets: 'haberman' and 'new_thyroid' for strategy 'sp_ddcl_point'.
Outputs saved to outputs/sensitivity/lambda_sweep.json
"""

from pathlib import Path
import json
import numpy as np

from dddcl.data.hf import load_dataset_from_release
from dddcl.data.splits import load_split
from dddcl.data.preprocess import prepare_processed_split
from dddcl.curriculum.geometry import compute_geometry
from dddcl.curriculum.partition import partition_by_class
from dddcl.curriculum.scoring import create_curriculum
from dddcl.training.loop import create_curriculum_batches, evaluate_dataset, train_dddcl_model
from dddcl.models.mlp import create_mlp_model


def run_lambda_sensitivity_analysis() -> list[dict]:
    rel_dir = Path("hf_release")
    splits_dir = Path("outputs/splits")
    out_dir = Path("outputs/sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["haberman", "new_thyroid"]
    lambda_values = [0.2, 0.4, 0.6]
    seeds = [0, 1, 2]
    strategy = "sp_ddcl_point"

    results = []

    print("=" * 60)
    print("Executing R3 Sensitivity Analysis: Initial Lambda Sweep [0.2, 0.4, 0.6]")
    print("=" * 60)

    for ds_id in datasets:
        ds_bundle = load_dataset_from_release(rel_dir, ds_id)
        for lam in lambda_values:
            accs = []
            f1s = []
            steps = []
            exposures = []

            for s in seeds:
                split_file = splits_dir / "paper_interpretation_clean_v1" / ds_id / f"seed_{s}.json"
                split_bundle = load_split(split_file, bundle=ds_bundle)
                proc = prepare_processed_split(ds_bundle, split_bundle)

                # Curriculum setup
                geom = compute_geometry(proc.X_train_raw, proc.y_train)
                part = partition_by_class(geom.normalized_distances, proc.y_train, n_levels=5)
                curr_res = create_curriculum(
                    strategy=strategy,
                    sample_ids=proc.train_ids,
                    normalized_distances=geom.normalized_distances,
                    bin_assignments=part.bin_assignments,
                    seed=s,
                )

                batches = create_curriculum_batches(
                    X_train=proc.X_train_scaled,
                    y_train=proc.y_train,
                    train_ids=proc.train_ids,
                    curriculum_result=curr_res,
                    batch_size=32,
                )

                model = create_mlp_model(
                    input_dim=proc.X_train_scaled.shape[1],
                    num_classes=ds_bundle.n_classes,
                    hidden_dims=(64, 32),
                    dropout_rate=0.0,
                    seed=s,
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                trained_m, history = train_dddcl_model(
                    model=model,
                    optimizer=optimizer,
                    batches=batches,
                    strategy=strategy,
                    epochs=5,
                    lambda_init=lam,
                    lambda_max=0.6,
                    lambda_growth=1.1,
                )

                _, metrics = evaluate_dataset(
                    model=trained_m,
                    X=proc.X_test_scaled,
                    y=proc.y_test,
                    num_classes=ds_bundle.n_classes,
                )

                accs.append(metrics["accuracy"])
                f1s.append(metrics["macro_f1"])
                if history:
                    steps.append(history[-1].get("total_optimizer_steps", 0))
                    exposures.append(history[-1].get("total_sample_exposures", 0))

            res_entry = {
                "dataset_id": ds_id,
                "strategy": strategy,
                "lambda_initial": lam,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs, ddof=1)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s, ddof=1)),
                "avg_optimizer_steps": float(np.mean(steps)),
                "avg_sample_exposures": float(np.mean(exposures)),
                "individual_runs": [
                    {"seed": s, "accuracy": a, "macro_f1": f}
                    for s, a, f in zip(seeds, accs, f1s)
                ],
            }
            results.append(res_entry)
            print(f"[*] {ds_id:<12} | lambda_init={lam:<4} -> Acc: {res_entry['accuracy_mean']:.4f} +- {res_entry['accuracy_std']:.4f} | Opt Steps: {res_entry['avg_optimizer_steps']:.1f}")

    out_file = out_dir / "lambda_sweep.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[+] R3 SENSITIVITY ANALYSIS COMPLETE! Saved to {out_file}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    import torch
    run_lambda_sensitivity_analysis()
