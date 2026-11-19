"""Script to execute official R3 Final Sensitivity Analysis on initial lambda values (0.2, 0.4, 0.6).

Analyzes the impact of the SPL starting threshold on convergence and final test accuracy
across benchmark datasets 'haberman' and 'new_thyroid' for strategy 'sp_ddcl_point' (30 seeds, 200 epochs).

Note: For lambda=0.4, metrics are directly reused from the verified 200-epoch R1 final runs
in outputs/runs_final/, saving compute and preserving exact alignment.

Outputs saved to:
- outputs/sensitivity_final/runs/<dataset>_lam<lambda>_s<seed>.json (individual atomic checkpoints)
- outputs/sensitivity_final/lambda_sweep_final.json (aggregated results)
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from dddcl.data.hf import load_dataset_from_release
from dddcl.data.splits import load_split
from dddcl.data.preprocess import prepare_processed_split
from dddcl.curriculum.geometry import compute_geometry
from dddcl.curriculum.partition import partition_by_class
from dddcl.curriculum.scoring import create_curriculum
from dddcl.training.loop import create_curriculum_batches, evaluate_dataset, train_dddcl_model
from dddcl.models.mlp import create_mlp_model


def run_final_sensitivity_analysis(
    datasets: list[str] = ("haberman", "new_thyroid"),
    lambda_values: list[float] = (0.2, 0.4, 0.6),
    seeds: list[int] = list(range(30)),
    epochs: int = 200,
    rel_dir: Path = Path("hf_release"),
    splits_dir: Path = Path("outputs/splits"),
    runs_final_dir: Path = Path("outputs/runs_final"),
    out_dir: Path = Path("outputs/sensitivity_final"),
    protocol_id: str = "paper_interpretation_clean_v1",
) -> list[dict]:
    runs_out = out_dir / "runs"
    runs_out.mkdir(parents=True, exist_ok=True)
    strategy = "sp_ddcl_point"

    print("=" * 65)
    print("Executing Official R3 Final Sensitivity Analysis (30 seeds, 200 epochs)")
    print(f"Datasets: {datasets}")
    print(f"Lambda initial values: {lambda_values}")
    print(f"Seeds: {min(seeds)}..{max(seeds)} ({len(seeds)} total)")
    print(f"Epochs: {epochs}")
    print(f"Output directory: {out_dir}")
    print("=" * 65)

    all_aggregated = []
    t_start = time.time()

    for ds_id in datasets:
        ds_bundle = load_dataset_from_release(rel_dir, ds_id)

        for lam in lambda_values:
            accs = []
            f1s = []
            steps = []
            exposures = []
            durations = []
            run_records = []

            print(f"\n[*] Dataset: {ds_id:<12} | Lambda initial: {lam}")

            for s in seeds:
                indiv_file = runs_out / f"{ds_id}_lam{lam}_s{s}.json"

                # Check if already computed
                if indiv_file.is_file() and indiv_file.stat().st_size > 0:
                    with open(indiv_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    accs.append(data["accuracy"])
                    f1s.append(data["macro_f1"])
                    steps.append(data["total_optimizer_steps"])
                    exposures.append(data["total_sample_exposures"])
                    durations.append(data.get("duration_seconds", 0.0))
                    run_records.append(data)
                    continue

                # If lambda == 0.4, try to load from outputs/runs_final
                if abs(lam - 0.4) < 1e-5:
                    r1_path = runs_final_dir / f"{ds_id}_sp_ddcl_point_s{s}" / "metrics.json"
                    if r1_path.is_file():
                        with open(r1_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        entry = {
                            "dataset_id": ds_id,
                            "strategy": strategy,
                            "seed": s,
                            "lambda_initial": lam,
                            "accuracy": m.get("test_accuracy", 0.0),
                            "macro_f1": m.get("test_macro_f1", 0.0),
                            "total_optimizer_steps": m.get("total_optimizer_steps", 0),
                            "total_sample_exposures": m.get("total_sample_exposures", 0),
                            "duration_seconds": m.get("duration_seconds", 0.0),
                            "source": "reused_from_r1_final",
                        }
                        with open(indiv_file, "w", encoding="utf-8") as f:
                            json.dump(entry, f, indent=2)
                        accs.append(entry["accuracy"])
                        f1s.append(entry["macro_f1"])
                        steps.append(entry["total_optimizer_steps"])
                        exposures.append(entry["total_sample_exposures"])
                        durations.append(entry["duration_seconds"])
                        run_records.append(entry)
                        print(f"  [reused R1] Seed {s:2d}: Acc={entry['accuracy']*100:5.2f}% | F1={entry['macro_f1']*100:5.2f}% | Steps={entry['total_optimizer_steps']}")
                        continue

                # Otherwise, execute training for 200 epochs
                t0 = time.time()
                split_file = splits_dir / protocol_id / ds_id / f"seed_{s}.json"
                split_bundle = load_split(split_file, bundle=ds_bundle)
                proc = prepare_processed_split(ds_bundle, split_bundle)

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
                    epochs=epochs,
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

                dur = time.time() - t0
                opt_steps = history[-1].get("total_optimizer_steps", 0) if history else 0
                sample_exp = history[-1].get("total_sample_exposures", 0) if history else 0

                entry = {
                    "dataset_id": ds_id,
                    "strategy": strategy,
                    "seed": s,
                    "lambda_initial": lam,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "total_optimizer_steps": opt_steps,
                    "total_sample_exposures": sample_exp,
                    "duration_seconds": dur,
                    "source": "fresh_training",
                }
                with open(indiv_file, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2)

                accs.append(entry["accuracy"])
                f1s.append(entry["macro_f1"])
                steps.append(entry["total_optimizer_steps"])
                exposures.append(entry["total_sample_exposures"])
                durations.append(entry["duration_seconds"])
                run_records.append(entry)
                print(f"  [fresh fit] Seed {s:2d}: Acc={entry['accuracy']*100:5.2f}% | F1={entry['macro_f1']*100:5.2f}% | Steps={opt_steps} | {dur:4.1f}s")

            agg_entry = {
                "dataset_id": ds_id,
                "strategy": strategy,
                "lambda_initial": lam,
                "seeds_completed": len(accs),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std_ddof1": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std_ddof1": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                "avg_optimizer_steps": float(np.mean(steps)),
                "avg_sample_exposures": float(np.mean(exposures)),
                "avg_duration_seconds": float(np.mean(durations)),
                "individual_runs": run_records,
            }
            all_aggregated.append(agg_entry)
            print(f"--> Summary {ds_id} (lam={lam}): Acc = {agg_entry['accuracy_mean']*100:5.2f}% +- {agg_entry['accuracy_std_ddof1']*100:5.2f}% | Opt steps = {agg_entry['avg_optimizer_steps']:.1f}")

    out_file = out_dir / "lambda_sweep_final.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_aggregated, f, indent=2)

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"[+] OFFICIAL R3 FINAL SENSITIVITY ANALYSIS COMPLETE ({total_time:.1f}s)")
    print(f"Saved aggregated results to {out_file}")
    print("=" * 65)
    return all_aggregated


def main():
    parser = argparse.ArgumentParser(description="Run Final R3 Sensitivity Matrix")
    parser.add_argument("--datasets", type=str, default="haberman,new_thyroid", help="Comma-separated datasets")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lambdas", type=str, default="0.2,0.4,0.6", help="Comma-separated lambdas")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]

    run_final_sensitivity_analysis(
        datasets=datasets,
        lambda_values=lambdas,
        seeds=list(range(args.seeds)),
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
