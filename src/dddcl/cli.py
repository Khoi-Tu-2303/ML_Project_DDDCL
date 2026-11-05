"""
DDDCL Command Line Interface.
Entrypoints for doctor, data checks, dataset splits, running experiments, and reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def get_project_root() -> Path:
    """Locate the root directory of the ML_Project_DDDCL repository."""
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / "requirements.in").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current


def run_doctor(output_json: Path | None = None) -> int:
    """
    Phase A Environment Doctor:
    Performs comprehensive system, environment, dependency, and deterministic CPU PyTorch checks.
    """
    print("=" * 60)
    print("DDDCL Environment & System Doctor (Phase A Verification)")
    print("=" * 60)

    report: Dict[str, Any] = {
        "status": "PASS",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "hash_seed": os.environ.get("PYTHONHASHSEED", "not_set"),
        },
        "environment_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "not_set"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "not_set"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "not_set"),
        },
        "packages": {},
        "pip_check": {},
        "torch_cpu_verification": {},
    }

    # 1. Python version check
    py_ver = sys.version_info
    print(f"[*] Python: {platform.python_version()} ({sys.executable})")
    if py_ver.major != 3 or py_ver.minor != 11:
        print(f"    [!] Warning: Expected Python 3.11.x, got {py_ver.major}.{py_ver.minor}")
        report["python"]["supported"] = False
    else:
        print("    [+] Python 3.11 runtime verified.")
        report["python"]["supported"] = True

    # 2. Package verification
    target_pkgs = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "imbalanced-learn",
        "torch",
        "matplotlib",
        "seaborn",
        "huggingface-hub",
        "wandb",
        "yaml",
        "dotenv",
        "pytest",
    ]

    print("\n[*] Checking core dependencies:")
    all_pkgs_ok = True
    for pkg in target_pkgs:
        import_name = "sklearn" if pkg == "scikit-learn" else ("imblearn" if pkg == "imbalanced-learn" else ("huggingface_hub" if pkg == "huggingface-hub" else pkg))
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"    [+] {pkg:<20}: {ver}")
            report["packages"][pkg] = {"status": "installed", "version": ver}
        except ImportError as e:
            print(f"    [-] {pkg:<20}: NOT INSTALLED ({e})")
            report["packages"][pkg] = {"status": "missing", "error": str(e)}
            all_pkgs_ok = False

    # 3. Pip check
    print("\n[*] Running pip check:")
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False)
        report["pip_check"]["returncode"] = proc.returncode
        report["pip_check"]["stdout"] = proc.stdout.strip()
        report["pip_check"]["stderr"] = proc.stderr.strip()
        if proc.returncode == 0:
            print("    [+] pip check: No broken requirements found.")
        else:
            print(f"    [-] pip check warning/error (code {proc.returncode}):")
            if proc.stdout:
                print(f"        {proc.stdout.strip()}")
            if proc.stderr:
                print(f"        {proc.stderr.strip()}")
            all_pkgs_ok = False
    except Exception as e:
        print(f"    [-] Could not run pip check: {e}")
        report["pip_check"]["error"] = str(e)

    # 4. PyTorch CPU & deterministic forward-backward check
    print("\n[*] Verifying PyTorch CPU execution & deterministic update:")
    try:
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)

        model = torch.nn.Linear(2, 1)
        x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        w_before = model.weight.detach().clone()
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        w_after = model.weight.detach().clone()

        is_finite = bool(torch.isfinite(loss).item())
        weights_updated = not bool(torch.equal(w_before, w_after))
        loss_val = float(loss.item())

        print(f"    [+] PyTorch Version: {torch.__version__}")
        print(f"    [+] Loss finite: {is_finite} (loss={loss_val:.6f})")
        print(f"    [+] Weights updated on step: {weights_updated}")

        # Determinism repeat test with seed 42
        torch.manual_seed(42)
        model2 = torch.nn.Linear(2, 1)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        optimizer2.zero_grad()
        loss2 = torch.nn.functional.binary_cross_entropy_with_logits(model2(x), y)
        loss2.backward()
        optimizer2.step()
        loss_val2 = float(loss2.item())
        w_after2 = model2.weight.detach().clone()

        reproducible = bool(torch.equal(w_after, w_after2)) and abs(loss_val - loss_val2) < 1e-7
        print(f"    [+] Deterministic execution repeat test: {reproducible}")

        report["torch_cpu_verification"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "loss_finite": is_finite,
            "weights_updated": weights_updated,
            "deterministic_reproducible": reproducible,
            "sample_loss": loss_val,
        }

        if not (is_finite and weights_updated and reproducible):
            all_pkgs_ok = False
            report["status"] = "FAIL"
    except Exception as e:
        print(f"    [-] PyTorch check failed: {e}")
        report["torch_cpu_verification"]["error"] = str(e)
        all_pkgs_ok = False
        report["status"] = "FAIL"

    # Summary
    print("\n" + "=" * 60)
    if all_pkgs_ok and report["status"] == "PASS":
        print("[+] DOCTOR STATUS: ALL CHECKS PASSED (Phase A Environment Gate Met)")
    else:
        print("[-] DOCTOR STATUS: FAILED CHECKS DETECTED")
        report["status"] = "FAIL"
    print("=" * 60)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[*] Report saved to {output_json}")

    return 0 if report["status"] == "PASS" else 1


def run_data_check(
    config_path: Optional[Path] = None,
    release_dir: Optional[Path] = None,
    dataset: Optional[str] = None
) -> int:
    """
    Phase B Data Check:
    Verifies dataset integrity, checksums, schema, finite features, and class distribution.
    Can verify via hosted HF config OR local release directory.
    """
    from dddcl.data.hf import load_dataset_from_release, load_dataset_from_hf

    print("=" * 60)
    print("DDDCL Data Check (Phase B Verification)")
    print("=" * 60)

    datasets_to_check = [dataset] if dataset and dataset != "all" else ["haberman", "new_thyroid", "banknote"]
    all_ok = True

    # If config_path provided and valid, try HF loading
    use_hf = False
    if config_path and config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                c = json.load(f)
            if "YOUR_" not in c.get("repo_id", "") and c.get("revision"):
                use_hf = True
                print(f"[*] Checking via hosted Hugging Face config: {config_path}")
        except Exception:
            use_hf = False

    if not use_hf:
        rel_dir = release_dir or Path("hf_release")
        print(f"[*] Checking local release directory: {rel_dir}")
        if not rel_dir.exists():
            print(f"[-] Release directory '{rel_dir}' not found.")
            return 1

    for ds_id in datasets_to_check:
        try:
            if use_hf and config_path:
                bundle = load_dataset_from_hf(config_path, ds_id)
                mode_str = "HOSTED HF"
            else:
                rel_dir = release_dir or Path("hf_release")
                bundle = load_dataset_from_release(rel_dir, ds_id)
                mode_str = "LOCAL RELEASE"

            print(f"\n[+] Dataset: {bundle.dataset_id} ({mode_str})")
            print(f"    - Rows (N): {len(bundle.frame)}")
            print(f"    - Features (d): {len(bundle.feature_columns)} {bundle.feature_columns}")
            print(f"    - Classes (C): {bundle.n_classes}")
            counts = bundle.y.value_counts().to_dict()
            print(f"    - Class distribution: {counts}")
            print(f"    - Manifest SHA-256: {bundle.manifest_hash[:16]}...")
            print(f"    - Status: VERIFIED OK")
        except Exception as e:
            print(f"\n[-] Dataset: {ds_id} FAILED: {e}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("[+] DATA CHECK STATUS: ALL DATASETS VERIFIED")
        return 0
    else:
        print("[-] DATA CHECK STATUS: FAILED DATASETS DETECTED")
        return 1


def run_make_splits(
    config_path: Path,
    release_dir: Optional[Path] = None,
    out_dir: Path = Path("outputs/splits")
) -> int:
    """
    Phase B Split Generation:
    Generates deterministic stratified 70/20/10 splits for all datasets and seeds in config.
    """
    from dddcl.data.hf import load_dataset_from_release, load_dataset_from_hf
    from dddcl.data.splits import create_stratified_split, save_split

    print("=" * 60)
    print("DDDCL Split Generation (Phase B Protocol)")
    print("=" * 60)

    if not config_path.exists():
        print(f"[-] Experiment config not found: {config_path}")
        return 1

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    protocol_id = cfg.get("protocol_id", "paper_interpretation_clean_v1")
    datasets = cfg.get("datasets", ["haberman", "new_thyroid", "banknote"])
    seeds = cfg.get("seeds", [0, 1, 2])
    rel_dir = release_dir or Path("hf_release")

    print(f"[*] Protocol: {protocol_id}")
    print(f"[*] Datasets: {datasets}")
    print(f"[*] Seeds: {seeds}")
    print(f"[*] Target directory: {out_dir}\n")

    total_splits = 0
    for ds_id in datasets:
        try:
            bundle = load_dataset_from_release(rel_dir, ds_id)
            print(f"[*] Processing {ds_id} (N={len(bundle.frame)}, C={bundle.n_classes}):")
            for seed in seeds:
                split = create_stratified_split(bundle, seed=seed, protocol_id=protocol_id)
                saved_path = save_split(split, out_dir=out_dir)
                n_train = len(split.train_ids)
                n_val = len(split.validation_ids)
                n_test = len(split.test_ids)
                total = n_train + n_val + n_test
                print(f"    [+] Seed {seed:2d}: Train={n_train} ({n_train/total:.1%}), Val={n_val} ({n_val/total:.1%}), Test={n_test} ({n_test/total:.1%}) -> {saved_path.name}")
                total_splits += 1
        except Exception as e:
            print(f"    [-] Failed for dataset '{ds_id}': {e}")
            return 1

    print("\n" + "=" * 60)
    print(f"[+] MAKE-SPLITS STATUS: {total_splits} SPLIT FILES SUCCESSFULLY GENERATED")
    print("=" * 60)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="DDDCL Project Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # doctor
    doctor_parser = subparsers.add_parser("doctor", help="Inspect and verify local environment")
    doctor_parser.add_argument("--json-out", type=Path, default=None, help="Save doctor report to JSON")

    # data-check
    data_check_parser = subparsers.add_parser("data-check", help="Verify hosted or local dataset release")
    data_check_parser.add_argument("--config", type=Path, default=None, help="Path to data config JSON (e.g. configs/data_hf.json)")
    data_check_parser.add_argument("--release-dir", type=Path, default=Path("hf_release"), help="Path to local release directory")
    data_check_parser.add_argument("--dataset", type=str, default="all", help="Dataset ID or 'all'")

    # make-splits
    splits_parser = subparsers.add_parser("make-splits", help="Generate and save split files")
    splits_parser.add_argument("--config", type=Path, required=True, help="Path to experiment config YAML")
    splits_parser.add_argument("--release-dir", type=Path, default=Path("hf_release"), help="Path to local release directory")
    splits_parser.add_argument("--out", type=Path, default=Path("outputs/splits"), help="Output directory for splits")

    # run placeholder contract
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--config", type=Path, required=True, help="Path to experiment config YAML")
    run_parser.add_argument("--dry-run", action="store_true", help="Print plan and estimate without training")
    run_parser.add_argument("--resume", action="store_true", help="Resume interrupted runs")

    # report placeholder contract
    report_parser = subparsers.add_parser("report", help="Aggregate run results and generate reports")
    report_parser.add_argument("--runs", type=Path, default=Path("outputs/runs"), help="Runs directory")
    report_parser.add_argument("--out", type=Path, default=Path("reports/generated"), help="Output directory")

    args = parser.parse_args()

    if args.command == "doctor":
        sys.exit(run_doctor(output_json=args.json_out))
    elif args.command == "data-check":
        sys.exit(run_data_check(config_path=args.config, release_dir=args.release_dir, dataset=args.dataset))
    elif args.command == "make-splits":
        sys.exit(run_make_splits(config_path=args.config, release_dir=args.release_dir, out_dir=args.out))
    elif args.command in ["run", "report"]:
        print(f"[!] Command '{args.command}' is defined in architecture spec and will be implemented in subsequent phases (C/D/E).")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
