"""
DDDCL Command Line Interface.
Entrypoints for doctor, data checks, running experiments, and reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


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


def main() -> None:
    parser = argparse.ArgumentParser(description="DDDCL Project Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # doctor
    doctor_parser = subparsers.add_parser("doctor", help="Inspect and verify local environment")
    doctor_parser.add_argument("--json-out", type=Path, default=None, help="Save doctor report to JSON")

    # data-check placeholder contract
    data_check_parser = subparsers.add_parser("data-check", help="Verify hosted or local dataset release")
    data_check_parser.add_argument("--config", type=Path, required=True, help="Path to data config JSON")

    # make-splits placeholder contract
    splits_parser = subparsers.add_parser("make-splits", help="Generate and save split files")
    splits_parser.add_argument("--config", type=Path, required=True, help="Path to experiment config YAML")

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
    elif args.command in ["data-check", "make-splits", "run", "report"]:
        print(f"[!] Command '{args.command}' is defined in architecture spec and will be implemented in subsequent phases (B/C/D).")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

