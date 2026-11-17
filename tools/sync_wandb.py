"""Upload offline W&B runs to the cloud and add the final test metrics.

Training runs are executed with ``tracking.mode: offline``. Each run keeps its W&B data in
``<runs_dir>/<run_id>/wandb/offline-run-*/run-*.wandb``. This script finds those files,
uploads them with ``wandb sync``, then writes the test metrics from ``metrics.json`` into the
summary of each uploaded run (the tracker itself never logs test metrics).

Run it with an environment that has a recent ``wandb`` (0.18.7 rejects the new 86-character
API keys). ``--project`` is required: pass the ``tracking.project`` of the config the runs were
made with. W&B refuses to reuse the id of a run that was deleted in the same project, so do not
delete runs from the project used for the final results.

    python tools/sync_wandb.py outputs/runs_final --project dynamic-ddcl --dry-run
    python tools/sync_wandb.py outputs/runs_final --project dynamic-ddcl

The API key is read from the ``WANDB_API_KEY`` environment variable, or from the
``WANDB_API_KEY=`` line of ``.env`` (it is never printed), or from a previous ``wandb login``.
Runs already uploaded (marked by a ``*.wandb.synced`` file) are skipped.

This script does not import the ``dddcl`` package.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

# Scalar fields of metrics.json copied into the W&B run summary.
SUMMARY_SCALAR_KEYS = (
    "test_accuracy",
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_loss",
    "duration_seconds",
)


def find_wandb_files(runs_dir: Path) -> list[Path]:
    """Return every offline ``*.wandb`` file under ``<runs_dir>/<run_id>/wandb/offline-run-*/``."""
    return sorted(runs_dir.glob("*/wandb/offline-run-*/*.wandb"))


def is_synced(wandb_file: Path) -> bool:
    """A ``.wandb`` file is synced when W&B left a ``.wandb.synced`` marker next to it."""
    return wandb_file.with_suffix(".wandb.synced").exists()


def run_dir_of(wandb_file: Path) -> Path:
    """Run directory (the one holding metrics.json) for a ``.wandb`` file."""
    return wandb_file.parents[2]


def read_env_key(env_file: Path, name: str = "WANDB_API_KEY") -> str | None:
    """Read ``name`` from a simple ``KEY=value`` env file. Returns None if absent or empty."""
    if not env_file.is_file():
        return None
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == name:
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def has_saved_login() -> bool:
    """True when a previous ``wandb login`` left an api.wandb.ai entry in a netrc file."""
    home = Path.home()
    for name in ("_netrc", ".netrc"):
        path = home / name
        if path.is_file():
            try:
                if "api.wandb.ai" in path.read_text(encoding="utf-8", errors="ignore"):
                    return True
            except OSError:
                continue
    return False


def build_summary(metrics: dict) -> dict:
    """Pick the values of metrics.json that go into the W&B run summary."""
    summary: dict = {}
    for key in SUMMARY_SCALAR_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            summary[key] = value
    recalls = metrics.get("test_per_class_recall")
    if isinstance(recalls, dict):
        for cls, value in recalls.items():
            if isinstance(value, (int, float)):
                summary[f"test_recall_class_{cls}"] = value
    return summary


def chunked(items: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    """Yield ``items`` in lists of at most ``size`` (keeps the command line short on Windows)."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sync_batches(
    files: Sequence[Path],
    project: str,
    entity: str | None,
    batch_size: int,
    parallelism: int,
    env: dict[str, str],
) -> None:
    """Call ``python -m wandb sync`` on the files, batch by batch."""
    for batch in chunked(list(files), batch_size):
        cmd = [sys.executable, "-m", "wandb", "sync", "--project", project, "-n", str(parallelism)]
        if entity:
            cmd += ["--entity", entity]
        cmd += [str(p) for p in batch]
        print(f"[*] wandb sync: {len(batch)} run file(s)")
        subprocess.run(cmd, env=env, check=False)


def update_summaries(
    run_dirs: Sequence[Path],
    project: str,
    entity: str | None,
    retries: int = 5,
) -> tuple[int, list[str]]:
    """Write test metrics into the summary of each uploaded run. Returns (updated, problems)."""
    import wandb

    api = wandb.Api()
    owner = entity or api.default_entity
    updated = 0
    problems: list[str] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            problems.append(f"{run_dir.name}: no metrics.json")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_id = str(metrics.get("run_id") or run_dir.name)
        summary = build_summary(metrics)
        if not summary:
            problems.append(f"{run_id}: metrics.json has no test metrics")
            continue
        for attempt in range(retries):
            try:
                run = api.run(f"{owner}/{project}/{run_id}")
                for key, value in summary.items():
                    run.summary[key] = value
                run.summary.update()
                updated += 1
                break
            except Exception as exc:  # run may not be visible right after upload
                if attempt == retries - 1:
                    problems.append(f"{run_id}: {type(exc).__name__}: {exc}")
                else:
                    time.sleep(2.0)
    return updated, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload offline W&B runs and add test metrics.")
    parser.add_argument("runs_dir", type=Path, nargs="?", default=Path("outputs/runs_final"))
    parser.add_argument(
        "--project",
        required=True,
        help="W&B project to upload into; use the tracking.project of the config the runs were made with",
    )
    parser.add_argument("--entity", default=None, help="W&B user/team (default: account of the API key)")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded; upload nothing")
    parser.add_argument("--no-metrics", action="store_true", help="Skip adding test metrics to run summaries")
    parser.add_argument("--batch-size", type=int, default=50, help="Run files per `wandb sync` call")
    parser.add_argument("--parallelism", type=int, default=5, help="Runs uploaded at the same time")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="File holding WANDB_API_KEY")
    args = parser.parse_args()

    if not args.runs_dir.is_dir():
        print(f"[-] Runs directory not found: {args.runs_dir}")
        return 1

    files = find_wandb_files(args.runs_dir)
    pending = [f for f in files if not is_synced(f)]
    already = len(files) - len(pending)

    attempts: dict[Path, int] = {}
    for f in pending:
        attempts[run_dir_of(f)] = attempts.get(run_dir_of(f), 0) + 1
    resumed = sorted(d.name for d, n in attempts.items() if n > 1)

    print(f"[*] Runs directory : {args.runs_dir}")
    print(f"[*] Offline files  : {len(files)} (already synced: {already}, to upload: {len(pending)})")
    print(f"[*] Project/entity : {args.project} / {args.entity or '(account default)'}")
    if resumed:
        print(
            f"[!] {len(resumed)} run(s) have several offline files (resumed after an interruption); "
            "wandb merges them into one run in start-time order: " + ", ".join(resumed[:5])
            + (" ..." if len(resumed) > 5 else "")
        )

    if args.dry_run:
        for f in pending[:10]:
            print(f"    would upload: {f}")
        if len(pending) > 10:
            print(f"    ... and {len(pending) - 10} more")
        print("[+] Dry run: nothing uploaded.")
        return 0
    if not pending:
        print("[+] Nothing to upload.")
        return 0

    env = dict(os.environ)
    if not env.get("WANDB_API_KEY"):
        key = read_env_key(args.env_file)
        if key:
            env["WANDB_API_KEY"] = key
            os.environ["WANDB_API_KEY"] = key
        elif not has_saved_login():
            print("[-] No W&B credentials: set WANDB_API_KEY, fill it in .env, or run `wandb login`.")
            return 1

    sync_batches(pending, args.project, args.entity, args.batch_size, args.parallelism, env)

    uploaded = [f for f in pending if is_synced(f)]
    failed = [f for f in pending if not is_synced(f)]
    print(f"[+] Uploaded: {len(uploaded)} / {len(pending)} offline file(s)")

    problems: list[str] = []
    if uploaded and not args.no_metrics:
        run_dirs = sorted({run_dir_of(f) for f in uploaded})
        updated, problems = update_summaries(run_dirs, args.project, args.entity)
        print(f"[+] Test metrics added to {updated} / {len(run_dirs)} run(s)")
        for line in problems[:10]:
            print(f"    [!] {line}")
    if failed:
        print(f"[-] {len(failed)} offline file(s) were not uploaded (re-run this script to retry):")
        for f in failed[:10]:
            print(f"    {f}")
    return 1 if (failed or problems) else 0


if __name__ == "__main__":
    sys.exit(main())
