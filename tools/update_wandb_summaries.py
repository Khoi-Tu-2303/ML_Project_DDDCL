"""Add the final test metrics of each run to its W&B run summary (resumable, with progress).

``tools/sync_wandb.py`` uploads the offline runs, then updates the summaries one run at a time
without printing progress and cannot continue after an interruption (it only looks at the files it
uploaded in the same call). This script does only the summary step:

* it reads ``<runs_dir>/<run_id>/metrics.json`` for every run;
* it skips runs whose W&B summary already has ``test_accuracy`` (so it can be re-run any time);
* it updates the others with a small thread pool and prints progress.

Optional ``--upload-files`` also attaches files of the run directory (default: ``metrics.json``;
change with ``--files metrics.json,predictions.csv``) to the W&B run, so they show in its Files
tab. A file already present on the W&B run is not uploaded again. This costs extra API calls per
run (one to list the files, one per file to upload), so the run is slower.

Default mode is read-only: it fetches ONE run, shows how long the call took, whether the summary
already holds test metrics, which files are already on W&B, and what would be written or uploaded.
Nothing is written without ``--apply``.

    python tools/update_wandb_summaries.py outputs/runs_final_wandb --project dynamic-ddcl
    python tools/update_wandb_summaries.py outputs/runs_final_wandb --project dynamic-ddcl --apply
    python tools/update_wandb_summaries.py outputs/runs_final_wandb --project dynamic-ddcl --apply --upload-files

The API key is read like in sync_wandb.py: ``WANDB_API_KEY`` env var, else the ``.env`` file, else
a previous ``wandb login``. It is never printed. Test metrics come from ``metrics.json`` only; the
test set is not touched. This script does not import the ``dddcl`` package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_wandb import build_summary, read_env_key  # noqa: E402

MARKER_KEY = "test_accuracy"


def collect_runs(runs_dir: Path) -> list[tuple[str, Path, dict]]:
    """Return (run_id, run_dir, summary_to_write) for every run dir with usable test metrics."""
    runs: list[tuple[str, Path, dict]] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        summary = build_summary(metrics)
        if summary:
            runs.append((str(metrics.get("run_id") or metrics_path.parent.name), metrics_path.parent, summary))
    return runs


def parse_file_names(text: str) -> list[str]:
    """Split ``--files`` into plain file names of the run directory (no sub-paths, no duplicates)."""
    names: list[str] = []
    for raw in text.split(","):
        name = raw.strip()
        if not name:
            continue
        if Path(name).name != name:
            raise ValueError(f"--files takes plain file names of the run directory, got {name!r}")
        if name not in names:
            names.append(name)
    return names


def files_on_wandb(run, names: list[str]) -> set[str]:
    """Names (among ``names``) really attached to the W&B run.

    List the run's files and compare names. Do NOT use ``run.files(names=[...])`` for this: on the
    server used here it returns a placeholder (size 0) for ANY name, even one that does not exist,
    so it always looks as if the file is there.
    """
    wanted = set(names)
    return {f.name for f in run.files(per_page=100) if f.name in wanted}


def update_one(
    api,
    owner: str,
    project: str,
    run_id: str,
    run_dir: Path,
    summary: dict,
    files: list[str],
    retries: int,
) -> str:
    """Return 'updated' (something was written/uploaded) or 'skipped' (nothing to do).

    Raises after ``retries`` tries. Every step re-checks W&B first, so a retry or a later re-run
    never repeats work that already succeeded.
    """
    missing = [name for name in files if not (run_dir / name).is_file()]
    if missing:  # retrying cannot create a local file, so fail at once
        raise FileNotFoundError(f"missing in {run_dir.name}: {', '.join(missing)}")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            run = api.run(f"{owner}/{project}/{run_id}")
            changed = False
            if run.summary.get(MARKER_KEY) is None:
                for key, value in summary.items():
                    run.summary[key] = value
                run.summary.update()
                changed = True
            if files:
                present = files_on_wandb(run, files)
                for name in files:
                    if name in present:
                        continue
                    run.upload_file(str(run_dir / name), root=str(run_dir))
                    changed = True
            return "updated" if changed else "skipped"
        except Exception as exc:  # network hiccup, run not visible yet, missing local file, ...
            last = exc
            time.sleep(2.0 * (attempt + 1))
    assert last is not None
    raise last


def main() -> int:
    parser = argparse.ArgumentParser(description="Write test metrics into W&B run summaries.")
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None, help="W&B user/team (default: account of the key)")
    parser.add_argument("--apply", action="store_true", help="Write to W&B (default: probe one run, read-only)")
    parser.add_argument(
        "--upload-files",
        action="store_true",
        help="Also attach files of the run directory to the W&B run (see --files); slower",
    )
    parser.add_argument(
        "--files",
        default="metrics.json",
        help="Comma-separated file names of the run directory to attach with --upload-files",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N runs (for a trial)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds for each API call")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    try:
        file_names = parse_file_names(args.files) if args.upload_files else []
    except ValueError as exc:
        print(f"[-] {exc}")
        return 2
    if args.upload_files and not file_names:
        print("[-] --upload-files needs at least one name in --files")
        return 2

    runs = collect_runs(args.runs_dir)
    print(f"[*] Runs with test metrics in metrics.json: {len(runs)}")
    if args.limit is not None:
        runs = runs[: max(args.limit, 0)]
        print(f"[*] --limit: processing only the first {len(runs)} run(s)")
    if file_names:
        print(f"[*] Files to attach to each run: {', '.join(file_names)}")
    if not runs:
        return 1

    if not os.environ.get("WANDB_API_KEY"):
        key = read_env_key(args.env_file)
        if key:
            os.environ["WANDB_API_KEY"] = key

    import wandb

    api = wandb.Api(timeout=args.timeout)
    owner = args.entity or api.default_entity
    print(f"[*] Target: {owner}/{args.project}")

    if not args.apply:
        run_id, run_dir, summary = runs[0]
        t0 = time.time()
        run = api.run(f"{owner}/{args.project}/{run_id}")
        dt = time.time() - t0
        has = run.summary.get(MARKER_KEY)
        print(f"[*] Probe run {run_id}: api.run took {dt:.1f}s; {MARKER_KEY} on W&B = {has!r}")
        print(f"[*] Would write: {json.dumps(summary, sort_keys=True)}")
        if file_names:
            t1 = time.time()
            present = files_on_wandb(run, file_names)
            print(f"[*] run.files took {time.time() - t1:.1f}s")
            for name in file_names:
                local = run_dir / name
                state = "already on W&B" if name in present else "would upload"
                note = "" if local.is_file() else "  [!] local file missing"
                print(f"[*]   {name}: {state} ({local.stat().st_size if local.is_file() else 0} bytes){note}")
        print("[+] Read-only probe done. Add --apply to update all runs.")
        return 0

    counts = {"updated": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {
            pool.submit(
                update_one, api, owner, args.project, rid, rdir, summ, file_names, args.retries
            ): rid
            for rid, rdir, summ in runs
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            rid = futures[fut]
            try:
                counts[fut.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                failures.append(f"{rid}: {type(exc).__name__}: {exc}")
            if done % 25 == 0 or done == len(runs):
                rate = done / max(time.time() - start, 1e-9)
                eta = (len(runs) - done) / rate if rate else 0
                print(f"[*] {done}/{len(runs)}  {counts}  {rate:.2f} run/s  ETA {eta/60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Re-run the same command to continue (finished runs are skipped).")
        pool.shutdown(wait=False, cancel_futures=True)
        return 130
    pool.shutdown(wait=True)

    for line in failures[:10]:
        print(f"    [!] {line}")
    if len(failures) > 10:
        print(f"    ... and {len(failures) - 10} more failure(s)")
    print(f"[+] Done: {counts}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
