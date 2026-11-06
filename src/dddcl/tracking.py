"""Experiment tracking module for DDDCL.

Follows protocols from docs/11_TRACKING_REPORTS.md:
- Local JSONL trace logging (outputs/runs/<run_id>/trace.jsonl).
- Graceful Weights & Biases (W&B) integration with strict config allowlist.
- Offline/disabled fallback when W&B credentials or network are absent.
- Preserves reproducibility without leaking secrets or environment variables.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping


# Explicit allowed config keys to prevent leaking sensitive environment variables
CONFIG_ALLOWLIST = {
    "dataset_id",
    "protocol_id",
    "strategy",
    "seed",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "hidden_dims",
    "dropout_rate",
    "activation",
    "lambda_init",
    "lambda_max",
    "lambda_growth",
    "pace_policy",
    "smote_policy",
    "partition_n_bins",
    "data_revision",
    "manifest_hash",
    "dataset_fingerprint",
    "split_seed",
    "run_id",
}


def filter_config_for_tracking(config: Mapping[str, Any]) -> dict[str, Any]:
    """Filter config dictionary against explicit allowlist."""
    filtered: dict[str, Any] = {}
    for k, v in config.items():
        if k in CONFIG_ALLOWLIST or not str(k).startswith("_"):
            # Ensure value is JSON serializable
            try:
                json.dumps(v)
                filtered[k] = v
            except (TypeError, OverflowError):
                filtered[k] = str(v)
    return filtered


class ExperimentTracker:
    """Unified local and optional remote experiment tracker.
    
    Parameters
    ----------
    run_dir : str | Path
        Directory where local traces and artifacts will be saved.
    config : Mapping[str, Any]
        Resolved experiment configuration.
    wandb_mode : str, default='offline'
        Mode for W&B tracking: 'online', 'offline', or 'disabled'.
    wandb_project : str, default='DDDCL_Reproducibility'
    wandb_entity : str | None, default=None
    """

    def __init__(
        self,
        run_dir: str | Path,
        config: Mapping[str, Any],
        wandb_mode: str = "offline",
        wandb_project: str = "DDDCL_Reproducibility",
        wandb_entity: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.run_dir / "trace.jsonl"
        self.config = filter_config_for_tracking(config)
        self.wandb_mode = wandb_mode.lower()
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        self._wandb_run = None
        self._wandb_url: str | None = None
        self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize W&B session if enabled and installed."""
        if self.wandb_mode == "disabled":
            self._write_wandb_url("disabled")
            return

        try:
            import wandb
            run_id = str(self.config.get("run_id", "run"))
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                id=run_id,
                resume="allow",
                mode=self.wandb_mode,
                config=self.config,
                dir=str(self.run_dir),
            )
            if self._wandb_run and hasattr(self._wandb_run, "url"):
                self._wandb_url = self._wandb_run.url
                self._write_wandb_url(self._wandb_url or self.wandb_mode)
            else:
                self._write_wandb_url(self.wandb_mode)
        except Exception:
            # Fallback to local only
            self._wandb_run = None
            self._write_wandb_url("fallback_local_error")

    def _write_wandb_url(self, content: str) -> None:
        """Write W&B URL or tracking status to file."""
        url_file = self.run_dir / "wandb_url.txt"
        with open(url_file, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

    def log_epoch(self, epoch_data: dict[str, Any]) -> None:
        """Log epoch-level metrics to JSONL trace and W&B."""
        record = dict(epoch_data)
        record["timestamp"] = time.time()

        # Write to local JSONL
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        # Log to W&B if active
        if self._wandb_run is not None:
            try:
                # W&B prefers clean scalar values
                wb_dict = {
                    k: v for k, v in record.items()
                    if isinstance(v, (int, float, str, bool))
                }
                self._wandb_run.log(wb_dict, step=record.get("epoch"))
            except Exception:
                pass

    def finish(self) -> None:
        """Finalize tracking session."""
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None

    @property
    def wandb_url(self) -> str | None:
        """Return the active W&B URL if online."""
        return self._wandb_url

