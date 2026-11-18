"""Vẽ biểu đồ "Error loss for <dataset>" (mean qua các seed) kèm khung zoom, theo file config.

Dùng:  python tools/plot_error_loss.py configs/plot_error_loss.yaml
       python tools/plot_error_loss.py configs/plot_error_loss.yaml --output outputs/figures/x.png
Đầu vào : <runs_dir>/<dataset>_<strategy>_s<seed>/trace.jsonl                 (source: ours, mặc định)
          <author_dir>/<Dataset>/<STRATEGY>/run_XX.json → history["loss"]     (source: author)
Đầu ra  : config["output"]  (mặc định outputs/figures/r1_error_loss.png)

Config (YAML hoặc JSON) quyết định: dataset nào / panel nào, so sánh chiến lược nào,
mỗi đường màu gì + nét liền hay đứt, khung zoom ra sao. Xem configs/plot_error_loss.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INSET = {"enabled": True, "xlim": [25, 200], "ylim": None,
                 "rect": [0.33, 0.25, 0.62, 0.52], "connectors": [2, 4]}


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def load_curve(run_dir: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Đọc trace.jsonl; epoch bị ghi trùng thì giữ dòng cuối."""
    by_epoch: dict[int, float] = {}
    for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_epoch[row["epoch"]] = row[metric]
    epochs = sorted(by_epoch)
    return np.array(epochs), np.array([by_epoch[e] for e in epochs], dtype=float)


def collect(runs_dir: Path, dataset: str, strategy: str, metric: str, seeds):
    """Trả (epoch, ma trận seeds×epochs) hoặc None nếu chưa có run."""
    dirs = [d for d in runs_dir.glob(f"{dataset}_{strategy}_s*")
            if (d / "trace.jsonl").exists() and d.name.rsplit("_s", 1)[1].isdigit()]
    if seeds is not None:
        wanted = {int(s) for s in seeds}
        dirs = [d for d in dirs if int(d.name.rsplit("_s", 1)[1]) in wanted]
    if not dirs:
        return None
    dirs.sort(key=lambda p: int(p.name.rsplit("_s", 1)[1]))
    curves = [load_curve(d, metric) for d in dirs]
    n = min(len(c[0]) for c in curves)
    return curves[0][0][:n], np.stack([c[1][:n] for c in curves])


# Kết quả chạy lại code tác giả (tools/run_author_code.py): outputs/author_rerun/<Dataset>/<NAME>/run_XX.json
AUTHOR_STRATEGY = {"no_curriculum": "NO_CL", "ddcl_density": "CL_DENSITY", "ddcl_point": "CL_POINT",
                   "self_paced": "SELF_PACED", "sp_ddcl_density": "SP_DENSITY", "sp_ddcl_point": "SP_POINT"}
AUTHOR_DATASET = {"clean1": "Clean1", "banknote": "Banknote", "haberman": "Haberman", "new_thyroid": "New-Thyroid"}
AUTHOR_METRIC = {"train_loss": "loss", "val_loss": "val_loss", "val_accuracy": "val_accuracy"}


def collect_author(author_dir: Path, dataset: str, strategy: str, metric: str):
    """history của Keras (model.fit) — chính là đại lượng tác giả vẽ trong Fig. 3."""
    d = author_dir / AUTHOR_DATASET.get(dataset, dataset) / AUTHOR_STRATEGY.get(strategy, strategy)
    runs = sorted(d.glob("run_*.json"))
    if not runs:
        return None
    ys = [json.loads(p.read_text(encoding="utf-8"))["history"][AUTHOR_METRIC.get(metric, metric)] for p in runs]
    n = min(len(y) for y in ys)
    return np.arange(1, n + 1), np.array([y[:n] for y in ys], dtype=float)


def plot_panel(ax, panel: dict, cfg: dict) -> None:
    source = panel.get("source", cfg.get("source", "ours"))   # ours | author
    runs_dir = resolve(panel.get("runs_dir", cfg["runs_dir"]))
    author_dir = resolve(cfg.get("author_dir", "outputs/author_rerun"))
    metric = cfg.get("metric", "train_loss")
    lw = cfg.get("linewidth", 1.5)
    band = cfg.get("band", False)
    series = []  # (epoch, mean, std, style dict)
    for s in panel["strategies"]:
        if source == "author":
            got = collect_author(author_dir, panel["dataset"], s["id"], metric)
        else:
            got = collect(runs_dir, panel["dataset"], s["id"], metric, cfg.get("seeds"))
        if got is None:
            print(f"[skip] {panel['dataset']} / {s['id']}: không có run ({source})")
            continue
        x, y = got
        sd = y.std(0, ddof=1) if y.shape[0] > 1 else np.zeros(y.shape[1])
        series.append((x, y.mean(0), sd, s))

    def draw(a, with_label: bool) -> None:
        for x, mu, sd, s in series:
            color = s.get("color")
            if band:
                a.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
            a.plot(x, mu, color=color, ls=s.get("linestyle", "solid"),
                   lw=s.get("linewidth", lw), label=s.get("label", s["id"]) if with_label else None)

    draw(ax, True)
    ax.set_title(panel.get("title", f"Error loss for {panel['dataset']}"), fontsize=13)
    ax.set_xlabel(cfg.get("xlabel", "Epoch"))
    ax.set_ylabel(cfg.get("ylabel", "Loss"))
    if series:
        ax.set_xlim(left=series[0][0][0] - 4)
        ax.legend(loc=panel.get("legend_loc", "upper right"), fontsize=10)

    ins = {**DEFAULT_INSET, **(cfg.get("inset") or {}), **(panel.get("inset") or {})}
    if not (ins["enabled"] and series):
        return
    x0, x1 = ins["xlim"]
    ylim = ins["ylim"]
    if ylim is None:  # suy ra từ dữ liệu trong khoảng zoom
        vals = np.concatenate([mu[(x >= x0) & (x <= x1)] for x, mu, _, _ in series])
        pad = 0.08 * (vals.max() - vals.min() or 1e-6)
        ylim = (vals.min() - pad, vals.max() + pad)
    axins = ax.inset_axes(ins["rect"])
    draw(axins, False)
    axins.set_xlim(x0, x1)
    axins.set_ylim(*ylim)
    axins.tick_params(labelsize=9)
    if ins["connectors"]:
        l1, l2 = ins["connectors"]
        mark_inset(ax, axins, loc1=l1, loc2=l2, fc="none", ec="0.7", lw=0.8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="file YAML/JSON, vd configs/plot_error_loss.yaml")
    ap.add_argument("--output", help="ghi đè đường dẫn ảnh đầu ra")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    panels = cfg["panels"]
    ncols = cfg.get("ncols", len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(cfg.get("figsize", [7.5 * ncols, 5.5 * nrows])),
                             dpi=cfg.get("dpi", 150), squeeze=False, facecolor="white")
    for ax, panel in zip(axes.ravel(), panels):
        plot_panel(ax, panel, cfg)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    if cfg.get("suptitle"):
        fig.suptitle(cfg["suptitle"], fontsize=13)
    fig.tight_layout()
    out = resolve(args.output or cfg.get("output", "outputs/figures/r1_error_loss.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
