"""Vẽ confusion matrix (kiểu "Confusion Matrix for <dataset> / <chiến lược>") theo file config.

Dùng:  python tools/plot/plot_confusion_matrix.py configs/plot_confusion_matrix.yaml
       python tools/plot/plot_confusion_matrix.py configs/plot_confusion_matrix.yaml --output-dir outputs/figures/cm_x
Đầu vào : <runs_dir>/<dataset>_<strategy>_s<seed>/metrics.json  → "test_confusion_matrix" (hàng = nhãn thật,
          cột = nhãn dự đoán; mỗi run đánh giá test đúng một lần).
Đầu ra  : layout "separate": một ảnh cho mỗi panel  <output_dir>/cm_<dataset>_<strategy>_<select>.png
          layout "grid"    : một ảnh ghép tất cả panel  → config["output"]

Chọn run nào để vẽ (`select`, đặt chung hoặc ghi đè trong từng panel):
  best      run có test_accuracy cao nhất (hoà thì lấy seed nhỏ nhất) — giống cách code tác giả chọn "best run" để vẽ
  seed:<k>  đúng seed k, ví dụ  seed:0
  sum       cộng ma trận của tất cả seed (hoặc các seed trong `seeds`)
`normalize` (tuỳ chọn): null = số mẫu | true = chuẩn hoá theo hàng (recall) | pred = theo cột | all = theo tổng.
Chỉ đọc kết quả của nhóm; kết quả chạy lại code tác giả không lưu confusion matrix nên không vẽ được ở đây.
Xem configs/plot_confusion_matrix.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[2]  # project root (this file lives in tools/plot/)

STRATEGY_LABEL = {  # tên như trong paper / config vẽ loss
    "no_curriculum": "No Curriculum",
    "ddcl_density": "Curriculum Learning - Density",
    "ddcl_point": "Curriculum Learning - Point",
    "self_paced": "Self-Paced Learning",
    "sp_ddcl_density": "Self-Paced DDCL - Density",
    "sp_ddcl_point": "Self-Paced DDCL - Point",
}
DATASET_LABEL = {
    "breast_cancer_diagnostic": "Breast Cancer Wisconsin (Diagnostic)",
    "cancer": "Cancer",
    "clean1": "Clean1",
    "banknote": "Banknote",
    "haberman": "Haberman's Survival",
    "liver": "Liver Disorder",
    "pima": "Pima Indians Diabetes",
    "new_thyroid": "New-Thyroid",
    "isolet": "ISOLET",
    "optical_digits": "Optical Recognition Handwritten Digits",
}


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def load_runs(runs_dir: Path, dataset: str, strategy: str, seeds) -> list[tuple[int, float, np.ndarray]]:
    """Trả list (seed, test_accuracy, confusion_matrix) sắp theo seed."""
    wanted = None if seeds is None else {int(s) for s in seeds}
    out = []
    for d in runs_dir.glob(f"{dataset}_{strategy}_s*"):
        tail = d.name.rsplit("_s", 1)[-1]
        mp = d / "metrics.json"
        if not (tail.isdigit() and mp.is_file()):
            continue
        seed = int(tail)
        if wanted is not None and seed not in wanted:
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if "test_confusion_matrix" in m:
            out.append((seed, float(m["test_accuracy"]), np.asarray(m["test_confusion_matrix"], dtype=float)))
    return sorted(out, key=lambda r: r[0])


def pick(runs: list, select: str) -> tuple[np.ndarray, str, str]:
    """Trả (ma trận, mô tả để in log, hậu tố tên file)."""
    if select == "best":
        seed, acc, cm = max(runs, key=lambda r: (r[1], -r[0]))
        return cm, f"best run: seed {seed}, acc={acc:.4f}", "best"
    if select.startswith("seed:"):
        k = int(select.split(":", 1)[1])
        for seed, acc, cm in runs:
            if seed == k:
                return cm, f"seed {k}, acc={acc:.4f}", f"seed{k}"
        raise KeyError(f"không có seed {k}")
    if select == "sum":
        return sum(r[2] for r in runs), f"tổng {len(runs)} seed", "sum"
    raise ValueError(f"select không hợp lệ: {select!r} (best | seed:<k> | sum)")


def normalize_cm(cm: np.ndarray, mode) -> np.ndarray:
    if mode in (None, False, "none"):
        return cm
    axis = {True: 1, "true": 1, "pred": 0}.get(mode)
    if mode == "all":
        return cm / cm.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cm / cm.sum(axis=axis, keepdims=True)
    return np.nan_to_num(out)


def draw(ax, cm: np.ndarray, title: str, cfg: dict, panel: dict) -> None:
    mode = panel.get("normalize", cfg.get("normalize"))
    shown = normalize_cm(cm, mode)
    n = cm.shape[0]
    fmt = "d" if mode in (None, False, "none") else ".2f"
    # cỡ chữ trong ô: nhỏ dần khi nhiều lớp (ISOLET 26 lớp)
    size = panel.get("annot_fontsize", cfg.get("annot_fontsize")) or (10 if n <= 10 else 7 if n <= 15 else 6.5)
    disp = ConfusionMatrixDisplay(shown if fmt == ".2f" else shown.astype(int))
    disp.plot(ax=ax, cmap=panel.get("cmap", cfg.get("cmap", "viridis")), values_format=fmt,
              colorbar=bool(cfg.get("colorbar", True)), xticks_rotation="horizontal")
    for t in disp.text_.ravel():
        t.set_fontsize(size)
    ax.set_title(title, fontsize=cfg.get("title_fontsize", 11))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")


def panel_title(panel: dict) -> str:
    if "title" in panel:
        return panel["title"]
    ds = panel.get("dataset_label") or DATASET_LABEL.get(panel["dataset"], panel["dataset"])
    st = panel.get("strategy_label") or STRATEGY_LABEL.get(panel["strategy"], panel["strategy"])
    return f"Confusion Matrix for {ds}\n{st}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None, help="ghi đè output_dir của config (layout separate)")
    ap.add_argument("--output", type=Path, default=None, help="ghi đè output của config (layout grid)")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    runs_dir = resolve(cfg["runs_dir"])
    layout = cfg.get("layout", "separate")
    dpi = int(cfg.get("dpi", 150))
    panels = cfg["panels"]

    # chuẩn bị dữ liệu từng panel
    prepared = []
    for p in panels:
        runs = load_runs(resolve(p.get("runs_dir", cfg["runs_dir"])), p["dataset"], p["strategy"],
                         p.get("seeds", cfg.get("seeds")))
        if not runs:
            print(f"[skip] {p['dataset']} / {p['strategy']}: không có run")
            continue
        cm, desc, suffix = pick(runs, p.get("select", cfg.get("select", "best")))
        print(f"[ok]   {p['dataset']} / {p['strategy']}: {desc}  (n_classes={cm.shape[0]}, n_test={int(cm.sum())})")
        prepared.append((p, cm, suffix))
    if not prepared:
        raise SystemExit("[-] Không có panel nào vẽ được.")

    if layout == "grid":
        ncols = int(cfg.get("ncols", 2))
        nrows = -(-len(prepared) // ncols)
        w, h = cfg.get("figsize", [7 * ncols, 6.5 * nrows])
        fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)
        for ax, (p, cm, _) in zip(axes.ravel(), prepared):
            draw(ax, cm, panel_title(p), cfg, p)
        for ax in axes.ravel()[len(prepared):]:
            ax.axis("off")
        fig.tight_layout()
        out = resolve(args.output or cfg.get("output", "outputs/figures/confusion_matrices.png"))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[+] {out}")
        return

    out_dir = resolve(args.output_dir or cfg.get("output_dir", "outputs/figures/confusion_matrices"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, cm, suffix in prepared:
        n = cm.shape[0]
        w, h = p.get("figsize", cfg.get("figsize_large" if n > 15 else "figsize_small", [7, 6] if n > 15 else [6, 5]))
        fig, ax = plt.subplots(figsize=(w, h))
        draw(ax, cm, panel_title(p), cfg, p)
        fig.tight_layout()
        out = out_dir / f"cm_{p['dataset']}_{p['strategy']}_{suffix}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[+] {out}")


if __name__ == "__main__":
    main()
