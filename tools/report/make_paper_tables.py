"""Vẽ ảnh Table 1 và Table 2 (bố cục như paper Dynamic DDCL) từ kết quả các lần chạy của nhóm.

Mỗi ô: Worst % / Best % / Average ±σ của test_accuracy trên các seed (mặc định std mẫu, ddof=1).
Hàng có Average cao nhất trong từng dataset được in đậm (giống paper); nếu có đồng hạng ở mức hiển thị
thì không in đậm hàng nào (paper làm vậy với Breast Cancer).

  Table 1: Breast Cancer Wisconsin (Diagnostic), Cancer, Haberman's Survival, Liver Disorder,
           Pima Indians Diabetes, New-Thyroid
  Table 2: ISOLET, Optical Recognition Handwritten Digits, Clean1, Banknote

Cách chạy (từ thư mục gốc dự án):
    python tools/report/make_paper_tables.py
    python tools/report/make_paper_tables.py --runs-dir outputs/single_model
    python tools/report/make_paper_tables.py --runs-dir outputs/runs_final_class_blocks_author_arch --dpi 300
    python tools/report/make_paper_tables.py --runs-dir outputs/runs_ablation_droplast_smote/droplast --tag droplast

Ra: <out-dir>/table1_<tag>.png và table2_<tag>.png (mặc định out-dir = outputs/figures, tag = tên thư mục runs).
Dataset nào chưa có run sẽ hiện "n/a"; chỉ cần có ít nhất một dataset của bảng đó thì bảng vẫn được vẽ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STRATEGIES = [
    ("no_curriculum", "No Curriculum"),
    ("ddcl_density", "DDCL-Density"),
    ("ddcl_point", "DDCL-Point"),
    ("self_paced", "Self-Paced"),
    ("sp_ddcl_density", "SP DDCL-Density"),
    ("sp_ddcl_point", "SP DDCL-Point"),
]

# (dataset_id, nhãn hiển thị (xuống dòng bằng \n), nhãn Self-Paced riêng nếu có)
TABLES = {
    1: [
        ("breast_cancer_diagnostic", "Breast Cancer\nWisconsin\n(Diagnostic)", "Self-Paced (SP)"),
        ("cancer", "Cancer", None),
        ("haberman", "Haberman's Survival", None),
        ("liver", "Liver Disorder", None),
        ("pima", "Pima Indians Diabetes", None),
        ("new_thyroid", "New-Thyroid", None),
    ],
    2: [
        ("isolet", "ISOLET", None),
        ("optical_digits", "Optical Recognition\nHandwritten Digits", None),
        ("clean1", "Clean1", None),
        ("banknote", "Banknote", None),
    ],
}


def load_accuracies(runs_dir: Path, dataset: str, strategy: str) -> np.ndarray:
    accs = []
    for p in sorted(runs_dir.glob(f"{dataset}_{strategy}_s*/metrics.json")):
        try:
            accs.append(float(json.loads(p.read_text(encoding="utf-8"))["test_accuracy"]) * 100.0)
        except (KeyError, ValueError, json.JSONDecodeError):
            print(f"[!] bỏ qua file lỗi: {p}", file=sys.stderr)
    return np.asarray(accs)


def compute_block(runs_dir: Path, dataset: str, ddof: int, decimals: int):
    """Trả về list 6 dict {worst,best,mean,std,n} (None nếu chưa có run)."""
    rows = []
    for sid, _ in STRATEGIES:
        a = load_accuracies(runs_dir, dataset, sid)
        if a.size == 0:
            rows.append(None)
            continue
        std = float(a.std(ddof=ddof)) if a.size > ddof else float("nan")
        rows.append({"worst": a.min(), "best": a.max(), "mean": a.mean(), "std": std, "n": int(a.size)})
    # in đậm nếu có đúng một hàng cao nhất ở mức hiển thị
    means = [round(r["mean"], decimals) if r else -1 for r in rows]
    top = max(means)
    winners = [i for i, m in enumerate(means) if m == top and top >= 0]
    bold = winners[0] if len(winners) == 1 else None
    return rows, bold


def draw_table(table_no: int, runs_dir: Path, out_path: Path, ddof: int, decimals: int, dpi: int, font: str) -> None:
    spec = TABLES[table_no]
    row_h, pad_top, pad_bot, block_gap = 0.215, 0.12, 0.10, 0.07
    n_blocks = len(spec)
    header_h = 0.34
    height = pad_top + header_h + n_blocks * (6 * row_h + block_gap) + pad_bot
    width = 6.4

    # vị trí cột (inch)
    x_ds, x_sc, x_w, x_b, x_a = 0.10, 1.88, 3.50, 4.45, 5.38
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [font, "STIXGeneral", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    fs = 9.2

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    y = pad_top
    ax.plot([0.02, width - 0.02], [y, y], color="black", lw=1.6)  # đường đầu bảng
    y_mid = y + header_h / 2
    ax.text(x_ds, y_mid, "Dataset", va="center", ha="left", fontsize=fs)
    ax.text(x_sc, y_mid, "Test Scenario", va="center", ha="left", fontsize=fs)
    ax.text(x_w, y_mid, "Worst %", va="center", ha="left", fontsize=fs)
    ax.text(x_b, y_mid, "Best %", va="center", ha="left", fontsize=fs)
    ax.text(x_a, y_mid, r"Average $\pm\sigma$", va="center", ha="left", fontsize=fs)
    y += header_h
    ax.plot([0.02, width - 0.02], [y, y], color="black", lw=0.8)  # dưới tiêu đề
    y += block_gap / 2

    for bi, (ds, label, sp_label) in enumerate(spec):
        rows, bold = compute_block(runs_dir, ds, ddof, decimals)
        y0 = y
        for i, ((sid, sname), r) in enumerate(zip(STRATEGIES, rows)):
            yc = y0 + (i + 0.5) * row_h
            if i % 2 == 0:  # nền xám nhạt ở hàng 1, 3, 5
                ax.add_patch(plt.Rectangle((x_sc - 0.08, y0 + i * row_h), width - x_sc - 0.04, row_h,
                                           facecolor="#efefef", edgecolor="none", zorder=0))
            weight = "bold" if bold == i else "normal"
            name = sp_label if (sid == "self_paced" and sp_label) else sname
            ax.text(x_sc, yc, name, va="center", ha="left", fontsize=fs, fontweight=weight)
            if r is None:
                for x in (x_w, x_b, x_a):
                    ax.text(x, yc, "n/a", va="center", ha="left", fontsize=fs, color="#888888")
                continue
            std_txt = f"{r['std']:.{decimals}f}" if np.isfinite(r["std"]) else "n/a"
            ax.text(x_w, yc, f"{r['worst']:.{decimals}f}", va="center", ha="left", fontsize=fs, fontweight=weight)
            ax.text(x_b, yc, f"{r['best']:.{decimals}f}", va="center", ha="left", fontsize=fs, fontweight=weight)
            ax.text(x_a, yc, f"{r['mean']:.{decimals}f} ±{std_txt}", va="center", ha="left",
                    fontsize=fs, fontweight=weight)
        # nhãn dataset căn giữa theo chiều dọc của khối
        ax.text(x_ds, y0 + 3 * row_h, label, va="center", ha="left", fontsize=fs, linespacing=1.25)
        y = y0 + 6 * row_h + block_gap
        is_last = bi == n_blocks - 1
        ax.plot([0.02, width - 0.02], [y - block_gap / 2, y - block_gap / 2],
                color="black", lw=1.6 if is_last else 0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[+] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("outputs/single_model"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    ap.add_argument("--tag", default=None, help="hậu tố tên file (mặc định: tên thư mục runs)")
    ap.add_argument("--tables", default="1,2", help="bảng cần vẽ: 1, 2 hoặc 1,2")
    ap.add_argument("--ddof", type=int, choices=(0, 1), default=1, help="std: 1=mẫu (mặc định), 0=tổng thể")
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--font", default="Times New Roman", help="font serif ưu tiên (mặc định rơi về STIX/DejaVu Serif)")
    args = ap.parse_args()

    if not args.runs_dir.is_dir():
        sys.exit(f"[-] Không thấy thư mục runs: {args.runs_dir}")
    tag = args.tag or args.runs_dir.name
    for t in (int(x) for x in args.tables.split(",")):
        draw_table(t, args.runs_dir, args.out_dir / f"table{t}_{tag}.png", args.ddof, args.decimals, args.dpi, args.font)


if __name__ == "__main__":
    main()
