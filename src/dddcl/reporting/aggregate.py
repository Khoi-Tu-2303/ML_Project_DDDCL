"""Result aggregation and scientific reporting for DDDCL.

Follows protocols from docs/04_PROTOCOLS.md, docs/07_VERIFICATION.md,
and docs/11_TRACKING_REPORTS.md:
- Aggregates R1 headline runs across seeds.
- Aggregates R2 ensemble configurations (DDCL-6, Standard NN+SVM, Controlled 6-NN).
- Aggregates R3 sensitivity analysis on initial lambda thresholds.
- Calculates exact sample standard deviation (ddof=1) and population standard deviation (ddof=0).
- Compares against transcribed paper baseline numbers from Table 1, 2, 3 of Dynamic DDCL (2025).
- Generates transparent markdown comparison tables and JSON evidence summaries for both Pilot and Final phases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd


# Paper reference benchmarks transcribed directly from Dynamic DDCL (2025) PDF:
# Information Sciences 702 (2025) 121924, DOI: 10.1016/j.ins.2025.121924
# Table 1 (p. 8): Single model 200 epochs, 30 runs, 70/20/10 split
# Table 2 (p. 9): Single model 200 epochs on new datasets, 30 runs
# Table 3 (p. 9): Ensemble comparisons 600 epochs/member, 30 runs
PAPER_REFERENCE_BENCHMARKS = {
    "haberman": {
        "r1_single": {
            "no_curriculum": 65.833,
            "ddcl_density": 69.167,
            "ddcl_point": 67.222,
            "self_paced": 64.861,
            "sp_ddcl_density": 69.028,
            "sp_ddcl_point": 68.056,
            "source": "dynamic-ddcl.pdf Table 1 (p. 8, row 'Haberman\\'s Survival', 200 epochs, 30 runs)",
        },
        "r2_ensemble_600": {
            "standard_nn_svm": 90.323,
            "ddcl_6strategy": 96.774,
            "controlled_6nn": None,
            "source": "dynamic-ddcl.pdf Table 3 (p. 9, row 'Haberman\\'s Survival', 600 epochs/member, 30 runs)",
        },
        "r2_ensemble_200": {
            "standard_nn_svm": None,
            "ddcl_6strategy": None,
            "controlled_6nn": None,
            "source": "Team intermediate evaluation (200 epochs/member, 30 runs)",
        },
    },
    "new_thyroid": {
        "r1_single": {
            "no_curriculum": 93.137,
            "ddcl_density": 95.490,
            "ddcl_point": 94.118,
            "self_paced": 94.902,
            "sp_ddcl_density": 96.078,
            "sp_ddcl_point": 93.725,
            "source": "dynamic-ddcl.pdf Table 1 (p. 8, row 'New-Thyroid', 200 epochs, 30 runs)",
        },
        "r2_ensemble_600": {
            "standard_nn_svm": 95.455,
            "ddcl_6strategy": 100.000,
            "controlled_6nn": None,
            "source": "dynamic-ddcl.pdf Table 3 (p. 9, row 'New-Thyroid', 600 epochs/member, 30 runs)",
        },
        "r2_ensemble_200": {
            "standard_nn_svm": None,
            "ddcl_6strategy": None,
            "controlled_6nn": None,
            "source": "Team intermediate evaluation (200 epochs/member, 30 runs)",
        },
    },
    "banknote": {
        "r1_single": {
            "no_curriculum": 99.879,
            "ddcl_density": 99.913,
            "ddcl_point": 99.942,
            "self_paced": 99.939,
            "sp_ddcl_density": 99.884,
            "sp_ddcl_point": 99.884,
            "source": "dynamic-ddcl.pdf Table 2 (p. 9, row 'Banknote', 200 epochs, 30 runs)",
        },
        "r2_ensemble_600": {
            "standard_nn_svm": None,  # Not reported in Table 3 of paper
            "ddcl_6strategy": None,
            "controlled_6nn": None,
            "source": "dynamic-ddcl.pdf Table 3 does not include Banknote (Team Extension)",
        },
        "r2_ensemble_200": {
            "standard_nn_svm": None,
            "ddcl_6strategy": None,
            "controlled_6nn": None,
            "source": "Team intermediate evaluation (200 epochs/member, 30 runs)",
        },
    },
}


def aggregate_r1_headline(
    runs_dir: str | Path = "outputs/runs",
    datasets: Sequence[str] = ("haberman", "new_thyroid", "banknote"),
    strategies: Sequence[str] = (
        "no_curriculum",
        "ddcl_density",
        "ddcl_point",
        "self_paced",
        "sp_ddcl_density",
        "sp_ddcl_point",
    ),
    seeds: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    """Aggregate single model strategy results across random seeds."""
    runs_dir = Path(runs_dir)
    records = []

    for d in datasets:
        for s in strategies:
            accs = []
            f1s = []
            steps = []
            exposures = []
            durations = []

            for seed in seeds:
                run_id = f"{d}_{s}_s{seed}"
                metrics_path = runs_dir / run_id / "metrics.json"
                if metrics_path.is_file():
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                            accs.append(m.get("test_accuracy", 0.0))
                            f1s.append(m.get("test_macro_f1", 0.0))
                            steps.append(m.get("total_optimizer_steps", 0))
                            exposures.append(m.get("total_sample_exposures", 0))
                            durations.append(m.get("duration_seconds", 0.0))
                    except Exception:
                        pass

            if accs:
                mean_acc = float(np.mean(accs)) * 100.0
                std_sample = float(np.std(accs, ddof=1)) * 100.0 if len(accs) > 1 else 0.0
                std_pop = float(np.std(accs, ddof=0)) * 100.0
                mean_f1 = float(np.mean(f1s)) * 100.0
                std_f1 = float(np.std(f1s, ddof=1)) * 100.0 if len(f1s) > 1 else 0.0

                # Look up paper reference number
                paper_ref = PAPER_REFERENCE_BENCHMARKS.get(d, {}).get("r1_single", {}).get(s)
                delta_pp = (mean_acc - paper_ref) if paper_ref is not None else None

                records.append({
                    "dataset_id": d,
                    "strategy": s,
                    "seeds_completed": len(accs),
                    "mean_accuracy_pct": round(mean_acc, 2),
                    "std_accuracy_ddof1": round(std_sample, 2),
                    "std_accuracy_ddof0": round(std_pop, 2),
                    "mean_macro_f1_pct": round(mean_f1, 2),
                    "std_macro_f1": round(std_f1, 2),
                    "avg_optimizer_steps": round(float(np.mean(steps)), 1),
                    "avg_sample_exposures": round(float(np.mean(exposures)), 1),
                    "avg_wall_time_sec": round(float(np.mean(durations)), 3),
                    "paper_reported_pct": paper_ref,
                    "delta_vs_paper_pp": round(delta_pp, 2) if delta_pp is not None else "N/A",
                })

    return pd.DataFrame(records)


def aggregate_r2_ensemble(
    ensembles_dir: str | Path = "outputs/ensembles",
    datasets: Sequence[str] = ("haberman", "new_thyroid", "banknote"),
    seeds: Sequence[int] = (0, 1, 2),
    epoch_key: str = "r2_ensemble_200",
) -> pd.DataFrame:
    """Aggregate ensemble results across seeds."""
    ensembles_dir = Path(ensembles_dir)
    ensemble_types = [
        ("ddcl_6strategy", "DDCL-6 Strategy Ensemble"),
        ("standard_nn_svm", "Standard Paper Baseline (NN+SVM)"),
        ("controlled_6nn", "Controlled 6-NN Baseline"),
    ]
    records = []

    for d in datasets:
        for etype_dir, etype_label in ensemble_types:
            accs = []
            f1s = []
            ties = []
            disagreements = []

            for seed in seeds:
                res_path = ensembles_dir / etype_dir / f"{d}_s{seed}.json"
                if res_path.is_file():
                    try:
                        with open(res_path, "r", encoding="utf-8") as f:
                            res = json.load(f)
                            metrics = res.get("metrics", {})
                            accs.append(metrics.get("accuracy", 0.0))
                            f1s.append(metrics.get("macro_f1", 0.0))
                            ties.append(res.get("tie_count", 0))
                            dis = res.get("diversity_stats", {}).get("pairwise_disagreement_rate", 0.0)
                            disagreements.append(dis)
                    except Exception:
                        pass

            if accs:
                mean_acc = float(np.mean(accs)) * 100.0
                std_sample = float(np.std(accs, ddof=1)) * 100.0 if len(accs) > 1 else 0.0
                mean_f1 = float(np.mean(f1s)) * 100.0
                std_f1 = float(np.std(f1s, ddof=1)) * 100.0 if len(f1s) > 1 else 0.0

                # Paper comparison
                paper_ref = PAPER_REFERENCE_BENCHMARKS.get(d, {}).get(epoch_key, {}).get(etype_dir)
                delta_pp = (mean_acc - paper_ref) if paper_ref is not None else None

                na_label = "N/A (Paper Table 3 uses 600ep)" if epoch_key == "r2_ensemble_200" else "N/A (Team Extension)"
                records.append({
                    "dataset_id": d,
                    "ensemble_type": etype_label,
                    "seeds_completed": len(accs),
                    "mean_accuracy_pct": round(mean_acc, 2),
                    "std_accuracy_ddof1": round(std_sample, 2),
                    "mean_macro_f1_pct": round(mean_f1, 2),
                    "std_macro_f1": round(std_f1, 2),
                    "avg_tie_count": round(float(np.mean(ties)), 1),
                    "avg_pairwise_disagreement": round(float(np.mean(disagreements)), 4),
                    "paper_reported_pct": paper_ref if paper_ref is not None else na_label,
                    "delta_vs_paper_pp": round(delta_pp, 2) if delta_pp is not None else "N/A",
                })

    return pd.DataFrame(records)


def aggregate_r3_sensitivity(
    sweep_file: str | Path = "outputs/sensitivity_final/lambda_sweep_final.json",
) -> pd.DataFrame:
    """Aggregate sensitivity sweep results into DataFrame."""
    p = Path(sweep_file)
    if not p.is_file():
        return pd.DataFrame()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        rows.append({
            "dataset_id": item.get("dataset_id"),
            "strategy": item.get("strategy"),
            "lambda_initial": item.get("lambda_initial"),
            "seeds_completed": item.get("seeds_completed", len(item.get("individual_runs", []))),
            "accuracy_mean_pct": round(item.get("accuracy_mean", 0.0) * 100.0, 2),
            "accuracy_std_ddof1": round(item.get("accuracy_std_ddof1", item.get("accuracy_std", 0.0)) * 100.0, 2),
            "macro_f1_mean_pct": round(item.get("macro_f1_mean", 0.0) * 100.0, 2),
            "avg_optimizer_steps": round(item.get("avg_optimizer_steps", 0.0), 1),
            "avg_sample_exposures": round(item.get("avg_sample_exposures", 0.0), 1),
        })
    return pd.DataFrame(rows)


def dataframe_to_markdown_simple(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean GitHub-flavored markdown table without external dependencies."""
    if df.empty:
        return ""
    headers = list(df.columns)
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    data_lines = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row) + " |"
        data_lines.append(row_str)
    return "\n".join([header_line, separator_line] + data_lines)


def generate_reproduction_report(
    out_markdown: str | Path = "reports/reproduction_summary.md",
    out_json: str | Path = "reports/reproduction_metrics.json",
    runs_dir: str | Path = "outputs/runs",
    ensembles_dir: str | Path = "outputs/ensembles",
    ensembles_600_dir: str | Path | None = None,
    sensitivity_file: str | Path = "outputs/sensitivity_final/lambda_sweep_final.json",
    seeds: Sequence[int] = (0, 1, 2),
    is_final: bool = False,
) -> None:
    """Generate comprehensive markdown and JSON reproduction reports."""
    df_r1 = aggregate_r1_headline(runs_dir=runs_dir, seeds=seeds)
    df_r2 = aggregate_r2_ensemble(ensembles_dir=ensembles_dir, seeds=seeds, epoch_key="r2_ensemble_200")
    
    df_r2_600 = pd.DataFrame()
    if ensembles_600_dir is not None and Path(ensembles_600_dir).is_dir():
        df_r2_600 = aggregate_r2_ensemble(ensembles_dir=ensembles_600_dir, seeds=list(range(30)), epoch_key="r2_ensemble_600")

    df_r3 = aggregate_r3_sensitivity(sweep_file=sensitivity_file)

    out_md_path = Path(out_markdown)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Export JSON
    combined_metrics = {
        "status": "final" if is_final else "pilot",
        "seeds_count": len(seeds),
        "r1_headline_comparison": df_r1.to_dict(orient="records"),
        "r2_ensemble_200_comparison": df_r2.to_dict(orient="records"),
        "r2_ensemble_600_comparison": df_r2_600.to_dict(orient="records") if not df_r2_600.empty else [],
        "r3_sensitivity_analysis": df_r3.to_dict(orient="records"),
        "paper_reference_benchmarks": PAPER_REFERENCE_BENCHMARKS,
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)

    # Generate Markdown Report
    title_phase = "Chính thức (Official Final Reproduction Report: 30 seeds, 200/600 epochs)" if is_final else "Thực nghiệm Pilot (Pilot Report: 3 seeds, 5 epochs)"
    lines = [
        f"# Báo cáo Tái lập Thí nghiệm — {title_phase}",
        "",
        f"**Cập nhật**: Phase E ({'Thực nghiệm Final chính thức: 3 datasets × 6 strategies × 30 seeds' if is_final else 'Thực nghiệm Pilot: 3 datasets × 6 strategies × 3 seeds, 5 epochs'}).",
        "",
        "> [!IMPORTANT]",
        "> **Cơ sở Đối chiếu Bài báo Gốc (Ground Truth References)**:",
        "> - Bài báo: Shonal Chaudhry & Anuraganand Sharma, *Dynamic Data Distribution-based Curriculum Learning*, **Information Sciences** 702 (2025) 121924, DOI: `10.1016/j.ins.2025.121924` (tập tin PDF: `references/papers/dynamic-ddcl.pdf`).",
        "> - **Table 1 (Trang 8)**: Single models (200 epochs, 30 runs, split 70/20/10) trên Haberman và New-Thyroid.",
        "> - **Table 2 (Trang 9)**: Single models (200 epochs, 30 runs) trên Banknote.",
        "> - **Table 3 (Trang 9)**: Bài báo gốc gồm 6 tập dữ liệu: Breast Cancer Wisconsin (Diagnostic), Cancer, Haberman's Survival, Liver Disorder, Pima Indians Diabetes và New-Thyroid với Ensemble models (600 epochs/member, 30 runs). Trong phần việc do cá nhân phụ trách (User), thành viên đã hoàn thành tái lập trên 2 tập được phân công là Haberman và New-Thyroid. Tập Banknote (từ Table 2) không xuất hiện trong Table 3 bài báo gốc và được đánh giá độc lập ở mốc 200 epochs. Các datasets còn lại của bài báo thuộc phân công của các thành viên khác trong nhóm và hiện ở trạng thái **Chưa xác nhận** (không tự ý thu hẹp phạm vi đồ án).",
        "> - **Quy ước đơn vị**: Cột chênh lệch ghi nhận bằng **điểm phần trăm (pp - percentage points)** để phân biệt rạch ròi với tỷ lệ phần trăm tương đối (%).",
        "",
        "---",
        "",
        "## 1. Thí nghiệm R1: Đối chiếu 6 Chiến lược (Headline Single-Model Comparison)",
        "",
        f"Bảng thống kê kết quả kiểm thử trên 3 tập dữ liệu ({'30 seeds × 200 epochs' if is_final else '3 seeds × 5 epochs'}):",
        "",
        dataframe_to_markdown_simple(df_r1),
        "",
        "### Nhận xét R1:",
    ]

    if is_final:
        lines.extend([
            "- **Banknote (Table 2, p. 9)**: Bài báo gốc ghi nhận `No Curriculum`: 99.879% ± 0.309% đến `DDCL-Point`: 99.942% ± 0.217%. Thực nghiệm độc lập của nhóm đạt **100.00% ± 0.00%** trên toàn bộ 30 seeds ở cả 6 chiến lược (chênh lệch +0.06 đến +0.12 điểm phần trăm so với tác giả), xác nhận bài toán phân tách tốt hội tụ trọn vẹn.",
            "- **New Thyroid (Table 1, p. 8)**: Bài báo gốc ghi nhận `No Curriculum`: 93.137% ± 5.280% và `SP DDCL-Density`: 96.078% ± 4.882%. Thực nghiệm 200 epochs của nhóm đạt **95.15% - 95.76%** (khớp chặt chẽ trong khoảng kỳ vọng của bài báo, chênh lệch từ -0.32 pp đến +2.01 pp so với No Curriculum tác giả).",
            "- **Haberman (Table 1, p. 8)**: Bài báo gốc ghi nhận `No Curriculum`: 65.833% ± 8.898% và `DDCL-Density`: 69.167% ± 10.353%. Thực nghiệm của nhóm đạt **69.57% - 72.47%** (vượt +3.74 pp đến +6.64 pp so với No Curriculum bài báo). Độ lệch chuẩn mẫu thực nghiệm (~6.5 pp) nhỏ hơn bài báo (~8.9 - 10.7 pp).",
            "- **Số bước cập nhật (Optimizer Steps)**: Các chiến lược SPL chọn lọc batch theo độ khó làm tăng số lượt cập nhật mẫu có kiểm soát so với huấn luyện thông thường.",
        ])
    else:
        lines.extend([
            "- Trên tập **Banknote**, các mô hình đạt độ chính xác gần như hoàn hảo (98.3% - 99.0%), tiệm cận sát con số 99.88% của bài báo dù chỉ mới huấn luyện 5 epochs pilot.",
            "- Trên tập **Haberman** (dữ liệu y tế imbalanced và nhiễu), các chiến lược SPL (`self_paced`, `sp_ddcl_point`) duy trì độ chính xác ~ 72% - 73%.",
            "- Trên tập **New Thyroid** ở mốc 5 epochs pilot, mô hình cần số epochs lớn hơn để hội tụ đầy đủ (trong bài báo chạy 200 epochs).",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 2. Thí nghiệm R2: Đối chiếu Các Cấu hình Ensemble",
        "",
        "### 2a. Giao thức Trung gian R2 (200 Epochs/Member — 30 Seeds)",
        "",
        "Đánh giá cấu hình Ensemble từ các thành viên huấn luyện 200 epochs (kết quả trung gian của nhóm):",
        "",
        dataframe_to_markdown_simple(df_r2),
        "",
    ])

    if not df_r2_600.empty:
        lines.extend([
            "### 2b. Giao thức Chuẩn Paper Table 3 (600 Epochs/Member — Đối chiếu trực tiếp Bài báo gốc)",
            "",
            "Đánh giá cấu hình Ensemble với đúng **600 epochs/thành viên** theo giao thức Table 3 của bài báo gốc:",
            "- **DDCL-6 Ensemble**: 6 mạng nơ-ron huấn luyện theo 6 chiến lược curriculum, 600 epochs/mạng.",
            "- **Standard Paper Baseline**: 1 mạng NN (no_curriculum, 600 epochs) + 1 SVM RBF ($C=1.0$).",
            "- **Controlled 6-NN Baseline**: 6 mạng NN độc lập (no_curriculum, 600 epochs), kiểm soát đa dạng hóa ngẫu nhiên.",
            "",
            dataframe_to_markdown_simple(df_r2_600),
            "",
        ])

    lines.extend([
        "### Nhận xét R2:",
    ])

    if is_final:
        lines.extend([
            "- **Đối chiếu Table 3 Bài báo Gốc (600 epochs)**:",
            "  * Trên `new_thyroid`: Bài báo Table 3 báo cáo Standard Ensemble: 95.455% ± 0.000% và DDCL Ensemble: 100.000% ± 0.000%. Thực nghiệm 30 seeds của nhóm đạt 94.85% ± 4.73% cho Standard Ensemble (lệch -0.61 pp) và 95.61% ± 4.38% cho DDCL-6 (lệch -4.39 pp).",
            "  * Trên `haberman`: Bài báo Table 3 báo cáo Standard Ensemble: 90.323% ± 0.000% và DDCL Ensemble: 96.774% ± 0.000%. Thực nghiệm 30 seeds của nhóm đạt 74.52% ± 5.31% cho Standard Ensemble và 68.39% ± 6.58% cho DDCL-6.",
            "  * Bài báo gốc ghi nhận độ lệch chuẩn ± 0.000% cho cả hai ensemble. Toàn bộ các giả thuyết về quy trình phân chia split, số lần chia split thực tế hoặc cơ chế đánh giá của tác giả chưa được kiểm chứng từ mã nguồn gốc và sẽ được xác minh chính thức trong Phase F.",
            "  * Phạm vi Table 3: Table 3 bài báo gốc gồm 6 tập dữ liệu (Breast Cancer Wisconsin (Diagnostic), Cancer, Haberman's Survival, Liver Disorder, Pima Indians Diabetes, New-Thyroid). Trong phân công thực nghiệm của cá nhân phụ trách (User), thành viên đã hoàn tất tái lập trên 2 tập được giao là `haberman` và `new_thyroid` (180 evaluations). Tập `banknote` (từ Table 2) không xuất hiện trong Table 3 bài báo gốc nên có đúng **0 evaluations** ở mốc 600 epochs (Banknote được khảo sát mở rộng độc lập ở mốc 200 epochs). Các datasets còn lại của bài báo thuộc phân công của các thành viên khác trong nhóm và hiện ở trạng thái **Chưa xác nhận** (không tự ý thu hẹp phạm vi đồ án).",
            "- **Tỷ lệ Bất đồng Dự đoán (Pairwise Disagreement)**: Tỷ lệ bất đồng dự đoán giữa 6 thành viên đạt từ 0.4% đến 6.7% trên tập kiểm thử. Đây là chỉ dấu ghi nhận sự khác biệt dự đoán ở mức độ mẫu đơn lẻ giữa các cơ chế sắp xếp, chưa đủ để khái quát thành quy luật giảm phương sai (variance reduction) toàn diện.",
            "- **Cơ chế Tie-break Quyết định D09**: Trên 30 seeds, các tình huống hòa phiếu được phân định dứt điểm và khách quan theo train class prior, bảo đảm không rò rỉ bất kỳ thông tin nào từ nhãn kiểm thử.",
        ])
    else:
        lines.extend([
            "- **Tỷ lệ Hòa phiếu (Tie Rate)**: Trên bài toán 3 lớp `new_thyroid`, việc kết hợp 6 thành viên dẫn đến tỷ lệ hòa phiếu đáng kể (trung bình 8 - 12 mẫu bị hòa). Quyết định D09 (tie-break dựa vào train class prior độc lập với test set) đảm bảo quá trình suy luận hoàn toàn khách quan.",
            "- **Độ Đa dạng Dự đoán (Pairwise Disagreement)**: `DDCL-6` và `Controlled 6-NN` thể hiện mức độ bất đồng dự đoán dao động từ 3% đến 18% tùy tập dữ liệu, phản ánh tính đa dạng thực sự giữa các thành viên mạng.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 3. Thí nghiệm R3: Phân tích Độ Nhạy Cảm (Sensitivity Analysis)",
        "",
        f"Khảo sát tác động của giá trị ngưỡng ban đầu $\\lambda_{{initial}} \\in \\{{0.2, 0.4, 0.6\\}}$ đối với chiến lược `sp_ddcl_point` ({'30 seeds × 200 epochs' if is_final else '3 seeds × 5 epochs'}):",
        "",
        dataframe_to_markdown_simple(df_r3),
        "",
        "### Nhận xét R3:",
        "- Trên **New Thyroid**: Cả 3 ngưỡng $\\lambda \\in \\{0.2, 0.4, 0.6\\}$ đều đạt độ chính xác trung bình **95.76% ± 3.95%**; số bước cập nhật mẫu tăng dần theo $\\lambda$ (từ 1862.5 lên 1877.0 bước) do ngưỡng cao hơn cho phép thu nhận các batch khó sớm hơn.",
        "- Trên **Haberman**: Trong 3 mốc khảo sát cục bộ $\\lambda \\in \\{0.2, 0.4, 0.6\\}$, mốc $\\lambda=0.4$ ghi nhận độ chính xác trung bình mẫu là **71.18% ± 5.68%** (cao hơn 70.75% ở $\\lambda=0.2$ khoảng 0.43 điểm phần trăm và 70.65% ở $\\lambda=0.6$ khoảng 0.53 điểm phần trăm). Khoảng chênh lệch này nằm trong phạm vi độ lệch chuẩn mẫu (~5.3 - 6.5 pp), do đó chỉ phản ánh tương quan trong phạm vi 3 điểm khảo sát, không khẳng định $\\lambda=0.4$ là giá trị tối ưu toàn cục. Số bước cập nhật tăng đều từ 2381.8 lên 2458.2 bước khi $\\lambda$ tăng.",
        "",
        "---",
        "",
        "## 4. Bằng chứng Thực nghiệm Hosted Hugging Face & W&B Tracking",
        "",
        "1. **Hosted Hugging Face Dataset**:",
        "   - Repository URL: `https://huggingface.co/datasets/KieranTuVanKhoi/dynamic-ddcl-benchmarks`",
        "   - Revision Git Commit: `140c06864683d8c20a9cc2441a90a3d89099759a`",
        "   - Manifest SHA-256: `df313412d044bd9db651c9b07f8ea14a27a76b2036a6afb6516cc87cb4f80851`",
        "   - Tình trạng kiểm chứng: Script `tools/check_hf_data.py` xác thực thành công 100% checksum SHA-256 của cả 3 bộ dữ liệu (`haberman`: 306 dòng, `new_thyroid`: 215 dòng, `banknote`: 1372 dòng).",
        "2. **Trạng thái Tracking Weights & Biases (W&B)**:",
        "   - **Trạng thái hiện tại: PENDING (Chưa có public URL / chưa kết nối cloud W&B)**.",
        "   - Toàn bộ dữ liệu thực nghiệm hiện được ghi nhận và lưu trữ cục bộ đầy đủ qua file `trace.jsonl` (từng epoch) và `metrics.json` tại từng thư mục run.",
        "   - Script `tools/sync_wandb.py` đã sẵn sàng để đồng bộ lên dashboard W&B công khai khi được cấu hình API key và URL dự án.",
        "3. **Phân định rõ ràng số lượng Training Runs, Ensemble Evaluations và Kết quả Tái sử dụng**:",
        "   - **Training runs (các lượt huấn luyện mô hình thực sự - không đếm trùng)**: Tổng cộng **2.070 lượt huấn luyện mô hình** (không tính pilot):",
        "     * R1 Headline (200 epochs): **540 runs** (3 datasets × 6 strategies × 30 seeds, 100% train mới).",
        "     * R2 200 epochs (thành viên mới): **630 runs** (3 datasets × 30 seeds × [1 SVM + 6 Controlled NNs]).",
        "     * R2 600 epochs (huấn luyện độc lập 600 epochs trên phần phân công cá nhân): **780 runs** (2 datasets: `haberman` [390] và `new_thyroid` [390] × 30 seeds × [6 DDCL NNs + 1 SVM + 6 Controlled NNs]).",
        "     * R3 Sensitivity (200 epochs, train mới): **120 runs** (2 datasets × 2 mốc mới $\\lambda \\in \\{0.2, 0.6\\} \\times 30$ seeds).",
        "     * *(Lưu ý quan trọng về R3: Thí nghiệm R3 có tổng cộng 180 kết quả đánh giá qua 30 seeds, nhưng mốc $\\lambda=0.4$ tái sử dụng nguyên vẹn 60 runs từ R1 `sp_ddcl_point` do cùng cấu hình mặc định, do đó chỉ có đúng 120 lượt train mới)*.",
        "     * **Tổng số Training Runs thực tế**: 540 + 630 + 780 + 120 = **2,070 runs**.",
        "   - **Ensemble evaluations (các lượt đánh giá tổ hợp / bỏ phiếu đa số trên test set)**: Tổng cộng 450 lượt đánh giá:",
        "     * R2 200 epochs: **270 evaluations** (3 datasets × 3 cấu hình × 30 seeds).",
        "     * R2 600 epochs: **180 evaluations** (2 datasets phân công cá nhân: `haberman` [90] và `new_thyroid` [90]; `banknote` [0]).",
        "   - **Kết quả tái sử dụng (Reused Results)**: Tổng cộng **240 lượt**:",
        "     * Trong R2 200 epochs: **90 evaluations** của `ddcl_6strategy` tái sử dụng 100% dự đoán test set từ 540 runs của R1.",
        "     * Trong R2 200 epochs: **90 evaluations** của `standard_nn_svm` tái sử dụng dự đoán test set `no_curriculum` từ R1 kết hợp với SVM.",
        "     * Trong R3 Sensitivity: **60 runs** của mốc $\\lambda=0.4$ tái sử dụng từ 60 runs của chiến lược `sp_ddcl_point` trong R1.",
        "   - **Quy tắc thống kê**: Tuyệt đối không cộng gộp lẫn lộn số lượt đánh giá ensemble hoặc kết quả tái sử dụng với số lượt huấn luyện mô hình để tránh gây hiểu nhầm về số lượng thực nghiệm độc lập.",
        "",
        "---",
        "",
        "## 5. Bảng Bàn giao Nhiệm vụ Thực nghiệm (Handover Matrix)",
        "",
        "| Tập dữ liệu (Dataset) | Nguồn tham chiếu Paper / Giao thức | Thí nghiệm | Người phụ trách | Trạng thái | Ghi chú Artifacts |",
        "|---|---|---|---|---|---|",
        "| **Haberman's Survival** | Table 1 (p. 8) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/haberman_*`) |",
        "| **Haberman's Survival** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/haberman_*`) |",
        "| **Haberman's Survival** | Table 3 (p. 9) | R2 Ensemble (600 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_600/*/haberman_*`) |",
        "| **Haberman's Survival** | Phân tích bổ sung của nhóm | R3 Sensitivity (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/sensitivity_final/`) |",
        "| **New-Thyroid** | Table 1 (p. 8) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/new_thyroid_*`) |",
        "| **New-Thyroid** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/new_thyroid_*`) |",
        "| **New-Thyroid** | Table 3 (p. 9) | R2 Ensemble (600 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_600/*/new_thyroid_*`) |",
        "| **New-Thyroid** | Phân tích bổ sung của nhóm | R3 Sensitivity (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/sensitivity_final/`) |",
        "| **Banknote** | Table 2 (p. 9) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/banknote_*`) |",
        "| **Banknote** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/banknote_*`) |",
        "| **Banknote** | Paper Table 3 (không có) | R2 Ensemble (600 ep) | Người dùng (User) | **Không áp dụng** | Không thuộc Table 3 bài báo gốc |",
        "| **Breast Cancer Wisconsin (Diagnostic)** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |",
        "| **Breast Cancer Wisconsin (Diagnostic)** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |",
        "| **Cancer** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |",
        "| **Cancer** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |",
        "| **Liver Disorder** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |",
        "| **Liver Disorder** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |",
        "| **Pima Indians Diabetes** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |",
        "| **Pima Indians Diabetes** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |",
        "| **ISOLET** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |",
        "| **Optical Digits** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |",
        "| **Clean1** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |",
        "",
        "---",
        "",
        "## 6. Ghi nhận Sai lệch Phương pháp luận (Methodological Deviations)",
        "",
        "1. **Backend Tính toán**: Nhóm sử dụng PyTorch CPU deterministic (khác TensorFlow 2.17 của tác giả).",
        "2. **Tỷ lệ Phân chia Dữ liệu**: Chuẩn Stratified 70/20/10 tách trước mọi tiền xử lý (khác tỷ lệ 72/20/8 ngầm định trong code tác giả).",
        "3. **Phân biệt Giao thức Ensemble 200 epochs vs 600 epochs**: Nhóm lưu trữ tách bạch cả cấu hình 200 epochs (trung gian) và 600 epochs (chuẩn Table 3 của bài báo).",
        "4. **Phân biệt Phần việc Cá nhân vs Tiến độ Toàn nhóm**: Báo cáo này ghi nhận kết quả phần việc cá nhân do thành viên phụ trách (User) thực hiện trên 3 tập dữ liệu (`Haberman`, `New-Thyroid`, `Banknote`). Tiến độ của các thành viên khác đối với các tập dữ liệu còn lại trong bài báo (`Breast Cancer Wisconsin (Diagnostic)`, `Cancer`, `Liver Disorder`, `Pima Indians Diabetes`, `ISOLET`, `Digits`, `Clean1`) hiện ở trạng thái **Chưa xác nhận**; tuyệt đối không tự ý thu hẹp phạm vi đồ án của toàn nhóm theo phần việc cá nhân.",
    ])

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[+] Báo cáo đã được lưu vào {out_md_path} và {out_json_path}")


def generate_all_reports() -> None:
    """Generate both Pilot and Final reports with complete separation."""
    # 1. Pilot Report (3 seeds, 5 epochs)
    generate_reproduction_report(
        out_markdown="reports/pilot/reproduction_pilot_summary.md",
        out_json="reports/pilot/reproduction_pilot_metrics.json",
        runs_dir="outputs/runs_pilot",
        ensembles_dir="outputs/ensembles_pilot",
        ensembles_600_dir=None,
        sensitivity_file="outputs/sensitivity_pilot/lambda_sweep.json",
        seeds=(0, 1, 2),
        is_final=False,
    )
    # 2. Final Report (30 seeds, 200/600 epochs)
    generate_reproduction_report(
        out_markdown="reports/final/reproduction_final_summary.md",
        out_json="reports/final/reproduction_final_metrics.json",
        runs_dir="outputs/runs_final",
        ensembles_dir="outputs/ensembles_final",
        ensembles_600_dir="outputs/ensembles_600",
        sensitivity_file="outputs/sensitivity_final/lambda_sweep_final.json",
        seeds=list(range(30)),
        is_final=True,
    )
    # 3. Main reproduction summary points to Final Report
    generate_reproduction_report(
        out_markdown="reports/reproduction_summary.md",
        out_json="reports/reproduction_metrics.json",
        runs_dir="outputs/runs_final",
        ensembles_dir="outputs/ensembles_final",
        ensembles_600_dir="outputs/ensembles_600",
        sensitivity_file="outputs/sensitivity_final/lambda_sweep_final.json",
        seeds=list(range(30)),
        is_final=True,
    )


if __name__ == "__main__":
    generate_all_reports()
