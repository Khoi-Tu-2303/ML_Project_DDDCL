"""Script to generate and execute notebooks/04_reproduction.ipynb.

Covers Phase E deliverables:
1. R1 Headline Single-Model Comparison (6 strategies across 3 pilot datasets).
2. R2 Ensemble Configurations (DDCL-6, Standard NN+SVM, Controlled 6-NN).
3. R3 Sensitivity Analysis on initial lambda values.
4. Visualizations: bar charts, diversity heatmaps, sensitivity curves.
5. Direct transcription and comparison against Dynamic DDCL (2025) paper benchmarks.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient


def create_reproduction_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "DDDCL Local",
            "language": "python",
            "name": "ddcl-local",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    }

    cells = []

    # Title & Markdown Overview
    cells.append(nbf.v4.new_markdown_cell(
        "# Notebook 04: Tái Lập Thí Nghiệm & Đánh Giá Ensemble (Phase E)\n\n"
        "Notebook này phân tích toàn bộ kết quả thực nghiệm của **Giai đoạn E (Reproduction + Ensemble)**:\n"
        "1. **Thí nghiệm R1 (Headline Comparison)**: Đối chiếu 6 chiến lược sắp xếp dữ liệu trên 3 tập dữ liệu pilot (`haberman`, `new_thyroid`, `banknote`).\n"
        "2. **Thí nghiệm R2 (Ensemble Comparison)**: So sánh hiệu năng giữa **DDCL-6 Ensemble**, **Standard Paper Baseline (NN+SVM)**, và **Controlled 6-NN Baseline**.\n"
        "3. **Thí nghiệm R3 (Sensitivity Analysis)**: Phân tích độ nhạy cảm của ngưỡng khởi đầu $\\lambda \\in \\{0.2, 0.4, 0.6\\}$ trong Self-Paced Learning.\n"
        "4. **Đối chiếu Bài Báo Gốc**: Phân tích sự tương quan và các điểm sai lệch phương pháp luận (Methodological Deviations)."
    ))

    # Cell 1: Imports & Setup
    cells.append(nbf.v4.new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from dddcl.reporting.aggregate import (\n"
        "    aggregate_r1_headline,\n"
        "    aggregate_r2_ensemble,\n"
        "    PAPER_REFERENCE_BENCHMARKS,\n"
        ")\n\n"
        "# Cấu hình thẩm mỹ biểu đồ\n"
        "sns.set_theme(style='whitegrid', font='sans-serif')\n"
        "plt.rcParams['font.size'] = 11\n"
        "print('Môi trường phân tích số liệu đã sẵn sàng!')"
    ))

    # Cell 2: Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Thí nghiệm R1: Đối chiếu 6 Chiến lược (Headline Single-Model Comparison)\n\n"
        "Chúng ta tổng hợp kết quả của 54 runs độc lập (3 datasets $\\times$ 6 strategies $\\times$ 3 seeds) "
        "từ thư mục `outputs/runs/`."
    ))

    # Cell 3: Code Section 1
    cells.append(nbf.v4.new_code_cell(
        "runs_dir = Path('../outputs/runs') if Path('../outputs/runs').exists() else Path('outputs/runs')\n"
        "df_r1 = aggregate_r1_headline(runs_dir=runs_dir)\n\n"
        "# Hiển thị bảng kết quả R1\n"
        "cols_show = [\n"
        "    'dataset_id', 'strategy', 'mean_accuracy_pct', 'std_accuracy_ddof1',\n"
        "    'mean_macro_f1_pct', 'avg_optimizer_steps', 'avg_sample_exposures',\n"
        "    'paper_reported_pct', 'delta_vs_paper_pct'\n"
        "]\n"
        "display(df_r1[cols_show])"
    ))

    # Cell 4: Markdown Section 1 Chart
    cells.append(nbf.v4.new_markdown_cell(
        "### Trực quan hóa Kết quả R1 theo Từng Tập Dữ liệu"
    ))

    # Cell 5: Code Section 1 Chart
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)\n"
        "datasets = ['haberman', 'new_thyroid', 'banknote']\n\n"
        "for idx, d in enumerate(datasets):\n"
        "    sub_df = df_r1[df_r1['dataset_id'] == d]\n"
        "    ax = axes[idx]\n"
        "    bars = ax.bar(\n"
        "        sub_df['strategy'],\n"
        "        sub_df['mean_accuracy_pct'],\n"
        "        yerr=sub_df['std_accuracy_ddof1'],\n"
        "        capsize=4,\n"
        "        color=sns.color_palette('muted', len(sub_df)),\n"
        "        alpha=0.85,\n"
        "    )\n"
        "    ax.set_title(f'Dataset: {d.upper()}', fontsize=13, fontweight='bold')\n"
        "    ax.set_ylabel('Độ chính xác Test trung bình (%)')\n"
        "    ax.set_xticklabels(sub_df['strategy'], rotation=45, ha='right')\n"
        "    # Thêm đường tham chiếu kết quả bài báo cho best method\n"
        "    best_paper = sub_df['paper_reported_pct'].max()\n"
        "    if pd.notna(best_paper):\n"
        "        ax.axhline(best_paper, color='red', linestyle='--', alpha=0.7, label=f'Paper Peak ({best_paper}%)')\n"
        "        ax.legend(loc='lower right')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 6: Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Thí nghiệm R2: Đối chiếu Các Cấu hình Ensemble\n\n"
        "Chúng ta đánh giá 3 cấu hình Ensemble:\n"
        "1. **DDCL-6 Strategy Ensemble**: Kết hợp 6 mô hình từ 6 chiến lược khác nhau.\n"
        "2. **Standard Paper Baseline (NN + SVM)**: Kết hợp 1 mạng Neural Network (không curriculum) và 1 mô hình SVM RBF ($C=1.0, \\gamma=\\text{scale}$).\n"
        "3. **Controlled 6-NN Baseline**: Kết hợp 6 mạng Neural Network cùng dùng `no_curriculum` với 6 seeds khởi tạo độc lập."
    ))

    # Cell 7: Code Section 2
    cells.append(nbf.v4.new_code_cell(
        "ensembles_dir = Path('../outputs/ensembles') if Path('../outputs/ensembles').exists() else Path('outputs/ensembles')\n"
        "df_r2 = aggregate_r2_ensemble(ensembles_dir=ensembles_dir)\n\n"
        "cols_r2 = [\n"
        "    'dataset_id', 'ensemble_type', 'mean_accuracy_pct', 'std_accuracy_ddof1',\n"
        "    'mean_macro_f1_pct', 'avg_tie_count', 'avg_pairwise_disagreement',\n"
        "    'paper_reported_pct', 'delta_vs_paper_pct'\n"
        "]\n"
        "display(df_r2[cols_r2])"
    ))

    # Cell 8: Markdown Section 2 Chart
    cells.append(nbf.v4.new_markdown_cell(
        "### Biểu đồ So sánh Ensemble và Mức độ Bất đồng Dự đoán (Disagreement Rate)"
    ))

    # Cell 9: Code Section 2 Chart
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n\n"
        "# Biểu đồ 1: Độ chính xác của các Ensemble theo Dataset\n"
        "sns.barplot(\n"
        "    data=df_r2,\n"
        "    x='dataset_id',\n"
        "    y='mean_accuracy_pct',\n"
        "    hue='ensemble_type',\n"
        "    palette='Set2',\n"
        "    ax=axes[0],\n"
        ")\n"
        "axes[0].set_title('So sánh Độ chính xác giữa các Cấu hình Ensemble', fontsize=12, fontweight='bold')\n"
        "axes[0].set_ylabel('Accuracy (%)')\n"
        "axes[0].set_xlabel('Tập Dữ Liệu')\n"
        "axes[0].legend(title='Ensemble Type', loc='lower right')\n\n"
        "# Biểu đồ 2: Mức độ Bất đồng Dự đoán (Pairwise Disagreement)\n"
        "sns.barplot(\n"
        "    data=df_r2,\n"
        "    x='dataset_id',\n"
        "    y='avg_pairwise_disagreement',\n"
        "    hue='ensemble_type',\n"
        "    palette='Set2',\n"
        "    ax=axes[1],\n"
        ")\n"
        "axes[1].set_title('Mức độ Bất đồng Dự đoán (Diversity / Disagreement Rate)', fontsize=12, fontweight='bold')\n"
        "axes[1].set_ylabel('Pairwise Disagreement Rate')\n"
        "axes[1].set_xlabel('Tập Dữ Liệu')\n"
        "axes[1].legend(title='Ensemble Type', loc='upper right')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 10: Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Thí nghiệm R3: Phân tích Độ Nhạy Cảm (Sensitivity Analysis)\n\n"
        "Khảo sát tác động của giá trị ngưỡng ban đầu $\\lambda_{initial} \\in \\{0.2, 0.4, 0.6\\}$ "
        "đối với chiến lược `sp_ddcl_point` trên 2 tập dữ liệu `haberman` và `new_thyroid`."
    ))

    # Cell 11: Code Section 3
    cells.append(nbf.v4.new_code_cell(
        "sweep_file = Path('../outputs/sensitivity/lambda_sweep.json') if Path('../outputs/sensitivity/lambda_sweep.json').exists() else Path('outputs/sensitivity/lambda_sweep.json')\n"
        "with open(sweep_file, 'r', encoding='utf-8') as f:\n"
        "    sweep_data = json.load(f)\n\n"
        "df_sweep = pd.DataFrame(sweep_data)\n"
        "cols_sweep = ['dataset_id', 'lambda_initial', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'avg_optimizer_steps', 'avg_sample_exposures']\n"
        "display(df_sweep[cols_sweep])"
    ))

    # Cell 12: Code Section 3 Chart
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n\n"
        "# Haberman Sensitivity\n"
        "hab_data = df_sweep[df_sweep['dataset_id'] == 'haberman']\n"
        "axes[0].plot(hab_data['lambda_initial'], hab_data['accuracy_mean'] * 100, marker='o', linewidth=2.5, color='#2b5c8f', label='Accuracy (%)')\n"
        "axes[0].set_title('Haberman: Tác động của Lambda Khởi đầu', fontsize=12, fontweight='bold')\n"
        "axes[0].set_xlabel('Initial Lambda Threshold')\n"
        "axes[0].set_ylabel('Accuracy (%)')\n"
        "axes[0].set_ylim(65, 80)\n\n"
        "# Secondary axis for optimizer steps\n"
        "ax0_steps = axes[0].twinx()\n"
        "ax0_steps.bar(hab_data['lambda_initial'], hab_data['avg_optimizer_steps'], alpha=0.25, width=0.08, color='orange', label='Optimizer Steps')\n"
        "ax0_steps.set_ylabel('Số bước cập nhật Optimizer')\n"
        "ax0_steps.grid(False)\n\n"
        "# New Thyroid Sensitivity\n"
        "thy_data = df_sweep[df_sweep['dataset_id'] == 'new_thyroid']\n"
        "axes[1].plot(thy_data['lambda_initial'], thy_data['accuracy_mean'] * 100, marker='s', linewidth=2.5, color='#c44e52', label='Accuracy (%)')\n"
        "axes[1].set_title('New Thyroid: Tác động của Lambda Khởi đầu', fontsize=12, fontweight='bold')\n"
        "axes[1].set_xlabel('Initial Lambda Threshold')\n"
        "axes[1].set_ylabel('Accuracy (%)')\n"
        "axes[1].set_ylim(30, 50)\n\n"
        "ax1_steps = axes[1].twinx()\n"
        "ax1_steps.bar(thy_data['lambda_initial'], thy_data['avg_optimizer_steps'], alpha=0.25, width=0.08, color='purple', label='Optimizer Steps')\n"
        "ax1_steps.set_ylabel('Số bước cập nhật Optimizer')\n"
        "ax1_steps.grid(False)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Cell 13: Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Tổng kết & Nhận xét Khoa học\n\n"
        "1. **Hành vi của Ngưỡng Batch trong SPL**: Thí nghiệm độ nhạy cảm R3 cho thấy khi $\\lambda$ ban đầu thấp hơn loss của các batch, thuật toán rơi vào fallback chọn batch min loss duy nhất, dẫn đến số bước cập nhật cực kỳ hạn chế (5 bước trong 5 epochs ở `new_thyroid`). Khi $\\lambda$ cao hơn (như trên `haberman`), số bước cập nhật tăng lên đáng kể và độ chính xác cải thiện rõ rệt.\n"
        "2. **Tính Ổn định của Controlled Ensemble**: Mô hình `Controlled 6-NN` đạt độ chính xác tương đương hoặc nhỉnh hơn `DDCL-6` trên cả 3 tập pilot, chứng minh rằng phần lớn lợi ích của ensemble đến từ việc kết hợp nhiều mạng ngẫu nhiên độc lập (variance reduction) chứ không chỉ riêng việc đa dạng hóa curriculum.\n"
        "3. **Tính Minh Bạch về Sai lệch**: Toàn bộ kết quả đối chiếu đều ghi rõ số liệu nào có trong paper và số liệu nào là mở rộng độc lập của nhóm sinh viên."
    ))

    nb.cells = cells
    return nb


def main():
    proj_root = Path(__file__).resolve().parent.parent
    nb_path = proj_root / "notebooks" / "04_reproduction.ipynb"

    print(f"[*] Building notebook: {nb_path}")
    nb = create_reproduction_notebook()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("[*] Executing notebook using kernel 'ddcl-local'...")
    client = NotebookClient(nb, timeout=600, kernel_name="ddcl-local")
    client.execute(cwd=str(proj_root / "notebooks"))

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[+] Notebook successfully executed and saved to {nb_path}!")


if __name__ == "__main__":
    main()

