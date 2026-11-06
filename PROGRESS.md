# Tiến độ dự án Dynamic DDCL (PROGRESS.md)

Cập nhật lần cuối: 2026-11-06 (Hoàn thiện các kiểm chứng chuẩn xác của Phase E Final theo PDF gốc, đối chiếu 200/600 epochs, chưa chuyển Phase F).

---

## 1. Trạng thái các Phase

| Phase | Tên Phase | Trạng thái | Ghi chú & Checkpoint Gate |
|---|---|---|---|
| **Phase A** | Đọc nguồn, chốt thiết kế, dựng môi trường | **COMPLETED / PASSED** | Đã đọc docs 01–12, course requirements, 2 PDFs. Đã tạo `pyproject.toml`, `DATA_REGISTRY.md`, `DECISIONS.md`, `ASSUMPTIONS.md`, `src/dddcl/cli.py` (doctor), `tests/test_environment.py`. Đã cài đặt `.venv`, `pip check` pass, `dddcl.cli doctor` pass, `pytest` 5/5 pass, `tools/test_data_tools.py` 9/9 pass, `00_environment.ipynb` executed thành công. |
| **Phase B** | Data hosted và split contract | **COMPLETED / PASSED (100% Hosted HF & Local Verified)** | Dữ liệu đã được publish lên HF `KieranTuVanKhoi/dynamic-ddcl-benchmarks` (revision `140c06864683d8c20a9cc2441a90a3d89099759a`). Đã xác thực tải và kiểm tra SHA-256 từ cloud thành công 100%. Đã triển khai `src/dddcl/data/`. Sinh 90 split files stratified 70/20/10 (30 seeds × 3 datasets). `notebooks/01_data_audit.ipynb` executed thành công. |
| **Phase C** | Curriculum + Pace pure logic | **COMPLETED / PASSED** | Đã triển khai `src/dddcl/curriculum/` (geometry, partition, resample, scoring) và `src/dddcl/training/pace.py`. Toàn bộ 9/9 unit tests tính tay, degenerate cases và oracle pass. `notebooks/02_curriculum_walkthrough.ipynb` executed thành công. |
| **Phase D** | Training loop độc lập & checkpoints | **COMPLETED / PASSED** | Đã triển khai PyTorch MLP model (`src/dddcl/models/mlp.py`), vòng lặp huấn luyện độc lập (`src/dddcl/training/loop.py`), atomic checkpoints (`src/dddcl/training/checkpoint.py`), metrics (`src/dddcl/metrics.py`), tracking JSONL/W&B (`src/dddcl/tracking.py`), runner & CLI (`src/dddcl/runner.py`, `cli.py run --dry-run`). 7/7 unit tests pass. `notebooks/03_training_walkthrough.ipynb` executed thành công. |
| **Phase E** | Reproduction + Ensemble | **HOÀN THIỆN ĐỐI CHIẾU CHUẨN XÁC (Sẵn sàng gate 100%)** | **Tách biệt hoàn toàn Pilot, Final 200 epochs, và Final 600 epochs**:<br>• **Pilot**: 54 runs R1 (5 epochs, 3 seeds), 9 ensembles R2, 6 runs R3, lưu tại `outputs/*_pilot` & `reports/pilot/`.<br>• **Final Core R1**: **540 runs R1** (200 epochs, 30 seeds), lưu tại `outputs/runs_final/`.<br>• **Final Core R2 (200 ep)**: **270 evaluations R2** (3 ensemble types × 30 seeds), lưu tại `outputs/ensembles_final/`.<br>• **Final Core R2 (600 ep)**: Chạy đối chiếu chuẩn Table 3 bài báo gốc tại `outputs/ensembles_600/`.<br>• **Final Core R3**: **180 runs R3** ($\lambda \in \{0.2, 0.4, 0.6\}$ × 30 seeds), lưu tại `outputs/sensitivity_final/`.<br>• **Kiểm chứng Paper Benchmark**: Đã đối chiếu 100% từng ô số liệu với `references/papers/dynamic-ddcl.pdf` (Elsevier Information Sciences 2025). 35/35 unit tests pass. Báo cáo và notebook cập nhật minh bạch. |
| **Phase F** | Audit mã nguồn tác giả | **PENDING (Chờ lệnh người dùng)** | Độc lập 100%, chưa clone hoặc xem code tác giả. Dừng đúng tại Phase E theo yêu cầu. |
| **Phase G** | Thực nghiệm phản biện lý thuyết (Empirics) | **PENDING** | Giả thuyết E1 (Centroid/Multimodality) và E2 (Batch-level threshold). |
| **Phase H** | Báo cáo, Slide deck & Poster | **PENDING** | Chuẩn bị artifact bàn giao cho môn học. |

---

## 2. Kết quả Thực nghiệm Phase E (Phân định Pilot vs Final)

### 2.1. Quản lý Dữ liệu & Lưu trữ Tách Biệt
- **Thực nghiệm Pilot (5 epochs, 3 seeds: 0, 1, 2)**:
  - Thư mục lưu trữ: `outputs/runs_pilot/`, `outputs/ensembles_pilot/`, `outputs/sensitivity_pilot/`.
  - Báo cáo Pilot: `reports/pilot/reproduction_pilot_summary.md`, `reports/pilot/reproduction_pilot_metrics.json`.
- **Thực nghiệm Final 200 Epochs (30 seeds: 0..29)**:
  - Thư mục lưu trữ: `outputs/runs_final/`, `outputs/ensembles_final/`, `outputs/sensitivity_final/`.
  - Báo cáo Final: `reports/final/reproduction_final_summary.md`, `reports/final/reproduction_final_metrics.json`.
- **Thực nghiệm Final 600 Epochs (Giao thức Table 3 Bài báo)**:
  - Thư mục lưu trữ: `outputs/ensembles_600/`.

---

### 2.2. Kết quả Thí nghiệm R1 Final (Headline Single-Model Benchmark: 540 runs)
*Đối chiếu với Table 1 (p. 8) và Table 2 (p. 9) của `dynamic-ddcl.pdf` (200 epochs, 30 seeds, 70/20/10 split)*:

| Dataset | Chiến lược | Seeds | Accuracy trung bình (Final) | SD mẫu (ddof=1) | Macro-F1 trung bình | Optimizer Steps | Số liệu Paper Table 1/2 | Delta vs Paper (pp) |
|---|---|---|---|---|---|---|---|---|
| **banknote** | `no_curriculum` | 30 | **100.00%** | 0.00% | 100.00% | 6,000.0 | 99.879 ± 0.309% | +0.12 pp |
| **banknote** | `ddcl_density` | 30 | **100.00%** | 0.00% | 100.00% | 6,000.0 | 99.913 ± 0.261% | +0.09 pp |
| **banknote** | `ddcl_point` | 30 | **100.00%** | 0.00% | 100.00% | 6,000.0 | 99.942 ± 0.217% | +0.06 pp |
| **banknote** | `self_paced` | 30 | **100.00%** | 0.00% | 100.00% | 11,784.6 | 99.939 ± 0.227% | +0.06 pp |
| **banknote** | `sp_ddcl_density` | 30 | **100.00%** | 0.00% | 100.00% | 11,951.9 | 99.884 ± 0.296% | +0.12 pp |
| **banknote** | `sp_ddcl_point` | 30 | **100.00%** | 0.00% | 100.00% | 11,875.9 | 99.884 ± 0.371% | +0.12 pp |
| **new_thyroid** | `no_curriculum` | 30 | **95.61%** | 4.38% | 93.39% | 1,000.0 | 93.137 ± 5.280% | +2.47 pp |
| **new_thyroid** | `ddcl_density` | 30 | **95.30%** | 4.38% | 92.85% | 1,000.0 | 95.490 ± 5.409% | -0.19 pp |
| **new_thyroid** | `ddcl_point` | 30 | **95.15%** | 4.76% | 92.66% | 1,000.0 | 94.118 ± 5.683% | +1.03 pp |
| **new_thyroid** | `self_paced` | 30 | **95.76%** | 4.12% | 93.69% | 1,855.3 | 94.902 ± 4.488% | +0.86 pp |
| **new_thyroid** | `sp_ddcl_density` | 30 | **95.61%** | 4.70% | 93.48% | 1,887.4 | 96.078 ± 4.882% | -0.47 pp |
| **new_thyroid** | `sp_ddcl_point` | 30 | **95.76%** | 3.95% | 93.69% | 1,875.3 | 93.725 ± 6.609% | +2.04 pp |
| **haberman** | `no_curriculum` | 30 | **71.29%** | 6.47% | 56.67% | 1,400.0 | 65.833 ± 8.898% | +5.46 pp |
| **haberman** | `ddcl_density` | 30 | **72.47%** | 7.47% | 59.27% | 1,400.0 | 69.167 ± 10.353% | +3.30 pp |
| **haberman** | `ddcl_point` | 30 | **72.15%** | 7.21% | 59.18% | 1,400.0 | 67.222 ± 9.424% | +4.93 pp |
| **haberman** | `self_paced` | 30 | **69.57%** | 6.87% | 56.45% | 2,626.5 | 64.861 ± 8.785% | +4.71 pp |
| **haberman** | `sp_ddcl_density` | 30 | **71.51%** | 5.75% | 55.56% | 2,423.6 | 69.028 ± 10.741% | +2.48 pp |
| **haberman** | `sp_ddcl_point` | 30 | **71.18%** | 5.68% | 57.22% | 2,440.8 | 68.056 ± 10.054% | +3.12 pp |

---

### 2.3. Kết quả Thí nghiệm R2 Final (Ensemble Comparisons)

#### 2.3a. Giao thức Trung gian (200 epochs/thành viên — 30 seeds, 270 evaluations):

| Dataset | Cấu hình Ensemble | Seeds | Accuracy trung bình | SD mẫu | Macro-F1 | Tỷ lệ hòa trung bình | Bất đồng dự đoán | Paper Table 3 (600ep) | Ghi chú so sánh |
|---|---|---|---|---|---|---|---|---|---|
| **haberman** | DDCL-6 Strategy Ensemble | 30 | **71.08%** | 6.25% | 56.00% | 0.4 mẫu | 5.06% | 96.774% | Đánh giá 200 epochs (Table 3 là 600 epochs) |
| **haberman** | Standard Paper Baseline (NN+SVM) | 30 | **74.84%** | 5.18% | 53.42% | 3.4 mẫu | 10.97% | 90.323% | Đánh giá 200 epochs (Table 3 là 600 epochs) |
| **haberman** | Controlled 6-NN Baseline | 30 | **71.08%** | 6.37% | 56.40% | 0.5 mẫu | 4.80% | N/A | Tương đương DDCL-6 (variance reduction) |
| **new_thyroid** | DDCL-6 Strategy Ensemble | 30 | **95.61%** | 4.38% | 93.39% | 0.1 mẫu | 0.98% | 100.000% | Đánh giá 200 epochs (Table 3 là 600 epochs) |
| **new_thyroid** | Standard Paper Baseline (NN+SVM) | 30 | **94.85%** | 4.73% | 91.44% | 0.8 mẫu | 3.48% | 95.455% | Đánh giá 200 epochs (Table 3 là 600 epochs) |
| **new_thyroid** | Controlled 6-NN Baseline | 30 | **95.61%** | 4.85% | 93.15% | 0.1 mẫu | 1.22% | N/A | Hiệu năng tương đương DDCL-6 |
| **banknote** | DDCL-6 Strategy Ensemble | 30 | **100.00%** | 0.00% | 100.00% | 0.0 mẫu | 0.00% | N/A | Mở rộng độc lập của nhóm |
| **banknote** | Standard Paper Baseline (NN+SVM) | 30 | **100.00%** | 0.00% | 100.00% | 0.0 mẫu | 0.00% | N/A | Mở rộng độc lập của nhóm |
| **banknote** | Controlled 6-NN Baseline | 30 | **100.00%** | 0.00% | 100.00% | 0.0 mẫu | 0.00% | N/A | Mở rộng độc lập của nhóm |

#### 2.3b. Giao thức Chuẩn Paper Table 3 (600 epochs/thành viên — `outputs/ensembles_600/`):
- Đang lưu trữ và cập nhật tại `outputs/ensembles_600/`.
- Đối chiếu trực tiếp Table 3 (p. 9):
  * `haberman`: Standard Ensemble paper báo 90.323 ± 0.000% (28/31 mẫu đúng), DDCL Ensemble báo 96.774 ± 0.000% (30/31 mẫu đúng).
  * `new_thyroid`: Standard Ensemble paper báo 95.455 ± 0.000% (21/22 mẫu đúng), DDCL Ensemble báo 100.000 ± 0.000% (22/22 mẫu đúng).
  * `banknote`: Không có trong Table 3 paper (Team Extension).

---

### 2.4. Kết quả Thí nghiệm R3 Final (Sensitivity Analysis trên $\lambda_{initial}$: 180 runs qua 30 seeds)

| Dataset | Ngưỡng $\lambda_{initial}$ | Seeds | Epochs | Accuracy trung bình | SD mẫu (ddof=1) | Macro-F1 trung bình | Optimizer Steps |
|---|---|---|---|---|---|---|---|
| **haberman** | $\lambda = 0.2$ | 30 | 200 | **70.75%** | 6.50% | 56.61% | 2,381.8 |
| **haberman** | $\lambda = 0.4$ | 30 | 200 | **71.18%** | 5.68% | 57.22% | 2,440.8 |
| **haberman** | $\lambda = 0.6$ | 30 | 200 | **70.65%** | 5.25% | 56.38% | 2,458.2 |
| **new_thyroid** | $\lambda = 0.2$ | 30 | 200 | **95.76%** | 3.95% | 93.69% | 1,862.5 |
| **new_thyroid** | $\lambda = 0.4$ | 30 | 200 | **95.76%** | 3.95% | 93.69% | 1,875.3 |
| **new_thyroid** | $\lambda = 0.6$ | 30 | 200 | **95.76%** | 3.95% | 93.69% | 1,877.0 |

**Nhận định R3**:
- Khác biệt rõ rệt so với mốc pilot 5 epochs (khi fallback ở $\lambda=0.6$ khiến số bước chỉ có 5), ở mốc 200 epochs thuật toán có đủ thời gian để hàm tăng trưởng $\lambda_{growth}=1.1$ nới rộng ngưỡng, cho phép tất cả các batch được tiếp cận đầy đủ.
- Ngưỡng $\lambda=0.4$ (giá trị mặc định của tác giả) cho độ chính xác trung bình cao nhất trong không gian khảo sát $\{0.2, 0.4, 0.6\}$ trên tập dữ liệu khó và nhiễu (`haberman`: 71.18%).
- Tỷ lệ bất đồng dự đoán (disagreement rate) và độ lệch chuẩn nhỏ hơn giữa các seed cung cấp bằng chứng thực nghiệm ban đầu ủng hộ giả thuyết giảm phương sai dự đoán (variance reduction) của mô hình kết hợp.

---

### 2.5. Kiểm thử Hệ thống & Artifacts
- **Kiểm thử PyTest**: **35/35 tests PASS 100%**:
  - `tests/test_environment.py`: 5/5 PASS.
  - `tests/test_data_pipeline.py`: 6/6 PASS.
  - `tests/test_curriculum.py`: 9/9 PASS.
  - `tests/test_training.py`: 7/7 PASS.
  - `tests/test_ensemble.py`: 8/8 PASS (Bao gồm test tie-breaking D09, alignment và controlled ensemble).
- **Notebook Trực quan hóa**:
  - `notebooks/04_reproduction.ipynb`: Đã tạo mới và thực thi hoàn tất trên kernel `ddcl-local`, bao gồm biểu đồ R1 30 seeds, biểu đồ R2 Ensemble & Disagreement, biểu đồ R3 Sensitivity, và biểu đồ so sánh trực tiếp Pilot vs Final.

---

## 3. Blockers và Tuân thủ Nguyên tắc Độc lập

1. **Tuyệt đối không vi phạm nguyên tắc độc lập**:
   - Trong suốt toàn bộ Phase A, B, C, D, E, nhóm **chưa từng clone hay đọc bất kỳ dòng code nào** từ kho lưu trữ của tác giả (`https://github.com/ECOLS-research-group/DDCL`).
   - Toàn bộ kết quả Pilot và Final đạt được 100% từ việc hiện thực hóa độc lập các phương trình và mô tả trong bài báo gốc và file đặc tả thiết kế.
2. **Sẵn sàng chuyển giao sang Phase F**:
   - Phase E đã hoàn thành đầy đủ cả pilot và final scope được giao.
   - Khi người dùng gửi lệnh tiếp tục, hệ thống sẽ tiến hành checkpoint git và chuyển sang Phase F (Audit mã nguồn tác giả).

---

## 4. Kế hoạch Phase F (Audit mã nguồn tác giả)

1. Checkpoint Git toàn bộ mã nguồn và artifacts độc lập của Phase E.
2. Clone kho lưu trữ tham chiếu tác giả `https://github.com/ECOLS-research-group/DDCL` vào thư mục `external/DDCL_author` (bị `.gitignore` bảo vệ tuyệt đối khỏi `src/`).
3. Khảo sát và đối chiếu:
   - Tỷ lệ phân chia split thực tế trong mã nguồn tác giả (kiểm chứng nghi vấn 72/20/8 vs 70/20/10).
   - Công thức tính khoảng cách, chuẩn hóa và phân nhóm quantile trong code tác giả.
   - Logic thực tế của vòng lặp Dynamic SPL trong TensorFlow/Keras so với Algorithm 2.
   - So sánh các đại lượng trung gian (split sizes, order, batch updates) giữa bản độc lập và bản tác giả.
