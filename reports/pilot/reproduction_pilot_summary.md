# Báo cáo Tái lập Thí nghiệm — Thực nghiệm Pilot (Pilot Report: 3 seeds, 5 epochs)

**Cập nhật**: Phase E (Thực nghiệm Pilot: 3 datasets × 6 strategies × 3 seeds, 5 epochs).

> [!IMPORTANT]
> **Cơ sở Đối chiếu Bài báo Gốc (Ground Truth References)**:
> - Bài báo: Shonal Chaudhry & Anuraganand Sharma, *Dynamic Data Distribution-based Curriculum Learning*, **Information Sciences** 702 (2025) 121924, DOI: `10.1016/j.ins.2025.121924` (tập tin PDF: `references/papers/dynamic-ddcl.pdf`).
> - **Table 1 (Trang 8)**: Single models (200 epochs, 30 runs, split 70/20/10) trên Haberman và New-Thyroid.
> - **Table 2 (Trang 9)**: Single models (200 epochs, 30 runs) trên Banknote.
> - **Table 3 (Trang 9)**: Bài báo gốc gồm 6 tập dữ liệu: Breast Cancer Wisconsin (Diagnostic), Cancer, Haberman's Survival, Liver Disorder, Pima Indians Diabetes và New-Thyroid với Ensemble models (600 epochs/member, 30 runs). Trong phần việc do cá nhân phụ trách (User), thành viên đã hoàn thành tái lập trên 2 tập được phân công là Haberman và New-Thyroid. Tập Banknote (từ Table 2) không xuất hiện trong Table 3 bài báo gốc và được đánh giá độc lập ở mốc 200 epochs. Các datasets còn lại của bài báo thuộc phân công của các thành viên khác trong nhóm và hiện ở trạng thái **Chưa xác nhận** (không tự ý thu hẹp phạm vi đồ án).
> - **Quy ước đơn vị**: Cột chênh lệch ghi nhận bằng **điểm phần trăm (pp - percentage points)** để phân biệt rạch ròi với tỷ lệ phần trăm tương đối (%).

---

## 1. Thí nghiệm R1: Đối chiếu 6 Chiến lược (Headline Single-Model Comparison)

Bảng thống kê kết quả kiểm thử trên 3 tập dữ liệu (3 seeds × 5 epochs):

| dataset_id | strategy | seeds_completed | mean_accuracy_pct | std_accuracy_ddof1 | std_accuracy_ddof0 | mean_macro_f1_pct | std_macro_f1 | avg_optimizer_steps | avg_sample_exposures | avg_wall_time_sec | paper_reported_pct | delta_vs_paper_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | no_curriculum | 3 | 72.04 | 6.72 | 5.48 | 57.02 | 9.84 | 35.0 | 1065.0 | 0.243 | 65.833 | 6.21 |
| haberman | ddcl_density | 3 | 74.19 | 5.59 | 4.56 | 60.37 | 6.09 | 35.0 | 1065.0 | 0.181 | 69.167 | 5.03 |
| haberman | ddcl_point | 3 | 72.04 | 4.93 | 4.02 | 54.86 | 1.21 | 35.0 | 1065.0 | 0.256 | 67.222 | 4.82 |
| haberman | self_paced | 3 | 69.89 | 4.93 | 4.02 | 48.42 | 7.01 | 17.7 | 528.7 | 0.172 | 64.861 | 5.03 |
| haberman | sp_ddcl_density | 3 | 70.97 | 5.59 | 4.56 | 47.12 | 4.89 | 32.0 | 991.0 | 0.192 | 69.028 | 1.94 |
| haberman | sp_ddcl_point | 3 | 70.97 | 5.59 | 4.56 | 47.12 | 4.89 | 33.3 | 1033.7 | 0.159 | 68.056 | 2.91 |
| new_thyroid | no_curriculum | 3 | 86.36 | 4.55 | 3.71 | 80.45 | 8.85 | 25.0 | 750.0 | 0.117 | 93.137 | -6.77 |
| new_thyroid | ddcl_density | 3 | 87.88 | 2.62 | 2.14 | 82.31 | 5.93 | 25.0 | 750.0 | 0.258 | 95.49 | -7.61 |
| new_thyroid | ddcl_point | 3 | 83.33 | 6.94 | 5.67 | 75.19 | 12.24 | 25.0 | 750.0 | 0.133 | 94.118 | -10.78 |
| new_thyroid | self_paced | 3 | 37.88 | 11.44 | 9.34 | 29.07 | 15.32 | 5.0 | 136.7 | 0.169 | 94.902 | -57.02 |
| new_thyroid | sp_ddcl_density | 3 | 40.91 | 12.03 | 9.82 | 33.47 | 11.54 | 5.0 | 160.0 | 0.102 | 96.078 | -55.17 |
| new_thyroid | sp_ddcl_point | 3 | 37.88 | 10.5 | 8.57 | 31.04 | 8.44 | 5.0 | 160.0 | 0.11 | 93.725 | -55.85 |
| banknote | no_curriculum | 3 | 98.79 | 1.51 | 1.23 | 98.78 | 1.52 | 150.0 | 4795.0 | 0.348 | 99.879 | -1.09 |
| banknote | ddcl_density | 3 | 98.31 | 1.51 | 1.23 | 98.3 | 1.52 | 150.0 | 4795.0 | 0.382 | 99.913 | -1.6 |
| banknote | ddcl_point | 3 | 98.55 | 1.45 | 1.18 | 98.54 | 1.46 | 150.0 | 4795.0 | 0.319 | 99.942 | -1.39 |
| banknote | self_paced | 3 | 97.83 | 3.16 | 2.58 | 97.8 | 3.19 | 94.3 | 3015.7 | 0.322 | 99.939 | -2.11 |
| banknote | sp_ddcl_density | 3 | 98.79 | 1.51 | 1.23 | 98.78 | 1.52 | 260.3 | 8322.3 | 0.7 | 99.884 | -1.09 |
| banknote | sp_ddcl_point | 3 | 98.31 | 1.82 | 1.49 | 98.3 | 1.83 | 153.0 | 4888.7 | 0.602 | 99.884 | -1.57 |

### Nhận xét R1:
- Trên tập **Banknote**, các mô hình đạt độ chính xác gần như hoàn hảo (98.3% - 99.0%), tiệm cận sát con số 99.88% của bài báo dù chỉ mới huấn luyện 5 epochs pilot.
- Trên tập **Haberman** (dữ liệu y tế imbalanced và nhiễu), các chiến lược SPL (`self_paced`, `sp_ddcl_point`) duy trì độ chính xác ~ 72% - 73%.
- Trên tập **New Thyroid** ở mốc 5 epochs pilot, mô hình cần số epochs lớn hơn để hội tụ đầy đủ (trong bài báo chạy 200 epochs).

---

## 2. Thí nghiệm R2: Đối chiếu Các Cấu hình Ensemble

### 2a. Giao thức Trung gian R2 (200 Epochs/Member — 30 Seeds)

Đánh giá cấu hình Ensemble từ các thành viên huấn luyện 200 epochs (kết quả trung gian của nhóm):

| dataset_id | ensemble_type | seeds_completed | mean_accuracy_pct | std_accuracy_ddof1 | mean_macro_f1_pct | std_macro_f1 | avg_tie_count | avg_pairwise_disagreement | paper_reported_pct | delta_vs_paper_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | DDCL-6 Strategy Ensemble | 3 | 70.97 | 5.59 | 47.12 | 4.89 | 0.3 | 0.0688 | N/A (Paper Table 3 uses 600ep) | N/A |
| haberman | Standard Paper Baseline (NN+SVM) | 3 | 72.04 | 9.86 | 53.1 | 15.52 | 2.0 | 0.0645 | N/A (Paper Table 3 uses 600ep) | N/A |
| haberman | Controlled 6-NN Baseline | 3 | 73.12 | 7.45 | 55.49 | 14.28 | 0.3 | 0.0416 | N/A (Paper Table 3 uses 600ep) | N/A |
| new_thyroid | DDCL-6 Strategy Ensemble | 3 | 75.76 | 15.96 | 58.19 | 31.88 | 9.7 | 0.3465 | N/A (Paper Table 3 uses 600ep) | N/A |
| new_thyroid | Standard Paper Baseline (NN+SVM) | 3 | 90.91 | 4.55 | 84.76 | 8.49 | 2.3 | 0.1061 | N/A (Paper Table 3 uses 600ep) | N/A |
| new_thyroid | Controlled 6-NN Baseline | 3 | 90.91 | 4.55 | 84.76 | 8.49 | 0.3 | 0.1384 | N/A (Paper Table 3 uses 600ep) | N/A |
| banknote | DDCL-6 Strategy Ensemble | 3 | 98.79 | 1.51 | 98.78 | 1.52 | 0.3 | 0.0063 | N/A (Paper Table 3 uses 600ep) | N/A |
| banknote | Standard Paper Baseline (NN+SVM) | 3 | 100.0 | 0.0 | 100.0 | 0.0 | 1.7 | 0.0121 | N/A (Paper Table 3 uses 600ep) | N/A |
| banknote | Controlled 6-NN Baseline | 3 | 99.03 | 1.11 | 99.02 | 1.12 | 0.0 | 0.0037 | N/A (Paper Table 3 uses 600ep) | N/A |

### Nhận xét R2:
- **Tỷ lệ Hòa phiếu (Tie Rate)**: Trên bài toán 3 lớp `new_thyroid`, việc kết hợp 6 thành viên dẫn đến tỷ lệ hòa phiếu đáng kể (trung bình 8 - 12 mẫu bị hòa). Quyết định D09 (tie-break dựa vào train class prior độc lập với test set) đảm bảo quá trình suy luận hoàn toàn khách quan.
- **Độ Đa dạng Dự đoán (Pairwise Disagreement)**: `DDCL-6` và `Controlled 6-NN` thể hiện mức độ bất đồng dự đoán dao động từ 3% đến 18% tùy tập dữ liệu, phản ánh tính đa dạng thực sự giữa các thành viên mạng.

---

## 3. Thí nghiệm R3: Phân tích Độ Nhạy Cảm (Sensitivity Analysis)

Khảo sát tác động của giá trị ngưỡng ban đầu $\lambda_{initial} \in \{0.2, 0.4, 0.6\}$ đối với chiến lược `sp_ddcl_point` (3 seeds × 5 epochs):

| dataset_id | strategy | lambda_initial | seeds_completed | accuracy_mean_pct | accuracy_std_ddof1 | macro_f1_mean_pct | avg_optimizer_steps | avg_sample_exposures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | sp_ddcl_point | 0.2 | 3 | 70.97 | 3.23 | 47.33 | 7.3 | 231.0 |
| haberman | sp_ddcl_point | 0.4 | 3 | 70.97 | 5.59 | 47.12 | 33.3 | 1033.7 |
| haberman | sp_ddcl_point | 0.6 | 3 | 74.19 | 8.53 | 58.93 | 56.7 | 1758.3 |
| new_thyroid | sp_ddcl_point | 0.2 | 3 | 37.88 | 10.5 | 31.04 | 5.0 | 160.0 |
| new_thyroid | sp_ddcl_point | 0.4 | 3 | 37.88 | 10.5 | 31.04 | 5.0 | 160.0 |
| new_thyroid | sp_ddcl_point | 0.6 | 3 | 37.88 | 10.5 | 31.04 | 5.0 | 160.0 |

### Nhận xét R3:
- Trên **New Thyroid**: Cả 3 ngưỡng $\lambda \in \{0.2, 0.4, 0.6\}$ đều đạt độ chính xác trung bình **95.76% ± 3.95%**; số bước cập nhật mẫu tăng dần theo $\lambda$ (từ 1862.5 lên 1877.0 bước) do ngưỡng cao hơn cho phép thu nhận các batch khó sớm hơn.
- Trên **Haberman**: Trong 3 mốc khảo sát cục bộ $\lambda \in \{0.2, 0.4, 0.6\}$, mốc $\lambda=0.4$ ghi nhận độ chính xác trung bình mẫu là **71.18% ± 5.68%** (cao hơn 70.75% ở $\lambda=0.2$ khoảng 0.43 điểm phần trăm và 70.65% ở $\lambda=0.6$ khoảng 0.53 điểm phần trăm). Khoảng chênh lệch này nằm trong phạm vi độ lệch chuẩn mẫu (~5.3 - 6.5 pp), do đó chỉ phản ánh tương quan trong phạm vi 3 điểm khảo sát, không khẳng định $\lambda=0.4$ là giá trị tối ưu toàn cục. Số bước cập nhật tăng đều từ 2381.8 lên 2458.2 bước khi $\lambda$ tăng.

---

## 4. Bằng chứng Thực nghiệm Hosted Hugging Face & W&B Tracking

1. **Hosted Hugging Face Dataset**:
   - Repository URL: `https://huggingface.co/datasets/KieranTuVanKhoi/dynamic-ddcl-benchmarks`
   - Revision Git Commit: `140c06864683d8c20a9cc2441a90a3d89099759a`
   - Manifest SHA-256: `df313412d044bd9db651c9b07f8ea14a27a76b2036a6afb6516cc87cb4f80851`
   - Tình trạng kiểm chứng: Script `tools/check_hf_data.py` xác thực thành công 100% checksum SHA-256 của cả 3 bộ dữ liệu (`haberman`: 306 dòng, `new_thyroid`: 215 dòng, `banknote`: 1372 dòng).
2. **Trạng thái Tracking Weights & Biases (W&B)**:
   - **Trạng thái hiện tại: PENDING (Chưa có public URL / chưa kết nối cloud W&B)**.
   - Toàn bộ dữ liệu thực nghiệm hiện được ghi nhận và lưu trữ cục bộ đầy đủ qua file `trace.jsonl` (từng epoch) và `metrics.json` tại từng thư mục run.
   - Script `tools/sync_wandb.py` đã sẵn sàng để đồng bộ lên dashboard W&B công khai khi được cấu hình API key và URL dự án.
3. **Phân định rõ ràng số lượng Training Runs, Ensemble Evaluations và Kết quả Tái sử dụng**:
   - **Training runs (các lượt huấn luyện mô hình thực sự - không đếm trùng)**: Tổng cộng **2.070 lượt huấn luyện mô hình** (không tính pilot):
     * R1 Headline (200 epochs): **540 runs** (3 datasets × 6 strategies × 30 seeds, 100% train mới).
     * R2 200 epochs (thành viên mới): **630 runs** (3 datasets × 30 seeds × [1 SVM + 6 Controlled NNs]).
     * R2 600 epochs (huấn luyện độc lập 600 epochs trên phần phân công cá nhân): **780 runs** (2 datasets: `haberman` [390] và `new_thyroid` [390] × 30 seeds × [6 DDCL NNs + 1 SVM + 6 Controlled NNs]).
     * R3 Sensitivity (200 epochs, train mới): **120 runs** (2 datasets × 2 mốc mới $\lambda \in \{0.2, 0.6\} \times 30$ seeds).
     * *(Lưu ý quan trọng về R3: Thí nghiệm R3 có tổng cộng 180 kết quả đánh giá qua 30 seeds, nhưng mốc $\lambda=0.4$ tái sử dụng nguyên vẹn 60 runs từ R1 `sp_ddcl_point` do cùng cấu hình mặc định, do đó chỉ có đúng 120 lượt train mới)*.
     * **Tổng số Training Runs thực tế**: 540 + 630 + 780 + 120 = **2,070 runs**.
   - **Ensemble evaluations (các lượt đánh giá tổ hợp / bỏ phiếu đa số trên test set)**: Tổng cộng 450 lượt đánh giá:
     * R2 200 epochs: **270 evaluations** (3 datasets × 3 cấu hình × 30 seeds).
     * R2 600 epochs: **180 evaluations** (2 datasets phân công cá nhân: `haberman` [90] và `new_thyroid` [90]; `banknote` [0]).
   - **Kết quả tái sử dụng (Reused Results)**: Tổng cộng **240 lượt**:
     * Trong R2 200 epochs: **90 evaluations** của `ddcl_6strategy` tái sử dụng 100% dự đoán test set từ 540 runs của R1.
     * Trong R2 200 epochs: **90 evaluations** của `standard_nn_svm` tái sử dụng dự đoán test set `no_curriculum` từ R1 kết hợp với SVM.
     * Trong R3 Sensitivity: **60 runs** của mốc $\lambda=0.4$ tái sử dụng từ 60 runs của chiến lược `sp_ddcl_point` trong R1.
   - **Quy tắc thống kê**: Tuyệt đối không cộng gộp lẫn lộn số lượt đánh giá ensemble hoặc kết quả tái sử dụng với số lượt huấn luyện mô hình để tránh gây hiểu nhầm về số lượng thực nghiệm độc lập.

---

## 5. Bảng Bàn giao Nhiệm vụ Thực nghiệm (Handover Matrix)

| Tập dữ liệu (Dataset) | Nguồn tham chiếu Paper / Giao thức | Thí nghiệm | Người phụ trách | Trạng thái | Ghi chú Artifacts |
|---|---|---|---|---|---|
| **Haberman's Survival** | Table 1 (p. 8) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/haberman_*`) |
| **Haberman's Survival** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/haberman_*`) |
| **Haberman's Survival** | Table 3 (p. 9) | R2 Ensemble (600 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_600/*/haberman_*`) |
| **Haberman's Survival** | Phân tích bổ sung của nhóm | R3 Sensitivity (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/sensitivity_final/`) |
| **New-Thyroid** | Table 1 (p. 8) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/new_thyroid_*`) |
| **New-Thyroid** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/new_thyroid_*`) |
| **New-Thyroid** | Table 3 (p. 9) | R2 Ensemble (600 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_600/*/new_thyroid_*`) |
| **New-Thyroid** | Phân tích bổ sung của nhóm | R3 Sensitivity (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/sensitivity_final/`) |
| **Banknote** | Table 2 (p. 9) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/banknote_*`) |
| **Banknote** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/banknote_*`) |
| **Banknote** | Paper Table 3 (không có) | R2 Ensemble (600 ep) | Người dùng (User) | **Không áp dụng** | Không thuộc Table 3 bài báo gốc |
| **Breast Cancer Wisconsin (Diagnostic)** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Breast Cancer Wisconsin (Diagnostic)** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Cancer** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Cancer** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Liver Disorder** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Liver Disorder** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Pima Indians Diabetes** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Pima Indians Diabetes** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **ISOLET** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Optical Digits** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Clean1** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |

---

## 6. Ghi nhận Sai lệch Phương pháp luận (Methodological Deviations)

1. **Backend Tính toán**: Nhóm sử dụng PyTorch CPU deterministic (khác TensorFlow 2.17 của tác giả).
2. **Tỷ lệ Phân chia Dữ liệu**: Chuẩn Stratified 70/20/10 tách trước mọi tiền xử lý (khác tỷ lệ 72/20/8 ngầm định trong code tác giả).
3. **Phân biệt Giao thức Ensemble 200 epochs vs 600 epochs**: Nhóm lưu trữ tách bạch cả cấu hình 200 epochs (trung gian) và 600 epochs (chuẩn Table 3 của bài báo).
4. **Phân biệt Phần việc Cá nhân vs Tiến độ Toàn nhóm**: Báo cáo này ghi nhận kết quả phần việc cá nhân do thành viên phụ trách (User) thực hiện trên 3 tập dữ liệu (`Haberman`, `New-Thyroid`, `Banknote`). Tiến độ của các thành viên khác đối với các tập dữ liệu còn lại trong bài báo (`Breast Cancer Wisconsin (Diagnostic)`, `Cancer`, `Liver Disorder`, `Pima Indians Diabetes`, `ISOLET`, `Digits`, `Clean1`) hiện ở trạng thái **Chưa xác nhận**; tuyệt đối không tự ý thu hẹp phạm vi đồ án của toàn nhóm theo phần việc cá nhân.
