# Báo cáo Tái lập Thí nghiệm — Chính thức (Official Final Reproduction Report: 30 seeds, 200/600 epochs)

**Cập nhật**: Phase E (Thực nghiệm Final chính thức: 3 datasets × 6 strategies × 30 seeds).

> [!IMPORTANT]
> **Cơ sở Đối chiếu Bài báo Gốc (Ground Truth References)**:
> - Bài báo: Shonal Chaudhry & Anuraganand Sharma, *Dynamic Data Distribution-based Curriculum Learning*, **Information Sciences** 702 (2025) 121924, DOI: `10.1016/j.ins.2025.121924` (tập tin PDF: `references/papers/dynamic-ddcl.pdf`).
> - **Table 1 (Trang 8)**: Single models (200 epochs, 30 runs, split 70/20/10) trên Haberman và New-Thyroid.
> - **Table 2 (Trang 9)**: Single models (200 epochs, 30 runs) trên Banknote.
> - **Table 3 (Trang 9)**: Bài báo gốc gồm 6 tập dữ liệu: Breast Cancer Wisconsin (Diagnostic), Cancer, Haberman's Survival, Liver Disorder, Pima Indians Diabetes và New-Thyroid với Ensemble models (600 epochs/member, 30 runs). Trong phần việc do cá nhân phụ trách (User), thành viên đã hoàn thành tái lập trên 2 tập được phân công là Haberman và New-Thyroid. Tập Banknote (từ Table 2) không xuất hiện trong Table 3 bài báo gốc và được đánh giá độc lập ở mốc 200 epochs. Các datasets còn lại của bài báo thuộc phân công của các thành viên khác trong nhóm và hiện ở trạng thái **Chưa xác nhận** (không tự ý thu hẹp phạm vi đồ án).
> - **Quy ước đơn vị**: Cột chênh lệch ghi nhận bằng **điểm phần trăm (pp - percentage points)** để phân biệt rạch ròi với tỷ lệ phần trăm tương đối (%).

---

## 1. Thí nghiệm R1: Đối chiếu 6 Chiến lược (Headline Single-Model Comparison)

Bảng thống kê kết quả kiểm thử trên 3 tập dữ liệu (30 seeds × 200 epochs):

| dataset_id | strategy | seeds_completed | mean_accuracy_pct | std_accuracy_ddof1 | std_accuracy_ddof0 | mean_macro_f1_pct | std_macro_f1 | avg_optimizer_steps | avg_sample_exposures | avg_wall_time_sec | paper_reported_pct | delta_vs_paper_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | no_curriculum | 30 | 71.29 | 6.47 | 6.36 | 56.67 | 9.96 | 1400.0 | 42600.0 | 9.951 | 65.833 | 5.46 |
| haberman | ddcl_density | 30 | 72.47 | 7.47 | 7.34 | 59.27 | 10.04 | 1400.0 | 42600.0 | 9.996 | 69.167 | 3.31 |
| haberman | ddcl_point | 30 | 72.15 | 7.21 | 7.09 | 59.18 | 10.32 | 1400.0 | 42600.0 | 10.198 | 67.222 | 4.93 |
| haberman | self_paced | 30 | 69.57 | 6.87 | 6.75 | 56.45 | 9.29 | 2626.5 | 79866.9 | 18.057 | 64.861 | 4.71 |
| haberman | sp_ddcl_density | 30 | 71.51 | 5.75 | 5.65 | 55.56 | 9.16 | 2423.6 | 73846.4 | 14.467 | 69.028 | 2.48 |
| haberman | sp_ddcl_point | 30 | 71.18 | 5.68 | 5.58 | 57.22 | 9.31 | 2440.8 | 75250.4 | 11.627 | 68.056 | 3.13 |
| new_thyroid | no_curriculum | 30 | 95.61 | 4.38 | 4.31 | 93.39 | 8.38 | 1000.0 | 30000.0 | 5.621 | 93.137 | 2.47 |
| new_thyroid | ddcl_density | 30 | 95.3 | 4.38 | 4.31 | 92.85 | 8.52 | 1000.0 | 30000.0 | 5.635 | 95.49 | -0.19 |
| new_thyroid | ddcl_point | 30 | 95.15 | 4.76 | 4.68 | 92.66 | 8.85 | 1000.0 | 30000.0 | 4.446 | 94.118 | 1.03 |
| new_thyroid | self_paced | 30 | 95.76 | 4.12 | 4.05 | 93.69 | 8.06 | 1855.3 | 55632.9 | 6.224 | 94.902 | 0.86 |
| new_thyroid | sp_ddcl_density | 30 | 95.61 | 4.7 | 4.62 | 93.48 | 8.61 | 1887.4 | 56638.9 | 6.286 | 96.078 | -0.47 |
| new_thyroid | sp_ddcl_point | 30 | 95.76 | 3.95 | 3.88 | 93.69 | 7.92 | 1875.3 | 56302.3 | 6.202 | 93.725 | 2.03 |
| banknote | no_curriculum | 30 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 6000.0 | 191800.0 | 22.187 | 99.879 | 0.12 |
| banknote | ddcl_density | 30 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 6000.0 | 191800.0 | 25.426 | 99.913 | 0.09 |
| banknote | ddcl_point | 30 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 6000.0 | 191800.0 | 26.062 | 99.942 | 0.06 |
| banknote | self_paced | 30 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 11784.6 | 376713.1 | 35.998 | 99.939 | 0.06 |
| banknote | sp_ddcl_density | 30 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 11951.9 | 382062.6 | 37.963 | 99.884 | 0.12 |
| banknote | sp_ddcl_point | 30 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 11875.9 | 379633.5 | 37.742 | 99.884 | 0.12 |

### Nhận xét R1:
- **Banknote (Table 2, p. 9)**: Bài báo gốc ghi nhận `No Curriculum`: 99.879% ± 0.309% đến `DDCL-Point`: 99.942% ± 0.217%. Thực nghiệm độc lập của nhóm đạt **100.00% ± 0.00%** trên toàn bộ 30 seeds ở cả 6 chiến lược (chênh lệch +0.06 đến +0.12 điểm phần trăm so với tác giả), xác nhận bài toán phân tách tốt hội tụ trọn vẹn.
- **New Thyroid (Table 1, p. 8)**: Bài báo gốc ghi nhận `No Curriculum`: 93.137% ± 5.280% và `SP DDCL-Density`: 96.078% ± 4.882%. Thực nghiệm 200 epochs của nhóm đạt **95.15% - 95.76%** (khớp chặt chẽ trong khoảng kỳ vọng của bài báo, chênh lệch từ -0.32 pp đến +2.01 pp so với No Curriculum tác giả).
- **Haberman (Table 1, p. 8)**: Bài báo gốc ghi nhận `No Curriculum`: 65.833% ± 8.898% và `DDCL-Density`: 69.167% ± 10.353%. Thực nghiệm của nhóm đạt **69.57% - 72.47%** (vượt +3.74 pp đến +6.64 pp so với No Curriculum bài báo). Độ lệch chuẩn mẫu thực nghiệm (~6.5 pp) nhỏ hơn bài báo (~8.9 - 10.7 pp).
- **Số bước cập nhật (Optimizer Steps)**: Các chiến lược SPL chọn lọc batch theo độ khó làm tăng số lượt cập nhật mẫu có kiểm soát so với huấn luyện thông thường.

---

## 2. Thí nghiệm R2: Đối chiếu Các Cấu hình Ensemble

### 2a. Giao thức Trung gian R2 (200 Epochs/Member — 30 Seeds)

Đánh giá cấu hình Ensemble từ các thành viên huấn luyện 200 epochs (kết quả trung gian của nhóm):

| dataset_id | ensemble_type | seeds_completed | mean_accuracy_pct | std_accuracy_ddof1 | mean_macro_f1_pct | std_macro_f1 | avg_tie_count | avg_pairwise_disagreement | paper_reported_pct | delta_vs_paper_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | DDCL-6 Strategy Ensemble | 30 | 71.08 | 6.25 | 56.0 | 9.61 | 0.4 | 0.0506 | N/A (Paper Table 3 uses 600ep) | N/A |
| haberman | Standard Paper Baseline (NN+SVM) | 30 | 74.84 | 5.18 | 53.42 | 11.3 | 3.4 | 0.1097 | N/A (Paper Table 3 uses 600ep) | N/A |
| haberman | Controlled 6-NN Baseline | 30 | 71.08 | 6.37 | 56.4 | 9.72 | 0.5 | 0.048 | N/A (Paper Table 3 uses 600ep) | N/A |
| new_thyroid | DDCL-6 Strategy Ensemble | 30 | 95.61 | 4.38 | 93.39 | 8.38 | 0.1 | 0.0098 | N/A (Paper Table 3 uses 600ep) | N/A |
| new_thyroid | Standard Paper Baseline (NN+SVM) | 30 | 94.85 | 4.73 | 91.44 | 9.17 | 0.8 | 0.0348 | N/A (Paper Table 3 uses 600ep) | N/A |
| new_thyroid | Controlled 6-NN Baseline | 30 | 95.61 | 4.85 | 93.15 | 9.01 | 0.1 | 0.0122 | N/A (Paper Table 3 uses 600ep) | N/A |
| banknote | DDCL-6 Strategy Ensemble | 30 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | N/A (Paper Table 3 uses 600ep) | N/A |
| banknote | Standard Paper Baseline (NN+SVM) | 30 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | N/A (Paper Table 3 uses 600ep) | N/A |
| banknote | Controlled 6-NN Baseline | 30 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | N/A (Paper Table 3 uses 600ep) | N/A |

### 2b. Giao thức Chuẩn Paper Table 3 (600 Epochs/Member — Đối chiếu trực tiếp Bài báo gốc)

Đánh giá cấu hình Ensemble với đúng **600 epochs/thành viên** theo giao thức Table 3 của bài báo gốc:
- **DDCL-6 Ensemble**: 6 mạng nơ-ron huấn luyện theo 6 chiến lược curriculum, 600 epochs/mạng.
- **Standard Paper Baseline**: 1 mạng NN (no_curriculum, 600 epochs) + 1 SVM RBF ($C=1.0$).
- **Controlled 6-NN Baseline**: 6 mạng NN độc lập (no_curriculum, 600 epochs), kiểm soát đa dạng hóa ngẫu nhiên.

| dataset_id | ensemble_type | seeds_completed | mean_accuracy_pct | std_accuracy_ddof1 | mean_macro_f1_pct | std_macro_f1 | avg_tie_count | avg_pairwise_disagreement | paper_reported_pct | delta_vs_paper_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | DDCL-6 Strategy Ensemble | 30 | 68.39 | 6.58 | 54.95 | 8.7 | 0.8 | 0.0673 | 96.774 | -28.39 |
| haberman | Standard Paper Baseline (NN+SVM) | 30 | 74.52 | 5.31 | 52.39 | 11.09 | 5.4 | 0.1742 | 90.323 | -15.81 |
| haberman | Controlled 6-NN Baseline | 30 | 68.39 | 6.13 | 55.18 | 7.72 | 0.8 | 0.0735 | N/A (Team Extension) | N/A |
| new_thyroid | DDCL-6 Strategy Ensemble | 30 | 95.61 | 4.38 | 93.53 | 8.25 | 0.0 | 0.0042 | 100.0 | -4.39 |
| new_thyroid | Standard Paper Baseline (NN+SVM) | 30 | 94.85 | 4.73 | 91.44 | 9.17 | 0.9 | 0.0424 | 95.455 | -0.61 |
| new_thyroid | Controlled 6-NN Baseline | 30 | 95.61 | 4.54 | 93.3 | 8.71 | 0.0 | 0.0113 | N/A (Team Extension) | N/A |

### Nhận xét R2:
- **Đối chiếu Table 3 Bài báo Gốc (600 epochs)**:
  * Trên `new_thyroid`: Bài báo Table 3 báo cáo Standard Ensemble: 95.455% ± 0.000% và DDCL Ensemble: 100.000% ± 0.000%. Thực nghiệm 30 seeds của nhóm đạt 94.85% ± 4.73% cho Standard Ensemble (lệch -0.61 pp) và 95.61% ± 4.38% cho DDCL-6 (lệch -4.39 pp).
  * Trên `haberman`: Bài báo Table 3 báo cáo Standard Ensemble: 90.323% ± 0.000% và DDCL Ensemble: 96.774% ± 0.000%. Thực nghiệm 30 seeds của nhóm đạt 74.52% ± 5.31% cho Standard Ensemble và 68.39% ± 6.58% cho DDCL-6.
  * Bài báo gốc ghi nhận độ lệch chuẩn ± 0.000% cho cả hai ensemble. Toàn bộ các giả thuyết về quy trình phân chia split, số lần chia split thực tế hoặc cơ chế đánh giá của tác giả chưa được kiểm chứng từ mã nguồn gốc và sẽ được xác minh chính thức trong Phase F.
  * Phạm vi Table 3: Table 3 bài báo gốc gồm 6 tập dữ liệu (Breast Cancer Wisconsin (Diagnostic), Cancer, Haberman's Survival, Liver Disorder, Pima Indians Diabetes, New-Thyroid). Trong phân công thực nghiệm của cá nhân phụ trách (User), thành viên đã hoàn tất tái lập trên 2 tập được giao là `haberman` và `new_thyroid` (180 evaluations). Tập `banknote` (từ Table 2) không xuất hiện trong Table 3 bài báo gốc nên có đúng **0 evaluations** ở mốc 600 epochs (Banknote được khảo sát mở rộng độc lập ở mốc 200 epochs). Các datasets còn lại của bài báo thuộc phân công của các thành viên khác trong nhóm và hiện ở trạng thái **Chưa xác nhận** (không tự ý thu hẹp phạm vi đồ án).
- **Tỷ lệ Bất đồng Dự đoán (Pairwise Disagreement)**: Tỷ lệ bất đồng dự đoán giữa 6 thành viên đạt từ 0.4% đến 6.7% trên tập kiểm thử. Đây là chỉ dấu ghi nhận sự khác biệt dự đoán ở mức độ mẫu đơn lẻ giữa các cơ chế sắp xếp, chưa đủ để khái quát thành quy luật giảm phương sai (variance reduction) toàn diện.
- **Cơ chế Tie-break Quyết định D09**: Trên 30 seeds, các tình huống hòa phiếu được phân định dứt điểm và khách quan theo train class prior, bảo đảm không rò rỉ bất kỳ thông tin nào từ nhãn kiểm thử.

---

## 3. Thí nghiệm R3: Phân tích Độ Nhạy Cảm (Sensitivity Analysis)

Khảo sát tác động của giá trị ngưỡng ban đầu $\lambda_{initial} \in \{0.2, 0.4, 0.6\}$ đối với chiến lược `sp_ddcl_point` (30 seeds × 200 epochs):

| dataset_id | strategy | lambda_initial | seeds_completed | accuracy_mean_pct | accuracy_std_ddof1 | macro_f1_mean_pct | avg_optimizer_steps | avg_sample_exposures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | sp_ddcl_point | 0.2 | 30 | 70.75 | 6.5 | 56.61 | 2381.8 | 73426.2 |
| haberman | sp_ddcl_point | 0.4 | 30 | 71.18 | 5.68 | 57.22 | 2440.8 | 75250.4 |
| haberman | sp_ddcl_point | 0.6 | 30 | 70.65 | 5.25 | 56.38 | 2458.2 | 75768.3 |
| new_thyroid | sp_ddcl_point | 0.2 | 30 | 95.76 | 3.95 | 93.69 | 1862.5 | 55916.7 |
| new_thyroid | sp_ddcl_point | 0.4 | 30 | 95.76 | 3.95 | 93.69 | 1875.3 | 56302.3 |
| new_thyroid | sp_ddcl_point | 0.6 | 30 | 95.76 | 3.95 | 93.69 | 1877.0 | 56354.7 |

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
