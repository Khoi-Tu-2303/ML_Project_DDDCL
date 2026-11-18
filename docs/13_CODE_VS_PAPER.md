# Báo cáo đối chiếu: Code hiện có so với bài báo gốc Dynamic DDCL

Bài báo: Chaudhry & Sharma, *Dynamic Data Distribution-based Curriculum Learning*, Information Sciences 702 (2025) 121924 (`dynamic-ddcl.pdf`).
Ngày lập: 2026-11-13. Căn cứ: đọc trực tiếp PDF (12 trang nội dung), mã nguồn `src/dddcl/`, `tools/`, `configs/`, `ASSUMPTIONS.md`, `DECISIONS.md` và `reports/final/reproduction_final_summary.md`.

Ký hiệu mức độ: **[KHỚP]** đúng với paper · **[DIỄN GIẢI]** paper không nêu rõ, code chọn một cách hiểu · **[LỆCH]** khác paper · **[CHƯA LÀM]** paper có, code không có.

---

## 1. Tóm tắt

1. **Khung thuật toán khớp paper**: 6 chiến lược (No Curriculum, DDCL-Density, DDCL-Point, Self-Paced, SP DDCL-Density, SP DDCL-Point), vòng lặp SPL theo Algorithm 2, DDCL Ensemble 6 thành viên bỏ phiếu đa số, Standard Ensemble (NN + SVM RBF C=1, gamma scale), Adam lr=0.001, λ = 0.4 → 0.6 hệ số 1.1, 200 epoch (đơn lẻ) / 600 epoch (ensemble), 30 lần chạy, chia 70/20/10.
2. **Ba chỗ code khác paper về bản chất**, đều ảnh hưởng tới việc so sánh số liệu:
   - Bước **SMOTE (optionalOversample) của Algorithm 1 không được dùng trong bất kỳ kết quả chính thức nào** (mặc định `none`, không config nào bật).
   - **MLP cố định (64, 32)**, không có Bayesian optimisation; paper dùng số tầng ẩn được tối ưu theo từng tập.
   - **Cách chia quantile** là thuật toán do nhóm tự thiết kế (`kde_level_partition_v1`), không phải hàm của tác giả.
3. **Phạm vi**: mới có 3/10 tập dữ liệu của paper (Haberman, New-Thyroid, Banknote). Các hình/bảng không tái lập được hoặc chưa làm: Table 4, Fig. 3–5 (xem §2).
4. **Kết quả**: ranking giữa các chiến lược **không tái lập được rõ ràng**: chênh lệch giữa 6 chiến lược trên New-Thyroid chỉ ≤ 0.6 điểm % trong khi độ lệch chuẩn ≈ 4.4; trên Haberman ≈ 3 điểm % so với độ lệch chuẩn 5.7–7.5. Số ensemble Table 3 của paper (đặc biệt Haberman 96.77%) lệch rất xa (−28 điểm %). Mục 5 giải thích nhiều khả năng đây là khác biệt giao thức đánh giá, chưa phải lỗi của code.
5. **Có 3 phát hiện trong code đáng xử lý** (mục 6): lỗi không gian đặc trưng khi bật SMOTE (tiềm ẩn), Standard Ensemble thực chất luôn chọn lớp đa số khi hai thành viên bất đồng, và 6 thành viên DDCL Ensemble dùng cùng một trọng số khởi tạo.

---

## 2. Phạm vi thực nghiệm so với paper

| Hạng mục paper | Trạng thái trong repo |
|---|---|
| Table 1: 6 tập cũ × 6 chiến lược | Chỉ Haberman, New-Thyroid (Breast Cancer, Cancer, Liver, Pima: chưa xác nhận người làm) |
| Table 2: 4 tập mới × 6 chiến lược | Chỉ Banknote (ISOLET, Optical Digits, Clean1: chưa xác nhận) |
| Table 3: Ensemble 600 epoch, 6 tập | Chỉ Haberman, New-Thyroid. Banknote không nằm trong Table 3 nên chạy thêm ở 200 epoch |
| Table 4: so sánh với SPAL, UFS_SP, SPKC, WSPLBF, ClusterGAN | [CHƯA LÀM] (không có code của các phương pháp đó; số liệu chỉ lấy từ paper) |
| Fig. 3 (đường loss theo epoch) | `trace.jsonl` có `train_loss` từng epoch, nhưng chưa thấy script vẽ lại hình |
| Fig. 4 (confusion matrix ISOLET, Digits) | [CHƯA LÀM] (tập chưa chạy) |
| Fig. 5 (cột so sánh) | [CHƯA LÀM] |
| Phân tích bổ sung của nhóm (không có trong paper) | R2 200 epoch, Controlled 6-NN, R3 độ nhạy λ ∈ {0.2, 0.4, 0.6} |

---

## 3. Đối chiếu từng thành phần thuật toán

| Thành phần | Paper | Code | Mức độ |
|---|---|---|---|
| Tâm lớp | Algorithm 1 dòng 2 (công thức viết `min` sai ký hiệu) | Trung bình từng lớp (`curriculum/geometry.py`) | [DIỄN GIẢI] (A03) |
| Khoảng cách | Euclid tới tâm lớp; "normalised" ở phần Point | L2 trên **raw features** (chưa chuẩn hóa), rồi chia cho `‖r‖₂` của lớp | [DIỄN GIẢI] (D04). Paper không nói rõ phép chuẩn hóa |
| KDE | `f̂ = 1/(|s|h) ΣK((x−x_i)/h)` trên `d_c` | `scipy.gaussian_kde` (Scott) trên **khoảng cách đã chuẩn hóa** | [DIỄN GIẢI] |
| Quantile | `getQuantile(f̂)`, Fig. 2a của bài Static DDCL có nhóm kích cỡ khác nhau | `kde_level_partition_v1`: cắt mức mật độ ngang đều từ 0 đến đỉnh, `n_levels = 5` | [DIỄN GIẢI] (A04). Số bin không cố định |
| Gộp bin giữa các lớp | Không mô tả | Gộp theo **chỉ số bin cục bộ** (bin 0 của lớp A cùng nhóm với bin 0 của lớp B) | [DIỄN GIẢI] (A05). Xem §6 về mức độ ghi nhận |
| Oversampling | `optionalOversample(q)` cho quantile mất cân bằng | `within_bin_minority_v1` có cài đặt, nhưng **mặc định `none` và không config nào bật** | [CHƯA LÀM] trong kết quả chính thức |
| DDCL-Density | Quantile đông mẫu = dễ, xếp từ nhiều → ít | Xếp bin theo số mẫu giảm dần, trong bin xếp theo khoảng cách tăng (`curriculum/scoring.py`) | [KHỚP] (thứ tự trong bin là lựa chọn thêm) |
| DDCL-Point | Khoảng cách ngắn = dễ, xếp từ dễ đến khó | Sort toàn bộ theo khoảng cách chuẩn hóa tăng dần | [KHỚP] |
| Self-Paced | SPL trên dữ liệu "không tiền xử lý làm thay đổi thứ tự gốc" | Hoán vị ngẫu nhiên cố định theo seed (giống No Curriculum) | [DIỄN GIẢI]. Paper không nói thứ tự gốc là ngẫu nhiên hay thứ tự file |
| No Curriculum | Không curriculum, không SPL | Hoán vị ngẫu nhiên cố định theo seed, **giữ nguyên thứ tự mọi epoch** (không shuffle lại) | [DIỄN GIẢI] |
| SPL: tính loss từng batch | `L_b = determineLoss(b)` | Eval, `no_grad`, trung bình loss từng batch (`training/loop.py`) | [KHỚP] |
| SPL: easy/hard | `L_b < λ` easy, `≥ λ` hard | Đúng như vậy; easy rỗng thì lấy batch có loss nhỏ nhất | [KHỚP] + fallback do nhóm thêm (A06) |
| SPL: Algorithm 2 dòng 10–13 | Train easy; nếu `L_E < λ` thì train `D ∪ sort(H)` | Train 1 lượt easy, đo lại loss easy, nếu `< λ` thì train lại toàn bộ batch theo loss tăng dần | [KHỚP] với pseudocode. Easy huấn luyện 2 lần trong 1 epoch. Phần mô tả bằng lời ("train for following epochs until...") có thể hiểu khác pseudocode, nhóm chọn theo pseudocode |
| λ | 0.4 đầu, max 0.6, hệ số 1.1 | `λ ← min(0.6, λ·1.1)` sau mỗi epoch | [DIỄN GIẢI] quy tắc cập nhật (D08). Hệ quả: λ chạm 0.6 từ epoch 6 và đứng yên 195 epoch còn lại |
| Kích thước batch | Không nêu | 32, `drop_last=false` | [DIỄN GIẢI] (A08 ghi `drop_last`; `batch_size=32` không có mục riêng) |
| MLP | Số tầng ẩn thay đổi, tối ưu Bayesian; Adam lr 0.001 | Cố định `(64, 32)`, ReLU, dropout 0, Kaiming init; Adam lr 0.001, weight decay 0 | [LỆCH] (A12 `pilot_fixed_backbone`) |
| Loss | Binary hoặc multiclass | 1 logit + BCEWithLogits / C logit + CrossEntropy | [KHỚP] |
| Chia dữ liệu | 70/20/10 | Stratified 70/20/10, cố định ID theo seed | [KHỚP] về chữ; xem §5 về tỷ lệ thực của tác giả |
| Validation | Có 20% nhưng paper không nói dùng làm gì | Chỉ theo dõi, không early stopping, lấy mô hình epoch cuối | [DIỄN GIẢI] |
| DDCL Ensemble | 6 NN, mỗi NN một phương pháp, bỏ phiếu đều | `ensemble.py`: hard voting đều, hòa 3–3 thì chọn lớp có prior train cao hơn | [KHỚP] + tie-break do nhóm thêm (A09) |
| Standard Ensemble | 1 NN + 1 SVM, đa số phiếu | Hard vote 2 thành viên + tie-break theo prior | [KHỚP] về thành phần; xem §6.2 về hệ quả |
| Số lần chạy | 30 | 30 seed (0–29) | [KHỚP] |

---

## 4. Kết quả so với paper (đơn vị: độ chính xác test %, trung bình 30 lần)

Nguồn: `reports/final/reproduction_final_summary.md`. Cột paper lấy từ Table 1/2.

| Tập | Chiến lược | Paper | Nhóm | Δ (điểm %) |
|---|---|---|---|---|
| Haberman | No Curriculum | 65.83 | 71.29 | +5.46 |
| | DDCL-Density | 69.17 | 72.47 | +3.31 |
| | Self-Paced | 64.86 | 69.57 | +4.71 |
| | SP DDCL-Density | 69.03 | 71.51 | +2.48 |
| New-Thyroid | No Curriculum | 93.14 | 95.61 | +2.47 |
| | DDCL-Density | 95.49 | 95.30 | −0.19 |
| | SP DDCL-Density | 96.08 | 95.61 | −0.47 |
| | SP DDCL-Point | 93.73 | 95.76 | +2.03 |
| Banknote | cả 6 chiến lược | 99.88 – 99.94 | 100.00 | +0.06 – +0.12 |

Nhận xét:
- Mức tuyệt đối của nhóm **cao hơn paper** ở hầu hết ô (đặc biệt Haberman). Các chiến lược của nhóm **gần như không phân biệt được** với No Curriculum: New-Thyroid chênh ≤ 0.6 điểm % với độ lệch chuẩn ≈ 4.4; Haberman chênh ≈ 3 điểm % với độ lệch chuẩn 5.7–7.5. Báo cáo gốc cũng ghi nhận độ lệch chuẩn của nhóm nhỏ hơn paper trên Haberman. **Chưa có kiểm định thống kê** nào trong repo; kết luận "chiến lược nào tốt hơn" nên tránh khi chưa có.
- Có một điểm trùng định tính với paper: Self-Paced thuần là chiến lược thấp nhất trên Haberman ở cả hai nguồn.
- Chi phí: SPL dùng khoảng 1.7–2.0 lần số bước tối ưu so với huấn luyện thường (ví dụ Haberman 2.4–2.6 nghìn so với 1.4 nghìn bước; Banknote ≈ 11.8–12 nghìn so với 6 nghìn), vì cơ chế replay. Paper không nêu chi phí này.

### Ensemble (Table 3)

| Tập | Mô hình | Paper | Nhóm (600 epoch) | Δ |
|---|---|---|---|---|
| Haberman | DDCL Ensemble | 96.77 | 68.39 | −28.39 |
| | Standard Ensemble | 90.32 | 74.52 | −15.81 |
| New-Thyroid | DDCL Ensemble | 100.00 | 95.61 | −4.39 |
| | Standard Ensemble | 95.45 | 94.85 | −0.61 |

---

## 5. Vì sao Table 3 của paper khó so sánh trực tiếp (suy luận từ số học, chưa kiểm chứng bằng mã tác giả)

Các giá trị phần trăm trong paper là phân số của số mẫu test nhỏ, nên có thể suy ra kích thước tập test:

- **Table 1 / Haberman**: 45.833 = 11/24, 79.167 = 19/24, 87.500 = 21/24, tập test khoảng **24 mẫu** (≈ 8% của 306). **New-Thyroid**: 82.353 = 14/17, tập test khoảng **17 mẫu** (≈ 8% của 215). Điều này phù hợp ghi chú trong D03 rằng code tác giả dùng khoảng 72/20/8. Nhóm dùng đúng 10% (≈ 31 và 22 mẫu).
- **Table 3 / Haberman**: 96.774 = 30/31 và 90.323 = 28/31, tập test **31 mẫu**. **New-Thyroid**: 95.455 = 21/22 và 100 = 22/22, tập test **22 mẫu**. Đây là tỷ lệ 10%, khác Table 1.
- Độ lệch chuẩn trong Table 3 đều là **±0.000** trên "30 lần chạy", tức mọi lần chạy cho cùng một kết quả. Nhiều khả năng Table 3 là một lần chia cố định (hoặc cùng một tập test) chứ không phải trung bình 30 lần chia ngẫu nhiên như nhóm đang làm.
- Điểm cao nhất trong Table 1 của Haberman là 87.5%, trong khi Table 3 báo 96.77% cho ensemble. Mức này cao hơn mọi lần chạy đơn lẻ tốt nhất.

Kết luận thận trọng: chênh lệch −28 điểm % trên Haberman nhiều khả năng đến từ khác biệt giao thức đánh giá (và có thể là một tập test thuận lợi), không nên đọc thành "tái lập thất bại". Báo cáo gốc cũng đã để mục này chờ kiểm chứng ở Phase F. Cách kiểm chứng đáng tin nhất là đọc mã tác giả (https://github.com/ECOLS-research-group/DDCL) để xác nhận cách chia dữ liệu trong ensemble.

---

## 6. Phát hiện trong code cần xử lý hoặc ghi nhận

Mỗi mục dưới đây đọc trực tiếp từ mã nguồn; **tôi chưa chạy thử** (xem §7).

1. **Bật SMOTE sẽ trộn hai không gian đặc trưng (lỗi tiềm ẩn, chưa ảnh hưởng kết quả hiện có).**
   `runner.py:prepare_curriculum_for_run` truyền `processed_train_X` (đã chuẩn hóa bằng StandardScaler) vào tham số `X_raw` của `resample_curriculum_bins`, trong khi tâm lớp (`geom.centroids`) và `class_norms` tính từ `raw_train_X` (chưa chuẩn hóa). Với mẫu tổng hợp, khoảng cách `‖x_synth(scaled) − μ(raw)‖` sẽ vô nghĩa. Vì không config nào bật SMOTE nên 2.070 lượt chạy hiện có không bị ảnh hưởng, nhưng ablation SMOTE (đang là "dự kiến") sẽ sai nếu không sửa. Đồng thời, các test chỉ có 7 lần nhắc tới smote/synth/n_levels, nên khả năng chưa test bao phủ trường hợp này.

2. **Standard Ensemble gần như luôn chọn lớp đa số khi hai thành viên bất đồng.**
   Với 2 thành viên (NN + SVM), mọi bất đồng đều là hòa 1–1 và được phá bằng prior của tập train (`ensemble.py:evaluate_standard_paper_baseline`). Comment trong code nhắc tới "average probabilities" nhưng không dùng. Hệ quả có thể thấy trong số liệu: Haberman có trung bình 3.4–5.4 phiếu hòa trên khoảng 31 mẫu test; độ chính xác cao nhất (74.5–74.8%) nhưng macro-F1 thấp nhất (52.4–53.4%), đúng kiểu thiên về lớp đa số (Haberman lớp 1 chiếm khoảng 73.5%, ước lượng từ phân bố dữ liệu, chưa kiểm tra lại với tập train thật). Nếu so "Standard Ensemble > DDCL Ensemble" trên Haberman, nên báo cùng macro-F1.

3. **Sáu thành viên DDCL dùng cùng trọng số khởi tạo.**
   `execute_single_run` gọi `create_mlp_model(seed=seed)` cho mọi chiến lược, nên với cùng seed, 6 mạng bắt đầu từ cùng một điểm. Đây là lựa chọn hợp lý để cô lập tác động của curriculum, nhưng làm giảm đa dạng của ensemble (bất đồng chỉ 0.4–6.7%), và có thể là lý do DDCL-6 và Controlled 6-NN cho kết quả trùng nhau (71.08 và 71.08 trên Haberman 200 epoch; 95.61 và 95.61 trên New-Thyroid). Paper không nói thành viên có khởi tạo độc lập hay không.

4. **Một số lựa chọn có trong code nhưng chưa được ghi đủ trong `ASSUMPTIONS.md`.**
   - `n_levels = 5` được hard-code trong `runner.py` và các script `tools/run_*` (mặc định của `partition_by_class` là 6; không đọc từ YAML). Các trường `curriculum.partition_policy`, `curriculum.bandwidth` ghi trong A04 không được runner đọc.
   - `batch_size = 32` không có mục giả định riêng.
   - Quy tắc "gộp bin theo chỉ số cục bộ" (A05) ảnh hưởng trực tiếp tới DDCL-Density kể cả khi SMOTE tắt (bin 0 của các lớp được tính chung), vì vậy cần xem như giả định áp dụng cho mọi kết quả Density, không chỉ cho SMOTE.
   - Thứ tự batch **không shuffle lại** giữa các epoch với 3 chiến lược không SPL, kể cả No Curriculum; khác với huấn luyện NN thông thường.

5. **Chi tiết nhỏ**: `train_single_epoch_spl` với `pace_policy` không hợp lệ sẽ im lặng bỏ qua lượt replay (`replay_batch_ids = []`) thay vì báo lỗi. Cấu hình hiện tại dùng mặc định hợp lệ nên chưa gây vấn đề.

---

## 7. Giới hạn của báo cáo này

- Thư mục `outputs/`, `hf_release/`, `data/`, `.venv/` **không có trên máy hiện tại** nên tôi không chạy `pytest` (README ghi 44 test pass) và không kiểm lại các số liệu `reports/`. Số liệu ở §4 là lấy nguyên từ báo cáo có sẵn.
- Phần suy luận kích thước tập test (§5) dựa trên phép chia số học, chưa được xác nhận bằng mã tác giả.
- Chưa so sánh chi tiết từng dòng Table 1 cho các ô không liệt kê ở §4 (xem đầy đủ trong báo cáo gốc).
- Chưa đọc nội dung `docs/02`, `04`, `05`, `07`, `10`, `12` và các notebook; có thể còn thông tin bổ sung ở đó.

## 8. Đề xuất việc tiếp theo (theo thứ tự ưu tiên)

1. Sửa lỗi không gian đặc trưng ở SMOTE (mục 6.1) và thêm test cho trường hợp `within_bin_minority_v1`, trước khi chạy bất kỳ ablation SMOTE nào.
2. Đọc mã tác giả để xác nhận: cách chia dữ liệu của ensemble (Table 3), `n_levels`/quantile, thứ tự dữ liệu "gốc" cho Self-Paced, và khởi tạo của các thành viên.
3. Báo cáo kèm macro-F1 và kiểm định thống kê (ví dụ Wilcoxon theo cặp seed) khi nhận xét tác động của curriculum.
4. Bổ sung `ASSUMPTIONS.md` các mục ở 6.4; cân nhắc thêm một biến thể Standard Ensemble dùng trung bình xác suất để tách ảnh hưởng của luật hòa phiếu.
