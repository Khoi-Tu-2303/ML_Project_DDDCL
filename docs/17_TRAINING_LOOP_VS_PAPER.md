# `training/loop.py`: vòng lặp huấn luyện và đối chiếu với paper

Tài liệu nối tiếp [14](14_PARTITION_BY_CLASS.md), [15](15_CURRICULUM_RESULT.md) và [16](16_OVERSAMPLE_VS_PAPER.md). Nó giải thích [`src/dddcl/training/loop.py`](../src/dddcl/training/loop.py) (cùng [`pace.py`](../src/dddcl/training/pace.py) và [`checkpoint.py`](../src/dddcl/training/checkpoint.py) khi liên quan), rồi đối chiếu với phần Self-Paced Learning của paper Dynamic DDCL (Chaudhry & Sharma, Information Sciences 702, 2025; file [`dynamic-ddcl.pdf`](dynamic-ddcl.pdf)).

Mỗi mục có ba phần: **làm gì**, **vì sao**, **so với paper**. Ký hiệu mức độ dùng chung với [13_CODE_VS_PAPER.md](13_CODE_VS_PAPER.md): **[KHỚP]**, **[DIỄN GIẢI]** (paper không nêu rõ), **[LỆCH]**, **[CHƯA LÀM]**.

> **Mức kiểm chứng.**
> - Phần paper: đọc trực tiếp Algorithm 2, mục 4.1.2 (trang 6), mục 5.1 (trang 8) và các bảng liên quan. Không tìm kiếm văn bản toàn PDF.
> - Phần code: đọc `loop.py`, `pace.py`, `checkpoint.py`, phần gọi trong `runner.py` và `models/mlp.py`.
> - Các số của Haberman lấy từ **output đã lưu** trong [notebook 03](../notebooks/03_training_walkthrough.ipynb); tôi không chạy lại notebook. Các số khác được ghi rõ là từ một lần chạy trên **dữ liệu giả lập** (213 mẫu nhị phân, 7 batch), không phải dữ liệu thật.
> - Chưa chạy `pytest`, chưa đọc mã tác giả.

---

## 0. Tổng quan

```
CurriculumResult.ordered_train_ids
   │  create_curriculum_batches        cắt batch liên tiếp, không shuffle
   ▼
list[BatchData]   (cố định cho cả lần chạy)
   │
   ▼  train_dddcl_model(strategy, epochs, λ…)
   │     mỗi epoch:
   │        non-SPL ─ train_single_epoch_standard   (mọi batch, đúng thứ tự)
   │        SPL     ─ train_single_epoch_spl        (lọc dễ/khó theo loss và λ)
   │           └ update_lambda sau epoch
   │     đánh giá validation (chỉ theo dõi)
   │     ghi record, lưu checkpoint
   ▼
(model, history)  →  runner đánh giá test ĐÚNG MỘT LẦN
```

| Thành phần | Dòng | Mức độ so với paper |
|---|---|---|
| `BatchData`, `EpochAccounting` | 38–64 | Không có trong paper |
| `create_curriculum_batches` | 67–129 | [DIỄN GIẢI] (kích thước batch, không shuffle) |
| `evaluate_batch_losses` | 132–150 | [KHỚP] (`determineLoss(b)`) |
| `evaluate_dataset` | 153–184 | Không có trong paper (chỉ theo dõi và đánh giá cuối) |
| `train_single_epoch_spl` | 187–274 | [KHỚP] với pseudocode, [DIỄN GIẢI] một số chi tiết |
| `train_single_epoch_standard` | 277–311 | [DIỄN GIẢI] |
| `train_dddcl_model` | 314–471 | [DIỄN GIẢI] về λ, checkpoint, chọn mô hình |

---

## 1. Dựng batch: `create_curriculum_batches`

**Làm gì.** Lấy `ordered_train_ids` từ `CurriculumResult`, cắt thành các đoạn liên tiếp `batch_size` mẫu (`drop_last=False`), tra chỉ số hàng theo ID rồi lấy `X`, `y`. Không có shuffle ở bất kỳ bước nào.

Ví dụ thật từ notebook 03 (Haberman, 213 mẫu train, `batch_size=32`): 7 batch, gồm 6 batch × 32 mẫu và 1 batch × 21 mẫu.

**Vì sao.**
- Thứ tự của curriculum chỉ có ý nghĩa nếu batch giữ nguyên thứ tự đó. Shuffle sẽ phá curriculum.
- `drop_last=False` để không bỏ mẫu nào, nên batch cuối nhỏ hơn.
- Tra theo ID (`id_to_idx`) thay vì theo vị trí giúp ghép đúng mẫu kể cả khi có mẫu `synth_*` ở cuối danh sách.

**So với paper.** Paper không nêu kích thước batch, và Algorithm 2 dùng `p` (tập batch) mà không định nghĩa cách chia. Chọn 32 và `drop_last=False` là **[DIỄN GIẢI]** của nhóm. `runner.py` hard-code 32 và không có mục giả định riêng (đã ghi ở [13_CODE_VS_PAPER.md](13_CODE_VS_PAPER.md) §6).

**Hệ quả.** Batch được cắt một lần và giữ nguyên cho mọi epoch. Với chiến lược `sp_ddcl_*`, thứ tự curriculum chỉ quyết định **mẫu nào nằm trong batch nào** (xem §3).

---

## 2. Chấm điểm batch: `evaluate_batch_losses`

**Làm gì.** Đặt model sang `eval()`, dùng `torch.no_grad()`, tính `compute_loss(..., reduction="mean")` cho từng batch, rồi khôi phục chế độ train trong `finally`.

**Vì sao.** Chấm điểm không được cập nhật mô hình và không bị dropout làm nhiễu (yêu cầu ở [CLAUDE.md](CLAUDE.md) và [03_METHOD_SPEC.md](03_METHOD_SPEC.md)). Hiện cấu hình dùng `dropout=0`, nên `eval()` và `train()` cho cùng loss.

**So với paper.** Algorithm 2 dòng 5: `L_b = determineLoss(b)`. Cách tính cụ thể không được nêu; loss trung bình theo mẫu trong từng batch là cách đọc tự nhiên → **[KHỚP]** về ý, chi tiết là [DIỄN GIẢI].

---

## 3. Một epoch SPL: `train_single_epoch_spl`

### 3.1 Các bước

| Bước | Dòng | Nội dung |
|---|---|---|
| 1 | 199 | `batch_losses = evaluate_batch_losses(...)` ở chế độ `eval` |
| 2 | 202–205 | `compute_pace_decision`: dễ khi `L_b < λ`, khó khi `L_b ≥ λ`; mỗi nhóm sắp theo (loss, batch_id) tăng dần |
| 2b | `pace.py:74-80` | Nếu tập dễ rỗng, lấy **một** batch có loss nhỏ nhất làm dễ (fallback) |
| 3 | 217–229 | **Lượt 1**: huấn luyện từng batch dễ theo loss tăng dần, mỗi batch một bước optimizer |
| 4 | 235–245 | Đánh giá lại loss trung bình của các batch dễ; nếu `< λ` thì replay |
| 5 | 249–252 | **Lượt 2** (chính sách mặc định `replay_easy_then_union_v1`): huấn luyện **mọi** batch (dễ và khó), sắp theo loss đã tính ở bước 1 |

Chính sách `hard_only_second_pass` chỉ huấn luyện batch khó ở lượt 2 (ablation). Giá trị `pace_policy` không hợp lệ làm `replay_batch_ids = []`, tức **im lặng bỏ qua lượt 2** (dòng 256–257).

### 3.2 Vì sao

- Đơn vị lọc là **batch**, vì paper nói phản hồi động ở mức batch (`foreach b ∈ p`).
- Loss tính một lần ở đầu epoch và dùng để sắp thứ tự cho cả hai lượt, nên thứ tự huấn luyện trong epoch xác định và có thể ghi lại.
- Đo lại loss của tập dễ sau lượt 1 là cách hiểu điều kiện `L_ε < λ` ở dòng 11 như một đại lượng sau khi huấn luyện dễ.
- Fallback batch tốt nhất tránh trường hợp epoch không huấn luyện gì mãi mãi (A06).

### 3.3 So với paper

| Dòng paper (Algorithm 2) | Code | Mức độ |
|---|---|---|
| 5: `L_b = determineLoss(b)` | `evaluate_batch_losses`, `eval` + `no_grad` | [KHỚP] |
| 6–7: `ℰ = ℰ ∪ {b | L_b < λ}`, `ℋ = ℋ ∪ {b | L_b ≥ λ}` | `compute_pace_decision` | [KHỚP] (đúng cả quy ước dấu bằng) |
| 9: `D ← sort(ℰ)` | Sắp batch dễ theo loss tăng | [KHỚP] |
| 10: `L_ℰ = train(D)` | Lượt 1 | [KHỚP] |
| 11: `if L_ℰ < λ` | Đo lại loss tập dễ sau lượt 1, so với λ | [DIỄN GIẢI] (paper không nói `L_ℰ` là loss trước hay sau huấn luyện) |
| 12–13: `D = D ∪ sort(ℋ)`, `L_ℰ = train(D)` | Lượt 2 trên tập hợp dễ ∪ khó | [KHỚP] |
| Tập dễ rỗng | Không nêu | Bổ sung của nhóm (A06) |

**Về thứ tự trong lượt 2.** Pseudocode viết `D ∪ sort(ℋ)`, tức tập dễ đã sắp rồi đến tập khó đã sắp. Code sắp **chung** mọi batch theo loss ở dòng 251. Vì mọi batch dễ có `L_b < λ ≤ L_b` của batch khó, sắp chung cho cùng thứ tự "toàn bộ dễ trước, rồi toàn bộ khó". Hai cách tương đương.

**Chỗ mô tả bằng lời khác pseudocode.** Mục 4.1.2 (trang 6) viết: dữ liệu đã lọc "được dùng để huấn luyện mô hình **trong các epoch tiếp theo cho tới khi** loss tổng thể của mô hình trở nên dễ (nhỏ hơn ngưỡng). Khi đó phần dữ liệu khó được đưa trở lại". Câu này có thể hiểu là huấn luyện **nhiều epoch chỉ trên tập dễ**, trong khi pseudocode và code làm trong **một** epoch (dễ, kiểm tra, rồi dễ ∪ khó). Nhóm chọn theo pseudocode (D07, `replay_easy_then_union_v1`). Cả hai cách đọc đều có cơ sở; chỉ mã tác giả mới phân định.

### 3.4 Hành vi đã quan sát

**Haberman thật (output đã lưu trong notebook 03, epoch 1, λ = 0.4, Adam lr 0.01):**

```
Loss từng batch trước khi train: 0.7274 0.8509 0.6025 0.5742 0.6711 0.7458 1.6927
Dễ: [3]   Khó: [2, 4, 0, 5, 1, 6]   Replay: False
Số bước optimizer: 1   Số lượt mẫu tiếp xúc: 32
```

Loss nhỏ nhất là 0.5742, lớn hơn λ = 0.4, nên tập dễ rỗng và chạy fallback. Sau một bước, loss batch đó chưa xuống dưới 0.4 nên không replay, và 6 batch còn lại không được huấn luyện trong epoch này.

**Dữ liệu giả lập** (213 mẫu nhị phân, 7 batch, `lr=0.001`, λ mặc định, `self_paced`):

```
epoch  λ      dễ  khó  replay  số bước
  1    0.400   1   6   False     1
  2    0.440   1   6   False     1
  3    0.484   1   6   True      8
  4    0.532   5   2   True     12
  5    0.586   7   0   True     14
 6+    0.600   7   0   True     14
```

Khi mọi batch đã thành "dễ" (tập khó rỗng), SPL **không còn lọc gì**: nó huấn luyện mọi batch hai lần mỗi epoch. Điều này khớp [13_CODE_VS_PAPER.md](13_CODE_VS_PAPER.md) §4 (SPL dùng khoảng 1.7–2.0 lần số bước so với huấn luyện thường). Thời điểm xảy ra trên dữ liệu thật của dự án sẽ khác; tôi chưa chạy.

### 3.5 Điểm cần lưu ý

1. **Loss dùng để sắp lượt 2 là loss cũ.** Nó được tính ở đầu epoch (bước 1), không tính lại sau lượt 1.
2. **`mean_easy_loss` là trung bình các loss batch, không có trọng số theo kích thước.** Batch cuối (21 mẫu) tính ngang batch 32 mẫu.
3. **Batch dễ được huấn luyện hai lần mỗi epoch** khi có replay (tăng số bước và số lần tiếp xúc).
4. **Khi không replay, batch khó không được huấn luyện gì ở epoch đó.** Điều này xảy ra ở những epoch đầu của Haberman.
5. **Thứ tự curriculum chỉ còn ảnh hưởng qua thành phần batch.** Trong mỗi epoch SPL, thứ tự huấn luyện là theo loss chứ không theo thứ tự curriculum.

---

## 4. Epoch thường: `train_single_epoch_standard`

**Làm gì.** `model.train()`, duyệt mọi batch theo thứ tự đã cắt, mỗi batch một bước optimizer. Không lọc, không replay. Dùng cho `no_curriculum`, `ddcl_density`, `ddcl_point`.

**Vì sao.** Đây là các chiến lược chỉ khác nhau ở thứ tự mẫu (curriculum tĩnh), nên vòng lặp huấn luyện giống hệt nhau.

**So với paper.** Static DDCL chỉ định thứ tự huấn luyện; paper không mô tả vòng lặp cụ thể → **[DIỄN GIẢI]**. Lưu ý quan trọng: **thứ tự batch không bị shuffle lại giữa các epoch**, kể cả `no_curriculum`, khác huấn luyện NN thông thường (đã ghi ở [13_CODE_VS_PAPER.md](13_CODE_VS_PAPER.md) §6). Kể cả `no_curriculum` cũng chỉ là một hoán vị ngẫu nhiên cố định.

---

## 5. Vòng lặp tổng: `train_dddcl_model`

### 5.1 Cấu trúc

1. `is_spl = strategy in {"self_paced", "sp_ddcl_density", "sp_ddcl_point"}`.
2. **Resume** (dòng 382–392): nếu có checkpoint, khôi phục trọng số, optimizer, RNG, epoch, λ, lịch sử và bộ đếm bước/tiếp xúc.
3. **Mỗi epoch** (dòng 394–453): chạy SPL hoặc epoch thường; cập nhật λ cho epoch kế (`min(λ_max, λ · growth)`); đánh giá validation; ghi `record`; chuyển `current_lambda = next_lambda`; lưu checkpoint.
4. Trả về `(model, history)`. Mô hình cuối là mô hình **sau epoch cuối**, không có early stopping.

### 5.2 Cập nhật λ

[`update_lambda`](../src/dddcl/training/pace.py#L104): `λ_{e+1} = min(λ_max, λ_e · growth)`, gọi sau **mỗi** epoch SPL bất kể có replay hay không. Với 0.4, 0.6 và 1.1: λ = 0.4, 0.44, 0.484, 0.5324, 0.5856, rồi 0.6 từ epoch 6 và giữ nguyên đến hết.

**So với paper.** Algorithm 2 không chứa bước cập nhật λ. Paper chỉ ghi (mục 4.1.2) "tham số SPL như ngưỡng và hệ số tăng được đặt trong mã nguồn" và (mục 5.1) "λ khởi đầu 0.4, tối đa 0.6, hệ số tăng 1.1". Ba con số này **[KHỚP]**; quy tắc "nhân hệ số mỗi epoch" là **[DIỄN GIẢI]** (D08/A06). Hệ quả: SPL chỉ "dần nới" trong khoảng 5 epoch đầu rồi chạy với ngưỡng cố định 0.6 trong khoảng 195 epoch còn lại.

### 5.3 Validation và đánh giá test

- Mỗi epoch, nếu có `X_val`, tính loss/accuracy/macro-F1 trên validation. Chỉ để theo dõi; **không** dùng để chọn mô hình hay dừng sớm.
- Test được đánh giá **đúng một lần** ở `runner.py` sau khi huấn luyện xong, bằng `evaluate_dataset` ([runner.py:324](../src/dddcl/runner.py#L324)).

**So với paper.** Paper chia 70/20/10 nhưng **không nói** tập validation 20% dùng để làm gì → **[DIỄN GIẢI]** (chỉ theo dõi). Lấy mô hình epoch cuối cũng là chọn của nhóm.

### 5.4 Bản ghi mỗi epoch (`record`)

Gồm `train_loss`, `pass2_loss`, `val_*`, `lambda`, `optimizer_steps`, `total_optimizer_steps`, `sample_exposures`, `easy_count`, `hard_count`, `replayed`, `fallback_used`, `duration_seconds`. Ghi qua `tracker.log_epoch(record)` nếu có tracker.

**Lưu ý.** `train_loss` bằng `pass1_loss` = trung bình loss tính **trong lúc huấn luyện lượt 1**. Với SPL đó là loss trên **các batch dễ**; với epoch thường đó là loss trên **mọi batch**. Hai đại lượng không cùng tập, nên đường `train_loss` của SPL và của chiến lược thường **không so trực tiếp** được với nhau. Paper vẽ loss theo epoch (Dynamic Fig. 3, "average error loss"), nhưng không nói rõ loss đó tính trên tập nào, nên khi vẽ lại cần chọn cách tính thống nhất.

### 5.5 Checkpoint và resume

- Lưu mỗi epoch (`checkpoint_freq=1` ở `runner.py`), ghi vào file tạm rồi `os.replace` để nguyên tử.
- Payload: trạng thái model, optimizer, `current_lambda` (đã là λ của epoch kế), `run_fingerprint`, bộ đếm, lịch sử, trạng thái RNG (Python, NumPy, PyTorch CPU).
- `load_checkpoint` báo lỗi nếu fingerprint không khớp.
- Notebook 03 (ô 9) chứng minh: train 4 epoch liên tục và train 2 epoch + resume 2 epoch cho **cùng loss từng epoch** (`0.611013, 0.494819, 0.378455, 0.296939`) và **trọng số giống hệt bit** (`torch.equal`).

**So với paper.** Paper không nói về checkpoint; đây là kỹ thuật bổ sung của nhóm để chạy ma trận lớn và tái lập được.

---

## 6. Đối chiếu tổng hợp với paper

| Nội dung | Paper | Code | Mức độ |
|---|---|---|---|
| Chia easy/hard theo loss batch và λ | Algorithm 2, dòng 5–7 | `compute_pace_decision` | [KHỚP] |
| Sắp tập dễ theo loss tăng | Dòng 9 | Có | [KHỚP] |
| Train dễ rồi kiểm tra `L_ℰ < λ` | Dòng 10–11 | Có, đo lại sau lượt 1 | [KHỚP] pseudocode / [DIỄN GIẢI] về `L_ℰ` |
| Đưa tập khó vào lại và train `D ∪ sort(ℋ)` | Dòng 12–13 | Replay tập hợp, sắp theo loss | [KHỚP] |
| Mô tả bằng lời: train nhiều epoch trên tập dễ trước khi đưa khó vào | Mục 4.1.2 | Không (làm trong một epoch) | Hai cách đọc; nhóm chọn pseudocode |
| Quy tắc cập nhật λ | Không có | `min(0.6, λ·1.1)` mỗi epoch | [DIỄN GIẢI] |
| λ khởi đầu, tối đa, hệ số tăng | 0.4, 0.6, 1.1 (mục 5.1) | 0.4, 0.6, 1.1 | [KHỚP] |
| Tập dễ rỗng | Không nêu | Lấy batch loss nhỏ nhất | Bổ sung (A06) |
| Kích thước batch | Không nêu | 32, `drop_last=False` | [DIỄN GIẢI] |
| Shuffle giữa các epoch | Không nêu | Không shuffle | [DIỄN GIẢI] |
| Số epoch, optimizer | 200 (ensemble 600), Adam lr 0.001 | Cấu hình theo YAML, lr mặc định 0.001 | [KHỚP] |
| Mô hình | Số tầng ẩn tối ưu Bayesian | `(64, 32)` cố định | [LỆCH] (A12) |
| Dùng tập validation | Có 20%, không nêu mục đích | Chỉ theo dõi, lấy mô hình epoch cuối | [DIỄN GIẢI] |
| Chi phí tính toán của SPL | Không nêu | Khoảng 1.7–2.0× số bước so với thường | Phát hiện của nhóm |

---

## 7. Hạn chế và việc nên làm

1. **`pace_policy` không hợp lệ bị bỏ qua im lặng** (dòng 256–257). Nên kiểm tra giá trị hợp lệ ở đầu và báo lỗi. Hiện cấu hình dùng mặc định hợp lệ nên chưa gây vấn đề.
2. **`train_loss` có nghĩa khác nhau giữa SPL và epoch thường** (§5.4). Nếu cần so đường loss với Dynamic Fig. 3, nên thêm một loss thống nhất, ví dụ loss `eval` trên **toàn bộ** train ở cuối mỗi epoch.
3. **Dùng `mean_easy_loss` có trọng số theo kích thước batch** là một lựa chọn cần cân nhắc (hiện batch cuối nhỏ tính ngang batch đầy).
4. **Các epoch đầu có thể chỉ huấn luyện 1 batch** (Haberman, epoch 1: 1 bước). Với λ = 0.4 và loss ban đầu ≈ 0.57–1.7, nhiều epoch đầu gần như không học. Nên ghi rõ trong báo cáo, và cân nhắc chạy đối chứng với λ khởi đầu cao hơn như một ablation riêng (R3 độ nhạy λ đã có sẵn trong kế hoạch).
5. **Quy tắc λ chạm 0.6 từ epoch 6** nên được ghi như một giả định ảnh hưởng tới toàn bộ kết quả SPL, không chỉ như chi tiết cài đặt.
6. **Cần đọc mã tác giả** (https://github.com/ECOLS-research-group/DDCL) để xác nhận: khi nào λ tăng, `L_ℰ` đo ở thời điểm nào, huấn luyện tập dễ trong một epoch hay nhiều epoch, kích thước batch, và có shuffle giữa các epoch hay không. Theo [12_SOURCES.md](12_SOURCES.md) việc đó chỉ làm sau khi bản độc lập hoàn tất.
