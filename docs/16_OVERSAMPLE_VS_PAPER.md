# Oversample: mô tả trong paper so với code hiện tại

Tài liệu nối tiếp [14_PARTITION_BY_CLASS.md](14_PARTITION_BY_CLASS.md) và [15_CURRICULUM_RESULT.md](15_CURRICULUM_RESULT.md). Nó đối chiếu bước oversample (SMOTE) mà hai bài báo mô tả với `within_bin_minority_v1` trong [`src/dddcl/curriculum/resample.py`](../src/dddcl/curriculum/resample.py), kèm các hành vi đã kiểm bằng cách chạy.

Ký hiệu mức độ dùng chung với [13_CODE_VS_PAPER.md](13_CODE_VS_PAPER.md): **[KHỚP]**, **[DIỄN GIẢI]** (paper không nêu rõ), **[LỆCH]** (khác paper), **[CHƯA LÀM]**.

Hai bài báo, gọi tắt:
- **Static**: Chaudhry et al., *Data Distribution-based Curriculum Learning*, IEEE Access 11 (2023), file [`static-ddcl.pdf`](static-ddcl.pdf).
- **Dynamic**: Chaudhry & Sharma, *Dynamic Data Distribution-based Curriculum Learning*, Information Sciences 702 (2025), file [`dynamic-ddcl.pdf`](dynamic-ddcl.pdf).

> **Mức kiểm chứng của tài liệu này.**
> - Phần paper: đọc trực tiếp bằng mắt các trang Static 3–9 và Dynamic 3–12 (phần phương pháp, thiết lập thí nghiệm, bảng, hình). Không có tìm kiếm văn bản toàn PDF, nên chưa loại trừ được một câu nhắc đến oversample ở các trang chưa đọc (trang 1–2 và phần tham khảo của hai bài).
> - Phần code: đọc `resample.py`, `runner.py`, `scoring.py`, config, test.
> - Các số ở §4 đến từ một lần chạy trên **dữ liệu giả lập 100 mẫu** (lớp 0: 80, lớp 1: 20), không phải dữ liệu thật của dự án. Script ở thư mục scratchpad của phiên làm việc, không nằm trong repo.
> - Chưa chạy `pytest`, chưa đọc mã tác giả. Thư mục kết quả chạy (`outputs/runs_final/`, …) không có trên máy này, chỉ có `outputs/splits/`.

---

## 1. Tóm tắt

1. **Paper mô tả oversample là bù quantile thưa.** Static Algorithm 1: với mỗi quantile `q` của từng lớp, nếu `|q| < 3` thì `q ← SMOTE(q)`. Mục đích ghi trong văn bản là "giải quyết tình trạng thiếu mẫu trong một quantile".
2. **Code cân bằng lớp trong bin gộp.** `within_bin_minority_v1` chạy trên bin gộp giữa các lớp theo chỉ số cục bộ, và nâng lớp ít mẫu trong bin lên bằng lớp nhiều mẫu.
3. **Hai cách làm khác nhau về đơn vị, điều kiện kích hoạt và mục tiêu** → **[LỆCH]**. Chúng gần như ngược nhau ở một điểm: quantile `|q| < 3` là đối tượng chính của paper, nhưng code bỏ qua bin có lớp ít nhất dưới 4 mẫu.
4. **Paper không nói đã bật oversample hay chưa trong thí nghiệm.** Không có bảng, hình hay tham số nào về oversample ở phần thực nghiệm của cả hai bài.
5. **Trong repo, oversample chưa bao giờ được bật.** Không config nào đặt `smote_policy`, báo cáo không nhắc tới, mặc định là `none`. Nhánh SMOTE cũng chưa có test.
6. **Có lỗi tiềm ẩn khi bật SMOTE** (không gian đặc trưng, bin không khớp khoảng cách, nuốt lỗi). Xem §5.

---

## 2. Paper mô tả gì

### 2.1 Static DDCL (nguồn chi tiết nhất)

**Algorithm 1 (trang 4), dòng 15–19:**

```
for q ∈ Q^s do
    if |q| < 3 then
        q ← SMOTE(q)
    end if
end for
```

`Q^s` là các quantile của lớp `s`, nên vòng lặp này nằm **trong** vòng `for all s ∈ S` (duyệt từng lớp).

**Văn bản (trang 4–5):**
- Sau khi chia quantile, oversample "được thực hiện tuỳ chọn trên các quantile có số mẫu có thể mất cân bằng cao so với các quantile khác".
- Mỗi quantile được xét số mẫu; các quantile **có ít mẫu nhất** được oversample bằng SMOTE "nếu có đủ mẫu".
- SMOTE được dùng "để giải quyết tình trạng thiếu mẫu trong một quantile". Nếu không đủ mẫu thì không oversample.

**Hình 1 (trang 4):** khối quyết định "Quantile oversampling required?"; nhánh Yes cho ra ba cột **Q1, Q1, Q2**, trong đó một cột Q1 được nối thêm phần xanh lá.

### 2.2 Dynamic DDCL (bản tóm tắt)

- **Algorithm 1** viết gọn: `q ← getQuantile(f̂_s)` rồi `q ← optionalOversample(q)`, không còn ngưỡng `|q| < 3`.
- **Văn bản (trang 5):** "oversampling được thực hiện trên các quantile có số mẫu mất cân bằng cao so với các quantile khác; quantile không mất cân bằng giữ nguyên".
- **Chú thích Hình 1 (trang 5):** "Quantiles are then calculated and optionally oversampled". Hình vẽ hai cột **C1, C2**, cột C2 thấp được nâng ngang C1. Nhãn C1/C2 không được giải thích; vì chú thích nói đối tượng là quantile và Hình 1 của Static dùng nhãn Q1/Q2 cho đúng khối này, nhiều khả năng đây là hai quantile, nhưng đó là suy luận.

### 2.3 Phần thực nghiệm

Không có thông tin về oversample ở các mục thiết lập và kết quả:

| Nguồn | Nội dung thiết lập | Nhắc oversample/SMOTE? |
|---|---|---|
| Dynamic mục 5.1 (trang 8) | NN tối ưu Bayesian, Adam lr 0.001, λ 0.4 → 0.6 hệ số 1.1, chia 70/20/10, 200 epoch, 30 lần chạy. Với SP DDCL-Density, dữ liệu "được tiền xử lý bằng Static DDCL" | Không. Không nêu bước tiền xử lý đó có gồm oversample hay không |
| Dynamic mục 5.2 (trang 10) | Ensemble 600 epoch, SVM C=1, RBF | Không |
| Static mục IV (trang 6–7) | NN, SVM, Random Forest, các kịch bản Density/Point | Không |
| Static Table 2–5, Fig. 3–5 | Kết quả theo chiến lược và tập dữ liệu | Không có biến thể có/không oversample |

**Một dấu vết số duy nhất.** Hình 2a của Static (Haberman) liệt kê 9 quantile với số mẫu 82, 76, 50, 41, 28, 19, 8, 2, 0. **Tổng bằng 306**, đúng bằng số mẫu gốc của Haberman trong Table 1. Hệ quả: số liệu trong hình là **trước khi oversample**, hoặc oversample không được áp dụng ở đó. Q2 chỉ có 2 mẫu (dưới ngưỡng `|q| < 3` của Algorithm 1) nhưng hình không cho biết có bị oversample hay không. Lưu ý 306 là cả tập, không riêng tập train.

### 2.4 Paper không nói gì

- `SMOTE(q)` sinh bao nhiêu mẫu (đến 3 mẫu, hay đến một mức khác).
- "Đủ mẫu" nghĩa là bao nhiêu mẫu, và `k_neighbors` dùng giá trị nào.
- `SMOTE` chạy thế nào trên một tập chỉ có **một lớp** (quantile của một lớp). Thư viện `imblearn` không chạy được trường hợp này: đã thử `SMOTE(k_neighbors=3).fit_resample(X, y)` với một lớp, kết quả là `ValueError: The target 'y' needs to have more than 1 class`.
- Có bật oversample trong thí nghiệm hay không (§2.3).

---

## 3. Code hiện tại làm gì

Hàm `resample_curriculum_bins` ([resample.py:29](../src/dddcl/curriculum/resample.py#L29)) được gọi từ [`runner.py:126`](../src/dddcl/runner.py#L126), sau `partition_by_class` và trước `create_curriculum`.

| Bước | Dòng | Nội dung |
|---|---|---|
| 1 | 51–60 | `smote_policy="none"`: trả bản sao dữ liệu, `n_synthetic=0` |
| 2 | 62–63 | Giá trị khác `none` và `within_bin_minority_v1` thì báo lỗi |
| 3 | 75–77 | Duyệt từng bin (chỉ số cục bộ, **đã gộp giữa các lớp**) |
| 4 | 82–84 | Bin chỉ có một lớp → `single_class_skipped` |
| 5 | 89–91 | Mọi lớp trong bin bằng số mẫu → `already_balanced` |
| 6 | 94–101 | Lớp ít mẫu nhất trong bin có `< k+1 = 4` mẫu → `insufficient_samples_for_k3` |
| 7 | 104–107 | `SMOTE(k_neighbors=3, random_state=seed)` trên `X_b` của bin |
| 8 | 114–123 | Khoảng cách mẫu mới `‖x_synth − μ_c‖ / class_norms[c]`; ID `synth_{dataset}_bin{b}_c{c}_{số đếm toàn cục}` |
| 9 | 125–130 | Nối mẫu mới vào cuối; **bin gán = bin đang xử lý** |
| 10 | 138–139 | Mọi ngoại lệ trong SMOTE → `smote_failed`, chạy tiếp |

**Mặc định và cấu hình.**
- `smote_policy` mặc định `"none"`, đọc ở cấp trên cùng của config: `raw_config.get("smote_policy", "none")` ([runner.py:232](../src/dddcl/runner.py#L232)).
- Không file nào trong `configs/` có khoá này; các script `tools/run_*` không truyền; `reports/`, `README.md`, `PROGRESS.md` không nhắc tới. → **Mọi kết quả hiện có không qua oversample.** Tôi không xác nhận trực tiếp bằng `resolved_config.json` vì thư mục kết quả chạy không có trên máy này.
- `ASSUMPTIONS.md` A05 ghi khoá là `curriculum.smote_policy`, nhưng runner đọc `smote_policy` ở cấp trên cùng. Đặt theo A05 thì runner bỏ qua âm thầm và vẫn chạy `none`.

---

## 4. Hành vi đã kiểm bằng cách chạy (dữ liệu giả lập)

Dữ liệu: 2 lớp (80 và 20 mẫu), 4 đặc trưng, `n_levels=5`. Chia bin cho 9 bin ở lớp 0 và 6 bin ở lớp 1.

```
bin 3: {lớp0: 8,  lớp1: 10} → smote_applied, +2 mẫu (lớp 0)
bin 5: {lớp0: 11, lớp1: 5}  → smote_applied, +6 mẫu (lớp 1)
bin 4: {lớp0: 46, lớp1: 1}  → insufficient_samples_for_k3
bin 0, 2                    → insufficient_samples_for_k3
bin 1, 6, 7, 8              → single_class_skipped
Tổng: +8 mẫu; lớp 0: 80 → 82, lớp 1: 20 → 26
```

Các quan sát:

1. **"Thiểu số" được xác định theo từng bin, không theo toàn tập.** Bin 3 sinh mẫu cho lớp 0, là lớp đa số toàn cục (`synth_demo_bin3_c0_0`).
2. **Mất cân bằng toàn cục gần như không đổi** (80:20 → 82:26).
3. **Bin mất cân bằng nặng nhất bị bỏ qua** (bin 4, 46 so với 1).
4. **Bin nhiều lớp bị bỏ cả bin nếu một lớp quá nhỏ.** Thử một bin có 3 lớp với 30, 12 và 2 mẫu: cả bin bị bỏ qua vì lớp 2 chỉ có 2 mẫu, dù lớp 1 so với lớp 0 hoàn toàn đủ điều kiện.
5. **Bin gán cho mẫu tổng hợp không đảm bảo khớp khoảng cách của nó.** Chỉ 3/8 mẫu có khoảng cách nằm trong đoạn `[b_j, b_{j+1}]` của bin được gán (theo biên của chính lớp đó); 5/8 nằm ngoài.
6. **SMOTE đổi thứ tự Density.** Số mẫu bin 3 từ 18 lên 20 và bin 5 từ 16 lên 22, nên thứ tự bin đổi từ `4, 3, 5, …` thành `4, 5, 3, …`.
7. **Không gian đặc trưng.** Truyền `X` chuẩn hoá thay vì thô, dải khoảng cách mẫu tổng hợp đổi từ 0.228–0.311 sang 0.234–0.399. Mức chênh ở đây nhỏ vì dữ liệu giả lập đã có thang gần 1; với dữ liệu thật có thang lệch (ví dụ Haberman: tuổi ~50, năm ~60, số hạch ~4) chênh lệch sẽ lớn hơn, nhưng chưa chạy trên dữ liệu thật.

---

## 5. So sánh từng khía cạnh

| Khía cạnh | Paper | Code | Mức độ |
|---|---|---|---|
| Oversample là bước tuỳ chọn | Có (`optionalOversample`, nhánh Yes/No ở Hình 1) | Có, `none` / `within_bin_minority_v1` | [KHỚP] |
| Vị trí trong pipeline | Sau chia quantile, trước chấm điểm | Sau `partition_by_class`, trước `create_curriculum` | [KHỚP] |
| Chỉ dùng dữ liệu train | Quy trình bắt đầu từ dữ liệu huấn luyện | Chỉ train, ID `synth_*` | [KHỚP] về ý (paper không nói tường minh) |
| Đơn vị xét | Quantile **của từng lớp** (`Q^s`) | Bin **gộp giữa các lớp** theo chỉ số cục bộ | [LỆCH] |
| Điều kiện kích hoạt | `|q| < 3` | Bin ≥ 2 lớp và lớp ít nhất ≥ 4 mẫu, lớp chưa cân bằng | [LỆCH] |
| Mục tiêu | Bù quantile thưa ("lack of samples in a quantile") | Cân bằng số mẫu giữa các lớp trong bin | [LỆCH] |
| Quantile chỉ có một lớp | Là đối tượng chính (mỗi quantile vốn một lớp) | Bỏ qua (`single_class_skipped`) | [LỆCH] (ngược nhau) |
| Quantile quá ít mẫu | Là đối tượng cần sinh thêm; "không đủ mẫu thì không sinh" | Bỏ qua nếu lớp ít nhất < 4 mẫu | [LỆCH] một phần |
| Số mẫu sinh ra | Không nêu | Lớp ít → bằng lớp nhiều trong bin (mặc định của `imblearn`) | [DIỄN GIẢI] |
| `k_neighbors` | Không nêu | 3 | [DIỄN GIẢI] |
| Khoảng cách mẫu mới | Không nêu | Tính lại theo tâm lớp thật | [DIỄN GIẢI] |
| Bin của mẫu mới | Không nêu | Bin đang xử lý, không kiểm lại | [DIỄN GIẢI] |
| Có bật trong thí nghiệm không | Không nêu | Không bật | Không so sánh được |
| Lọc "bin nào cần oversample" | "Quantile có ít mẫu nhất" | Xử lý mọi bin đủ điều kiện (spec gợi ý "nửa số bin ít mẫu nhất" nhưng chưa làm) | [CHƯA LÀM] |

Điểm cần nhớ: `DECISIONS.md` D06 có trích "Static DDCL Algorithm 1 (dòng 15-19)", tức đúng đoạn `|q| < 3` ở trên, nhưng lựa chọn (b) `within_bin_minority_v1` mô tả khác ("sinh mẫu cân bằng cho lớp thiểu số trong bin gộp"). Spec [03_METHOD_SPEC.md](03_METHOD_SPEC.md) mục 8 nói paper "cho phép oversampling nhóm ít mẫu nếu đủ mẫu", khá sát paper, nhưng phần cài đặt đi theo hướng cân bằng lớp. Tôi không biết lý do nhóm chọn như vậy.

---

## 6. Vấn đề kỹ thuật trong code (độc lập với paper)

Đã nêu chi tiết ở [13_CODE_VS_PAPER.md](13_CODE_VS_PAPER.md) §6.1 và xác nhận thêm ở §4:

1. **Không gian đặc trưng.** `runner.py` truyền `processed_train_X` (đã chuẩn hoá) vào tham số tên `X_raw`, trong khi `centroids` và `class_norms` tính từ dữ liệu thô. Khoảng cách của mẫu tổng hợp so không gian chuẩn hoá với tâm thô.
2. **Bin và khoảng cách có thể mâu thuẫn** (§4, điểm 5).
3. **`except Exception` nuốt mọi lỗi** ([resample.py:138](../src/dddcl/curriculum/resample.py#L138)). Lỗi thật chỉ hiện thành một dòng `smote_failed` trong báo cáo.
4. **Báo cáo thiếu số liệu sau resample.** Chỉ có `before` trong mỗi bin, không có `after`.
5. **Khoá config không khớp tài liệu** (§3).
6. **Chưa có test cho nhánh SMOTE.** Chỉ một chỗ gọi `resample_curriculum_bins` trong test, với `smote_policy="none"` ([test_curriculum.py:221-230](../tests/test_curriculum.py#L221-L230)). `ASSUMPTIONS.md` A05 trích `tests/test_resample.py` nhưng file này không tồn tại.

---

## 7. Hệ quả cho việc so sánh với paper

- Vì paper không nói đã bật oversample hay chưa, việc nhóm tắt SMOTE **không thể** gọi là "khác paper" hay "giống paper". Chỉ nói được là paper không cung cấp thông tin.
- Chênh lệch tuyệt đối giữa số của nhóm và Table 1–3 của paper không nên quy một phần cho oversample khi chưa có bằng chứng.
- Ablation oversample trong kế hoạch ([04_PROTOCOLS.md](04_PROTOCOLS.md), [05_EMPIRICS.md](05_EMPIRICS.md)) nên ghi rõ là nhóm tự thêm để kiểm tra một cách diễn giải, không phải để tái lập chi tiết paper nêu. Nếu thực hiện, cần chỉ rõ đang kiểm tra cách hiểu nào: bù quantile thưa (theo Algorithm 1) hay cân bằng lớp trong bin (hiện tại).

---

## 8. Việc nên làm (đề xuất, chưa thực hiện)

Theo thứ tự ưu tiên:

1. **Sửa lỗi không gian đặc trưng** (§6, điểm 1) và thêm test cho `within_bin_minority_v1` trước khi chạy bất kỳ ablation SMOTE nào: ID duy nhất, không có mẫu tổng hợp trong validation/test, bin và khoảng cách nhất quán.
2. **Quyết định cách hiểu oversample cần kiểm tra.** Nếu muốn theo Algorithm 1, cần giả định ba điểm paper bỏ ngỏ (§2.4): quantile `|q| < 3` được sinh tới bao nhiêu mẫu, `k_neighbors`, và cách nội suy khi quantile chỉ có một lớp (vì `imblearn.SMOTE` không chạy với một lớp, cần cài nội suy riêng). Nếu giữ cân bằng lớp trong bin, nên đổi tên cho sát hành vi (ví dụ `within_bin_balance_v1`) và ghi rõ là **[LỆCH]** so với Algorithm 1 trong `ASSUMPTIONS.md`.
3. **Sửa tài liệu**: khoá config `smote_policy` (cấp trên cùng, không phải `curriculum.smote_policy`), và cập nhật D06/A05 để phản ánh đúng quan hệ với Algorithm 1 của Static.
4. **Thu hẹp `except Exception`** hoặc ghi ngoại lệ vào báo cáo; thêm số liệu `after` vào báo cáo từng bin.
5. **Đọc mã tác giả** (https://github.com/ECOLS-research-group/DDCL) để xác nhận: oversample có bật trong thí nghiệm không, `SMOTE(q)` sinh bao nhiêu mẫu, `k_neighbors`, và quantile được chia theo từng lớp hay đã gộp. Theo [12_SOURCES.md](12_SOURCES.md) việc này chỉ làm sau khi bản độc lập hoàn tất.
