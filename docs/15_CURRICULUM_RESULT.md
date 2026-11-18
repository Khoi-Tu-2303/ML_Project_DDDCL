# `create_curriculum` và `CurriculumResult`: xếp thứ tự mẫu và đóng gói kết quả

Tài liệu nối tiếp [14_PARTITION_BY_CLASS.md](14_PARTITION_BY_CLASS.md). Hàm `partition_by_class` cho ra chỉ số bin cục bộ; tài liệu này giải thích bước kế tiếp trong [`src/dddcl/curriculum/scoring.py`](../src/dddcl/curriculum/scoring.py): từ `bin_assignments` và khoảng cách chuẩn hóa tới danh sách ID đã xếp thứ tự, điểm số, mã băm chính sách, và đối tượng `CurriculumResult`. Trọng tâm là nhánh `ddcl_density` / `sp_ddcl_density` ([scoring.py:91-125](../src/dddcl/curriculum/scoring.py#L91-L125)), có đối chiếu với hai nhánh còn lại.

Mỗi bước có ba phần: **làm gì**, **vì sao**, **so với paper**. Ký hiệu mức độ dùng chung với [13_CODE_VS_PAPER.md](13_CODE_VS_PAPER.md): **[KHỚP]**, **[DIỄN GIẢI]**, **[LỆCH]**, **[CHƯA LÀM]**.

> **Về nguồn và mức kiểm chứng.**
> - Phần "so với paper" dựa trên việc đọc trực tiếp PDF `dynamic-ddcl.pdf` các trang 3–8 (Algorithm 1–3, mục 4.1.1, 4.1.2, 4.2, 5.1), cộng với tài liệu đối chiếu có sẵn trong repo.
> - Các ví dụ số trong §3 và §6 đã được chạy thật bằng `create_curriculum` trong `.venv` của dự án; kết quả khớp với những gì ghi ở đây.
> - Chưa chạy `pytest`, chưa đọc mã tác giả. Mọi nhận định về "tác giả làm gì" ngoài những gì PDF viết rõ đều là suy luận.

---

## 0. Tổng quan

```
sample_ids, normalized_distances, bin_assignments   (sau partition và SMOTE tùy chọn)
   │
   ▼  create_curriculum(strategy, ..., seed)
   │     ├─ kiểm tra độ dài và strategy
   │     ├─ dist_map, bin_map                         (tra cứu theo ID)
   │     ├─ nhánh theo strategy  →  ordered_ids, scores
   │     └─ policy_hash
   ▼
CurriculumResult (frozen)
   ├─ ordered_train_ids   ◄── thứ tự huấn luyện, thứ DUY NHẤT được dùng về sau
   ├─ distances, group_ids, scores, resampling_report, policy_hash   ◄── chỉ để ghi vết
```

| Bước | Nội dung | Mức độ so với paper |
|---|---|---|
| 1 | Kiểm tra đầu vào, dựng `dist_map` và `bin_map` | Không có trong paper |
| 2 | Chọn nhánh theo strategy | [KHỚP] về khung 6 strategy |
| 3 | Nhánh density: xếp bin theo số mẫu giảm dần | [KHỚP] |
| 4 | Nhánh density: xếp mẫu trong bin theo khoảng cách | [DIỄN GIẢI] |
| 5 | Gán `scores` | Không có trong paper |
| 6 | Tính `policy_hash` | Không có trong paper |
| 7 | Đóng gói `CurriculumResult` | Không có trong paper |

Điểm cần nhớ trước: huấn luyện chỉ đọc `ordered_train_ids` ([loop.py:108](../src/dddcl/training/loop.py#L108)). Các trường còn lại không ảnh hưởng tới kết quả huấn luyện (xem §5).

---

## 1. Kiểm tra đầu vào và dựng bảng tra

**Làm gì.**

```python
if len(normalized_distances) != n_samples or len(bin_assignments) != n_samples:
    raise ValueError(...)
dist_map = {sid: float(d) for sid, d in zip(sample_ids, normalized_distances)}
bin_map  = {sid: int(b) for sid, b in zip(sample_ids, bin_assignments)}
```

Sau đó kiểm tra `strategy` thuộc 6 giá trị hợp lệ, nếu không thì báo lỗi kèm danh sách giá trị được phép.

**Vì sao.** Ba mảng song song theo vị trí (`sample_ids`, khoảng cách, bin) mà lệch độ dài thì sẽ ghép sai mẫu mà không báo lỗi. Chuyển sang từ điển theo ID giúp các bước sau (ensemble, báo cáo) tra theo ID thay vì theo vị trí mảng, đúng quy tắc "kết quả khóa theo sample ID" trong [CLAUDE.md](CLAUDE.md). Ép `float` / `int` để bỏ kiểu NumPy, tuần tự hóa JSON được.

**So với paper.** Paper không có phần này; đây là kỹ thuật của nhóm.

---

## 2. Ba họ thứ tự

| Strategy | Khoá sắp xếp | Điểm (`scores`) |
|---|---|---|
| `no_curriculum`, `self_paced` | Hoán vị ngẫu nhiên cố định `RandomState(seed).permutation(n)` | vị trí (rank), 0..n-1 |
| `ddcl_point`, `sp_ddcl_point` | `(distance ↑, str(sample_id))` trên toàn tập | chính khoảng cách |
| `ddcl_density`, `sp_ddcl_density` | Hai tầng, mục 3 và 4 | số mẫu của bin chứa mẫu đó |

Tiền tố `sp_` không đổi thứ tự cơ sở. Nó chỉ làm vòng huấn luyện lọc batch theo loss và ngưỡng λ ([`training/loop.py`](../src/dddcl/training/loop.py)).

---

## 3. Nhánh density, tầng ngoài: xếp bin theo số mẫu giảm dần

**Làm gì.**

```python
bin_counts  = Counter(bin_assignments)
sorted_bins = sorted(bin_counts.keys(), key=lambda b: (-bin_counts[b], b))
```

Khoá sắp xếp là $(-\,n_b,\ b)$ với $n_b$ là số mẫu của bin $b$: bin đông xếp trước, hai bin đông bằng nhau thì bin có chỉ số nhỏ hơn xếp trước.

**Ví dụ đã chạy** (10 mẫu, 2 lớp; `bin_assignments` là chỉ số cục bộ trong từng lớp):

```
id       s0   s1   s2   s3   s4   s5   s6   s7   s8   s9
lớp       0    0    0    0    0    0    1    1    1    1
khoảng  .10  .20  .25  .30  .80  .90  .15  .20  .70  .75
bin       0    0    0    0    1    1    0    0    1    1

bin_counts  = {0: 6, 1: 4}     → sorted_bins = [0, 1]
```

Khi hai bin bằng nhau về số mẫu, ví dụ 3 và 3 với bin `[1,1,1,0,0,0]`, thứ tự ra là `s3,s4,s5` (bin 0) trước `s0,s1,s2` (bin 1), tức chỉ số bin nhỏ hơn đứng trước. Đã kiểm bằng cách chạy.

**Vì sao.** Paper coi nhóm đông mẫu là vùng "dày" của phân bố và do đó dễ học. Khoá `(-count, b)` là cách xác định rõ ràng nhất của "đông trước". Tie-break theo chỉ số bin làm kết quả xác định, không phụ thuộc thứ tự băm của `Counter`.

**So với paper.** Paper, mục 4.1.2 (trang 6), mô tả Self-Paced DDCL-Density gồm hai bước: bước đầu "chấm điểm các quantile theo số mẫu (gọi là density); quantile có số mẫu cao hơn được chấm là *easy*, ít hơn là *hard*; sau đó dữ liệu huấn luyện được sắp từ density cao xuống thấp". Nhánh code này làm đúng điều đó → **[KHỚP]**. Paper không nói cách phá hòa khi hai nhóm bằng nhau; tie-break theo chỉ số bin là chọn của nhóm → **[DIỄN GIẢI]**.

**Giới hạn quan trọng: đối tượng được đếm là "bin cục bộ gộp giữa các lớp".** `bin_assignments` là chỉ số cục bộ trong từng lớp (doc 14, §2), nhưng `Counter` ở đây gộp bin cùng chỉ số của mọi lớp thành một. Trong ví dụ trên, "bin 0" gồm 4 mẫu của lớp 0 và 2 mẫu của lớp 1, dù biên khoảng cách của hai bin này khác nhau. Đây là quy tắc `pool_by_local_bin_index_v1` (A05), mức **[DIỄN GIẢI]**; paper không mô tả cách hợp nhất nhóm giữa các lớp. Spec [03_METHOD_SPEC.md](03_METHOD_SPEC.md) mục 7 cũng ghi nhận đây là điểm mơ hồ và gợi ý so với phương án `class_specific` (đếm theo cặp `(lớp, bin)`), nhưng phương án này chưa có trong code.

---

## 4. Nhánh density, tầng trong: xếp mẫu trong bin theo khoảng cách

**Làm gì.**

```python
bin_to_samples = {b: [] for b in sorted_bins}                  # khoá theo thứ tự đã sắp
for sid, b, d in zip(sample_ids, bin_assignments, normalized_distances):
    bin_to_samples[b].append((sid, float(d)))

for b in sorted_bins:
    samples_in_bin.sort(key=lambda s: (s[1], str(s[0])))       # khoảng cách ↑, rồi ID
    for sid, _ in samples_in_bin:
        ordered_ids.append(sid)
        scores[sid] = float(bin_counts[b])
```

Kết quả của ví dụ trên (đã chạy):

```
ddcl_density : s0 s6 s1 s7 s2 s3 | s8 s9 s4 s5
                  bin 0 (6 mẫu)    bin 1 (4 mẫu)
```

`s1` và `s7` cùng khoảng cách 0.20, nên `s1` đứng trước `s7` nhờ tie-break theo ID.

**Vì sao.**
- Bước gom chỉ thêm mẫu theo thứ tự xuất hiện trong `sample_ids`, tức thứ tự tùy ý của dữ liệu. Không sort thì thứ tự trong bin phụ thuộc cách dữ liệu được nạp.
- Chọn khoảng cách tăng dần giữ nhất quán với Point: gần tâm là dễ. Như vậy Density là "Point có nhóm", với nhóm được xếp theo số mẫu.
- Khoá `(distance, str(id))` làm kết quả xác định với mọi `seed`.

**So với paper.** Paper chỉ nói xếp theo density; **không nói thứ tự mẫu trong cùng quantile** → **[DIỄN GIẢI]**. Điểm này đáng chú ý vì nó quyết định thứ tự thực tế bên trong mỗi bin, và do đó ảnh hưởng tới cách chia batch (batch liên tiếp, không shuffle). Ghi nhận trong doc 14 §10 là "lựa chọn thêm của nhóm".

**Hệ quả khi so với Point.** Hai kiểu chỉ khác nhau ở tầng ngoài. Trong ví dụ trên, `ddcl_density` và `ddcl_point` ra cùng thứ tự `s0 s6 s1 s7 s2 s3 s8 s9 s4 s5` (đã chạy), vì bin đông hơn cũng đúng là bin gần tâm hơn. Hai kiểu chỉ khác nhau khi bin xa tâm lại đông hơn bin gần tâm, điều thường xảy ra với KDE một đỉnh khi bin lõi nằm ở giữa dải chỉ số (doc 14 §6).

---

## 5. Điểm số `scores`

**Làm gì.** Mỗi mẫu có một số thực; ý nghĩa phụ thuộc strategy:

| Strategy | `scores[sid]` | Số lớn nghĩa là |
|---|---|---|
| density | số mẫu của bin (`float(bin_counts[b])`) | dễ hơn |
| point | khoảng cách chuẩn hóa | khó hơn |
| random | vị trí trong hoán vị | không có nghĩa về độ khó |

**Nhận xét.**
- **Cực tính ngược nhau.** Density: điểm cao = dễ. Point: điểm cao = khó. Ai đọc `scores` mà không biết strategy sẽ hiểu sai.
- **Điểm density không phân biệt mẫu trong cùng bin.** Mọi mẫu của một bin có cùng điểm; thứ tự thực tế nằm ở `ordered_train_ids`, không phục hồi được từ `scores`.
- **Không có code nào đọc `scores`.** Đã tìm trong toàn bộ `*.py` (src, tools, tests): chỉ `ordered_train_ids` được dùng ([loop.py:108](../src/dddcl/training/loop.py#L108), notebook 02). `scores` chỉ xuất hiện trong một assertion `len(res.scores) == len(X_train)` ở [test_curriculum.py:253](../tests/test_curriculum.py#L253). Cũng vậy với `distances`, `group_ids`.
- **`current_rank` là biến chết.** Được khởi tạo và tăng trong vòng lặp ([scoring.py:105, 113](../src/dddcl/curriculum/scoring.py#L105)) nhưng không đọc ở đâu; có thể xóa mà không đổi kết quả.

**So với paper.** Paper dùng động từ "score" cho mô tả khái niệm (quantile/khoảng cách được "chấm điểm" dễ/khó) và không định nghĩa cấu trúc dữ liệu điểm số. Trường `scores` là thiết kế của nhóm, không có tương ứng trong paper. Vì chưa có người dùng, đây là dữ liệu ghi vết chứ không phải một phần của thuật toán.

---

## 6. `policy_hash`

**Làm gì.**

```python
policy_hash = compute_policy_hash(strategy, {"seed": seed, "n_samples": n_samples})
# SHA-256 của strategy + các cặp "k:v" theo thứ tự khóa, lấy 16 ký tự hex đầu
```

**Quan sát đã kiểm chứng bằng cách chạy.**
- Cùng dữ liệu, `ddcl_density` với `seed=42` và `seed=1` cho hai hash khác nhau (`00906d9c…` và `a94d53a4…`), dù thứ tự ra giống hệt: nhánh density không dùng `seed`. Hash đổi mà thứ tự không đổi.
- Hash **không** gồm `n_levels`, `smote_policy`, hay nội dung dữ liệu. Hai cấu hình khác nhau ở các tham số đó vẫn cùng hash nếu cùng `strategy`, `seed`, `n_samples`.
- Hash **không được dùng** để kiểm tra hay so sánh ở bất kỳ đâu trong `src/`. Nó chỉ được tính và lưu vào kết quả; ở `ensemble.py:401` nó bị gán thẳng là chuỗi `"controlled"`. Việc nhận diện lần chạy nằm ở config fingerprint trong `status.json` (xem [CLAUDE.md](CLAUDE.md)), không phải ở đây.

**Vì sao nhóm thêm.** Ý định hợp lý là có một dấu vân tay chính sách để đối chiếu giữa các lần chạy. Nhưng hiện chưa đủ thông tin để làm việc đó và cũng chưa ai đối chiếu.

**So với paper.** Không có trong paper.

---

## 7. Đóng gói `CurriculumResult`

**Làm gì.** Dataclass `frozen=True` với bảy trường:

| Trường | Kiểu | Vai trò thực tế |
|---|---|---|
| `strategy` | `str` | ghi vết |
| `ordered_train_ids` | `List[str]` | **thứ tự huấn luyện, duy nhất được dùng** |
| `distances` | `Dict[str, float]` | ghi vết |
| `group_ids` | `Dict[str, int]` | ghi vết (là bin **cục bộ**, không phải nhóm đã gộp) |
| `scores` | `Dict[str, float]` | ghi vết (cực tính tùy strategy) |
| `resampling_report` | `Dict` | báo cáo SMOTE, chỉ truyền qua |
| `policy_hash` | `str` | ghi vết |

**Nhận xét.**
- **`frozen` chỉ chặn gán lại thuộc tính.** Đã chạy: `r.strategy = "x"` ném `FrozenInstanceError`, nhưng `r.scores["s0"] = -1.0` vẫn thành công vì từ điển bên trong có thể sửa. Với `ordered_train_ids` (một `list`) cũng vậy. Nếu muốn bất biến thật sự cần dùng `tuple` / `MappingProxyType`; hiện tại "bất biến" chỉ là quy ước.
- **Tên `group_ids` dễ gây hiểu nhầm.** Giá trị là chỉ số bin cục bộ trong lớp, nên hai mẫu cùng `group_id` chưa chắc cùng vùng khoảng cách nếu khác lớp (doc 14 §10).
- Sau SMOTE, `sample_ids` đầu vào đã gồm cả ID `synth_*`, nên `bin_counts` và `ordered_train_ids` có cả mẫu tổng hợp. Mặc định SMOTE tắt nên điều này chưa xảy ra trong kết quả hiện có.

**So với paper.** Paper kết thúc Algorithm 1 bằng `Result: S_cur ← dataScoring(st, q)`: một **tập dữ liệu huấn luyện đã được sắp lại**. Code trả về một **danh sách ID đã sắp**, và `create_curriculum_batches` ([loop.py:107-127](../src/dddcl/training/loop.py#L107-L127)) mới dựng dữ liệu theo thứ tự đó. Hai cách tương đương về thứ tự, khác nhau về biểu diễn → không ảnh hưởng thuật toán.

Một chi tiết bố cục: trong pseudocode của Algorithm 1, dòng `Result: S_cur ← dataScoring(st, q)` và `return S_cur` được in thụt vào bên trong vòng `foreach s ∈ S` (duyệt từng lớp). Đọc theo mặt chữ có thể hiểu việc chấm điểm xảy ra trong từng lớp; nhóm hiểu là chấm điểm toàn cục sau khi có bin của mọi lớp (D06/A05). Paper không giải thích thêm, nên đây là điểm cần xác nhận với mã tác giả.

---

## 8. Đối chiếu với paper (tổng hợp)

| Nội dung | Paper | Code | Mức độ |
|---|---|---|---|
| Density: nhóm đông mẫu là dễ, xếp nhiều → ít | Mục 4.1.2, trang 6 | `sorted(..., key=(-count, bin))` | [KHỚP] |
| Density: tie-break khi hai nhóm bằng nhau | Không nêu | Chỉ số bin nhỏ trước | [DIỄN GIẢI] |
| Density: thứ tự mẫu trong nhóm | Không nêu | Khoảng cách ↑, rồi ID | [DIỄN GIẢI] |
| Density: đối tượng được đếm là nhóm gộp giữa các lớp theo chỉ số cục bộ | Không nêu | `Counter(bin_assignments)` (A05) | [DIỄN GIẢI] |
| Point: khoảng cách ngắn là dễ, xếp dễ → khó | Mục 4.1.2, trang 7 ("normalised Euclidean distance … shorter = easy") | Sort toàn bộ theo khoảng cách chuẩn hóa ↑ | [KHỚP] |
| Point: tie-break | Không nêu | Theo `str(sample_id)` | [DIỄN GIẢI] |
| Self-Paced: giữ thứ tự gốc của dữ liệu | Mục 5.1, trang 8: "without applying any pre-processing to alter the initial order" | Hoán vị ngẫu nhiên cố định theo `seed` (giống No Curriculum) | [DIỄN GIẢI] (paper không nói thứ tự gốc là ngẫu nhiên hay thứ tự file) |
| Đầu ra: tập dữ liệu đã sắp (`S_cur`) | Algorithm 1 | Danh sách ID đã sắp | Tương đương biểu diễn |
| `scores`, `group_ids`, `distances`, `policy_hash`, `resampling_report` | Không có | Trường ghi vết, không dùng | Bổ sung của nhóm |

**Kết luận về độ tin cậy.** Hai phần cốt lõi của thứ tự (Density theo số mẫu giảm dần, Point theo khoảng cách tăng dần) khớp mô tả của paper. Phần có thể làm kết quả khác tác giả nằm ở ba chỗ paper không nói: (1) cách hợp nhất bin giữa các lớp, (2) thứ tự mẫu trong bin, (3) thứ tự gốc của Self-Paced.

---

## 9. Hạn chế và việc nên làm

1. **Gộp bin cục bộ giữa các lớp (§3).** Ảnh hưởng tới mọi kết quả `ddcl_density` và `sp_ddcl_density`, kể cả khi SMOTE tắt. Nên có một ablation `class_specific` (nhóm theo cặp `(lớp, bin)`) để biết mức độ ảnh hưởng; hiện chưa có.
2. **Cực tính `scores` không nhất quán (§5).** Nếu sau này `scores` được dùng (ví dụ làm báo cáo hay SPL dùng điểm), cần thống nhất "điểm cao = dễ" hoặc "điểm cao = khó" cho mọi strategy.
3. **`policy_hash` yếu và chưa dùng (§6).** Hoặc bổ sung `n_levels`, `smote_policy` và hash của `sample_ids`/khoảng cách rồi thực sự đối chiếu, hoặc bỏ để khỏi tạo cảm giác có kiểm soát.
4. **`current_rank` thừa (§5).** Xóa.
5. **Bất biến chỉ là quy ước (§7).** Cân nhắc `tuple` hoặc `MappingProxyType` nếu cần bảo đảm thật.
6. **Test còn mỏng cho nhánh density.** [`test_density_scoring_oracle`](../tests/test_curriculum.py#L71) kiểm thứ tự giữa các bin bằng so sánh **tập hợp**, không kiểm thứ tự mẫu trong bin, không có ca hòa số mẫu giữa hai bin, và không có ca nhiều lớp (nơi quy tắc gộp bin có tác dụng). Test `ddcl_point` thì có ca hòa khoảng cách.
7. **Cần xác nhận với mã tác giả**: thứ tự trong quantile, cách hợp nhất giữa các lớp, thứ tự gốc của Self-Paced, và vị trí thật của `dataScoring` so với vòng lặp lớp.
