# `partition_by_class`: quá trình chia nhóm (bin) theo mật độ KDE

Tài liệu giải thích hàm `partition_by_class` và hàm lõi `kde_level_partition_v1` trong [`src/dddcl/curriculum/partition.py`](../src/dddcl/curriculum/partition.py). Với mỗi bước có ba phần: **làm gì**, **vì sao chọn cách này**, **so với paper**.

Paper: Chaudhry & Sharma, *Dynamic Data Distribution-based Curriculum Learning*, Information Sciences 702 (2025) (gọi tắt là Dynamic DDCL), và bài tiền nhiệm Static DDCL (2024).

Ký hiệu mức độ so với paper (dùng chung với [13_CODE_VS_PAPER.md](13_CODE_VS_PAPER.md)):

- **[KHỚP]**: đúng với những gì paper nêu.
- **[DIỄN GIẢI]**: paper không nêu rõ, nhóm chọn một cách hiểu (có ghi trong `ASSUMPTIONS.md` / `DECISIONS.md`).
- **[LỆCH]**: khác paper.
- **[CHƯA LÀM]**: paper có, code không có.

> **Lưu ý về nguồn.** Phần "so với paper" dựa trên các tài liệu đối chiếu đã có trong repo (`DECISIONS.md` D05, `ASSUMPTIONS.md` A03–A05, `docs/03_METHOD_SPEC.md`, `docs/13_CODE_VS_PAPER.md`). Tài liệu này không đọc lại PDF paper và không chạy lại code. Điểm quan trọng nhất cần nhớ: **paper không định nghĩa công thức chia quantile**, nên gần như toàn bộ cách chia ở đây là giả định của nhóm (A04), không phải phương trình của paper.

---

## 0. Tổng quan

```
raw_train_X, y
   │  geometry.compute_geometry          (tâm lớp, khoảng cách, chuẩn hóa L2 theo lớp)
   ▼
normalized_distances  (n_samples,)
   │  partition_by_class                 ◄── tài liệu này
   │     └─ với mỗi lớp c: kde_level_partition_v1(khoảng cách của lớp c)
   ▼
PartitionResult
   ├─ boundaries[c]        : [b0, b1, ..., bM]  biên bin của lớp c
   ├─ bin_assignments      : chỉ số bin CỤC BỘ (0..M-1) của từng mẫu
   ├─ n_bins_per_class[c]  : số bin của lớp c
   └─ fallback_classes     : các lớp bị đưa về 1 bin
   │
   ▼  (tùy chọn) resample.py (SMOTE, mặc định tắt) → scoring.py (xếp thứ tự DDCL-Density)
```

Trong pipeline thật, hàm được gọi ở [`runner.py:123`](../src/dddcl/runner.py#L123) với `n_levels=5` (hard-code; mặc định của hàm là 6).

| Bước | Nội dung | Mức độ so với paper |
|---|---|---|
| 1 | Nhận khoảng cách đã chuẩn hóa | [DIỄN GIẢI] (A03) |
| 2 | Tách theo lớp, chia độc lập từng lớp | [KHỚP] về khung, [DIỄN GIẢI] về chi tiết |
| 3 | Các trường hợp suy biến (fallback) | Không có trong paper (bổ sung của nhóm) |
| 4 | Ước lượng KDE 1D (Gaussian, Scott) | [KHỚP] về ý tưởng, [DIỄN GIẢI] về kernel/bandwidth |
| 5 | Lưới đánh giá mật độ | [DIỄN GIẢI] |
| 6 | Cắt mức mật độ ngang, tìm giao điểm | [DIỄN GIẢI] (A04), phần cốt lõi |
| 7 | Sắp xếp và gộp biên gần nhau | [DIỄN GIẢI] |
| 8 | Gán mẫu vào bin | [DIỄN GIẢI] |
| 9 | Gói kết quả `PartitionResult` | Không có trong paper |
| 10 | Bin cục bộ được gộp giữa các lớp ở bước sau | [DIỄN GIẢI] (A05) |

---

## 1. Đầu vào: khoảng cách chuẩn hóa

**Làm gì.** `partition_by_class` nhận `normalized_distances` (một số cho mỗi mẫu train) và nhãn `y`. Giá trị này do [`geometry.py`](../src/dddcl/curriculum/geometry.py) tính:

Gọi $S_c = \{\, i : y_i = c \,\}$ là tập chỉ số các mẫu train thuộc lớp $c$, và $n_c$ là số phần tử của $S_c$.

1. Tâm lớp là trung bình cộng:

$$\mu_c = \frac{1}{n_c} \sum_{i \in S_c} x_i$$

2. Khoảng cách Euclid từ mẫu $i$ (thuộc lớp $c$) tới tâm lớp của chính nó, tính trên **raw features** (chưa qua StandardScaler):

$$r_i = \lVert x_i - \mu_c \rVert_2 = \sqrt{\sum_{d=1}^{D} \left( x_{i,d} - \mu_{c,d} \right)^2}$$

3. Chuẩn hóa theo lớp, chia cho chuẩn $L_2$ của vector khoảng cách của lớp đó:

$$\hat{r}_i = \frac{r_i}{\sqrt{\sum_{j \in S_c} r_j^{2}}}$$

Nếu mẫu số bằng 0 (mọi mẫu của lớp trùng tâm, hoặc lớp chỉ có 1 mẫu) thì đặt $\hat{r}_i = 0$.

**Vì sao.**
- Tâm là trung bình vì đó là nghiệm tối thiểu hóa tổng bình phương khoảng cách (A03); công thức trong paper bị viết sai ký hiệu `min`.
- Dùng raw features vì đó là cách đọc sát thứ tự mô tả của paper (A10); không gian đã chuẩn hóa là ablation riêng.
- Chuẩn hóa theo lớp để đưa khoảng cách của các lớp về một thang chung. Các hằng số tuyệt đối trong `partition.py` ($\tau = 10^{-4}$, $10^{-7}$, $10^{-12}$, $10^{-5}$) vì vậy có ý nghĩa tương đương giữa các lớp và các tập dữ liệu.

**Lưu ý.** Mẫu số có thể viết lại theo căn quân phương (RMS) của khoảng cách:

$$\sqrt{\sum_{j \in S_c} r_j^{2}} = \sqrt{n_c}\cdot \mathrm{RMS}(r), \qquad \mathrm{RMS}(r) = \sqrt{\frac{1}{n_c}\sum_{j \in S_c} r_j^{2}}$$

nên giá trị chuẩn hóa điển hình cỡ $\hat{r} \approx 1/\sqrt{n_c}$. Lớp nhiều mẫu có khoảng cách chuẩn hóa nhỏ hơn lớp ít mẫu, nên ngưỡng gộp $\tau = 10^{-4}$ tương đối "thô" hơn ở lớp lớn. Đây là hệ quả của phép chuẩn hóa, chưa thấy ảnh hưởng nào được kiểm tra riêng.

**So với paper.** Paper nhắc đến khoảng cách "normalised" ở phần Point nhưng **không chỉ rõ phép chuẩn hóa** (`03_METHOD_SPEC.md` mục 4). Công thức chia cho chuẩn L2 là đề xuất độc lập của nhóm, không được gọi là phương trình paper. → **[DIỄN GIẢI]**.

---

## 2. Tách theo lớp và chia độc lập từng lớp

**Làm gì.**

```python
for c in np.unique(y):
    indices = np.where(y == c)[0]
    bounds, local_bins, fallback = kde_level_partition_v1(normalized_distances[indices], n_levels)
    boundaries[c]      = bounds
    bin_assignments[indices] = local_bins        # ghi lại đúng vị trí mẫu gốc
    n_bins_per_class[c] = len(bounds) - 1
    if fallback: fallback_classes.append(c)
```

Mỗi lớp có KDE riêng, biên bin riêng, và số bin riêng. Kết quả `bin_assignments` là chỉ số bin **cục bộ trong lớp** (0 là bin gần tâm nhất).

**Vì sao.**
- Khoảng cách được định nghĩa tới tâm **của chính lớp đó**, nên phân bố khoảng cách của các lớp khác nhau về hình dạng và thang đo. Một KDE chung sẽ trộn các phân bố không cùng bản chất.
- Mỗi lớp có thể nhiều hoặc ít mẫu, có phân tán rộng hoặc hẹp. Chia riêng giúp mỗi lớp có một "bản đồ độ khó" nội tại của nó.
- Ghi theo `indices` giữ nguyên thứ tự mẫu gốc, các bước sau (SMOTE, scoring) làm việc theo vị trí/ID mà không cần ánh xạ lại.
- Chuyển `int(c)` làm khóa để từ điển kết quả tuần tự hóa được ra JSON.

**So với paper.** Algorithm 1 của paper tính tâm theo lớp và KDE `f̂` trên `d_c`, tức phân bố khoảng cách của từng lớp, nên cách chia theo lớp là **[KHỚP]** về khung. Paper **không nói rõ** quy tắc hợp nhất các nhóm giữa các lớp, nhóm xử lý ở bước 10 (A05) → **[DIỄN GIẢI]**.

---

## 3. Các trường hợp suy biến (fallback)

**Làm gì.** `kde_level_partition_v1` trả `used_fallback = True` với biên $[\,\hat{r}_{\min} - 10^{-5},\ \hat{r}_{\max} + 10^{-5}\,]$ (một bin duy nhất, mọi mẫu vào bin 0) khi:

| Điều kiện | Lý do kỹ thuật |
|---|---|
| $n = 0$ | Lớp rỗng: trả biên $[0, 1]$, mảng bin rỗng |
| $\hat{r}_{\max} - \hat{r}_{\min} < 10^{-7}$ hoặc $n < 3$ | Khoảng cách gần như hằng số, hoặc quá ít điểm để ước lượng KDE |
| `gaussian_kde(...)` ném ngoại lệ | Ma trận hiệp phương sai suy biến (ví dụ mọi điểm trùng nhau) |
| $f_{\max} < 10^{-12}$ | Mật độ cực đại quá nhỏ, mức cắt $t_k = f_{\max}\,k/L$ mất ý nghĩa |
| Sau khi gộp biên còn dưới 2 biên ($M < 1$) | Còn dưới 2 biên: không tạo được bin |

Khi fallback, biên là $[\,\hat{r}_{\min} - 10^{-5},\ \hat{r}_{\max} + 10^{-5}\,]$. Độ lệch $10^{-5}$ để biên bao trọn mọi mẫu (mẫu nằm đúng tại min/max vẫn nằm trong khoảng).

**Vì sao.** Các lớp nhỏ hoặc tập dữ liệu nhỏ (New-Thyroid chỉ khoảng 150 mẫu train, có lớp rất ít mẫu) rất dễ gây suy biến. Thay vì để chương trình lỗi hoặc sinh bin vô nghĩa, nhóm chọn quay về một bin duy nhất và **ghi lại** lớp nào bị fallback (`fallback_classes`), để kết quả luôn dùng được và vẫn kiểm tra được sau này.

**So với paper.** Paper không bàn tới các trường hợp này. Đây là phần kỹ thuật bổ sung của nhóm, không đổi ý tưởng thuật toán. `03_METHOD_SPEC.md` mục 5 yêu cầu có fallback "một nhóm và ghi event".

---

## 4. Ước lượng mật độ KDE một chiều

**Làm gì.**

```python
kde = gaussian_kde(distances, bw_method="scott")
```

Với tập khoảng cách chuẩn hóa $\{\hat{r}_1, \dots, \hat{r}_n\}$ của một lớp, hàm mật độ ước lượng là

$$\hat{f}(x) = \frac{1}{n\,h} \sum_{i=1}^{n} K\!\left( \frac{x - \hat{r}_i}{h} \right), \qquad K(u) = \frac{1}{\sqrt{2\pi}}\, e^{-u^{2}/2}$$

và băng thông theo quy tắc Scott (một chiều):

$$h = \sigma \cdot n^{-1/5}$$

với $\sigma$ là độ lệch chuẩn mẫu của $\{\hat{r}_i\}$. (SciPy tính Scott bằng hệ số $n^{-1/(d+4)}$ với $d = 1$, nhân với độ lệch chuẩn của dữ liệu.)

**Vì sao.**
- Cần một đường cong mật độ **trơn** của khoảng cách để định nghĩa "vùng đông mẫu" (dễ) và "vùng thưa" (khó). Histogram thì phụ thuộc cách đặt biên bin, đúng thứ ta đang cần tìm.
- Gaussian + Scott là mặc định chuẩn, không cần siêu tham số thêm, tránh việc phải dò tham số trên dữ liệu test (CLAUDE.md cấm dùng test để chọn). `scipy.stats.gaussian_kde` đã có sẵn và xác định được (deterministic).
- Chỉ 1 chiều (khoảng cách) nên KDE ổn định ngay với ít mẫu, không bị lời nguyền chiều cao.

**So với paper.** Paper dùng KDE dạng $\hat{f}(x) = \dfrac{1}{\lvert s\rvert\, h} \sum_{i} K\!\left(\dfrac{x - x_i}{h}\right)$ trên khoảng cách theo lớp (đây chính là công thức ở trên, với $\lvert s\rvert = n$). Việc dùng KDE là **[KHỚP]**. Loại kernel, băng thông và cách chọn `h` không được nêu rõ trong tài liệu đối chiếu → chọn Gauss + Scott là **[DIỄN GIẢI]**. `ASSUMPTIONS.md` A04 ghi `curriculum.bandwidth=scott`.

---

## 5. Lưới đánh giá mật độ

**Làm gì.**

```python
pad = (max_val - min_val) * 0.05
x_grid  = np.linspace(max(0.0, min_val - pad), max_val + pad, grid_points)   # 1000 điểm
density = kde.evaluate(x_grid)
peak_density = density.max()
```

**Vì sao.**
- Lưới đều 1000 điểm là cách rẻ nhất để biến hàm liên tục thành mảng, và là điều kiện để dò giao điểm bằng đổi dấu ở bước 6.
- Đệm 5% hai đầu để đường cong không bị "cắt cụt" ngay tại min/max, nếu không thì mật độ ở rìa bị đánh giá thiếu và có thể mất giao điểm sát biên.
- `max(0.0, ...)`: khoảng cách không âm nên không cần phần lưới âm.
- `peak_density` làm mốc chiều cao để chia mức cắt theo tỷ lệ, không phụ thuộc thang đo tuyệt đối của mật độ.

**So với paper.** Paper không nói tới lưới. `03_METHOD_SPEC.md` yêu cầu "kernel/bandwidth/grid explicit trong config" để tái lập → **[DIỄN GIẢI]**. Hiện `grid_points=1000` là tham số mặc định của hàm, không được runner đọc từ YAML.

---

## 6. Cắt mức mật độ ngang và tìm giao điểm (cốt lõi)

**Làm gì.**

```python
crossings = [min_val, max_val]
for k in range(1, n_levels):
    target = peak_density * (k / n_levels)        # mức cắt thứ k
    diff   = density - target
    idx    = np.where(np.diff(np.sign(diff)))[0]  # chỗ đổi dấu
    # nội suy tuyến tính trong đoạn [x_grid[idx], x_grid[idx+1]]:
    x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
    if min_val <= x_cross <= max_val: crossings.append(x_cross)
```

Ký hiệu: $L$ là số mức (`n_levels`), $f_{\max} = \max_x \hat{f}(x)$ là mật độ đỉnh trên lưới, $x_1 < x_2 < \dots < x_G$ là các điểm lưới ($G = 1000$).

**Mức cắt thứ $k$:**

$$t_k = f_{\max} \cdot \frac{k}{L}, \qquad k = 1, 2, \dots, L-1$$

Với $L = 5$: $t_k = 0.2 f_{\max},\ 0.4 f_{\max},\ 0.6 f_{\max},\ 0.8 f_{\max}$.

**Phát hiện giao điểm.** Đặt $g_k(x) = \hat{f}(x) - t_k$. Đoạn lưới $[x_m, x_{m+1}]$ chứa một giao điểm khi $g_k$ đổi dấu:

$$g_k(x_m)\cdot g_k(x_{m+1}) < 0 \quad\Longleftrightarrow\quad \mathrm{sign}\, g_k(x_m) \neq \mathrm{sign}\, g_k(x_{m+1})$$

**Nội suy tuyến tính** trên đoạn đó, với $y_0 = g_k(x_m)$ và $y_1 = g_k(x_{m+1})$, là nghiệm của $g_k = 0$ trên đường thẳng nối hai điểm:

$$x^{\ast} = x_m - y_0 \cdot \frac{x_{m+1} - x_m}{y_1 - y_0}$$

(chỉ tính khi $\lvert y_1 - y_0 \rvert > 10^{-12}$ để tránh chia cho 0).

**Điều kiện giữ lại:** $\hat{r}_{\min} \le x^{\ast} \le \hat{r}_{\max}$, với $\hat{r}_{\min},\hat{r}_{\max}$ là khoảng cách chuẩn hóa nhỏ nhất và lớn nhất thực tế của lớp.

**Tập biên thô:**

$$\mathcal{B} = \{\hat{r}_{\min},\ \hat{r}_{\max}\} \ \cup\ \{\, x^{\ast} \text{ thu được ở mọi mức } k \,\}$$

Trên toàn lưới (kể cả vùng đệm 5%), mỗi mức cắt đường cong KDE tại ít nhất 2 điểm (hai sườn của mỗi đỉnh). Tuy nhiên sau bước lọc chỉ giữ giao điểm nằm trong $[\hat r_{\min}, \hat r_{\max}]$, nên với phân bố lệch, một sườn có thể mất giao điểm của các mức thấp: hai sườn **không nhất thiết có số biên bằng nhau** (xem mục 11, điểm 1).

```
mật độ
  ▲        ___
peak ─ ─ ─/   \─ ─ ─         ← ngoài cùng: không cắt
 4/5 ─ ─ /─────\─ ─          × ×  (2 giao điểm)
 3/5 ─ ─/───────\─ ─         × ×
 2/5 ─ /─────────\─          × ×
 1/5 ─/───────────\─         × ×
     /             \___
    ├─┬──┬───┬──────┬───┬─┬─► khoảng cách chuẩn hóa
   min                       max       → bin hẹp ở nơi đường cong dốc, rộng ở đuôi
```

**Vì sao.**
- **Bin phản ánh hình dạng mật độ.** Vùng dốc (mật độ đổi nhanh) có nhiều giao điểm nên bin hẹp; vùng đuôi phẳng, thưa nên bin rộng. Số mẫu mỗi bin vì thế khác nhau nhiều, đúng như "nhóm đông mẫu" và "nhóm ít mẫu" mà DDCL-Density cần phân biệt.
- **Mức cắt theo tỷ lệ với đỉnh** ($t_k = f_{\max}\, k/L$) nên bất biến với thang đo của mật độ.
- **Dò bằng đổi dấu** không giả định số đỉnh: KDE nhiều đỉnh vẫn cho đủ giao điểm (xem mục "Hạn chế").
- **Nội suy tuyến tính** cho biên mịn hơn lấy điểm lưới gần nhất, với chi phí tính toán không đáng kể.
- **Lọc $\hat{r}_{\min} \le x^{\ast} \le \hat{r}_{\max}$** loại giao điểm rơi vào vùng đệm 5% không có dữ liệu.
- **Không dùng `qcut`** (equal-frequency): bằng chứng từ Figure 2a (xem bên dưới) cho thấy các nhóm của paper có số mẫu rất khác nhau.

**So với paper.** Đây là phần **khác biệt quan trọng nhất**:

| | Paper | Code |
|---|---|---|
| Hàm chia | `getQuantile(f̂)`, **không có công thức** | `kde_level_partition_v1` do nhóm tự thiết kế |
| Tính chất đã biết | Figure 2a (Static DDCL §III) có các nhóm số mẫu khác hẳn nhau (Q6 có 82 mẫu, Q1 có 0 mẫu) | Bin có kích thước thay đổi theo mật độ; không bảo đảm số bin bằng số mức |
| Tên gọi | "quantile" | Cắt mức mật độ ngang, không phải quantile theo nghĩa thống kê |

→ **[DIỄN GIẢI]** (D05/A04, trạng thái FROZEN). Lập luận của nhóm: vì Figure 2a có nhóm trống và nhóm lệch kích thước, tác giả **không** dùng quantile chia đều số mẫu, nên mức cắt mật độ "phản ánh đúng tinh thần Data Distribution-based". Đây là **suy luận từ hình**, không phải xác nhận từ mã nguồn tác giả; không thể khẳng định kết quả giống hệt. Chưa có sensitivity so với equal-frequency (`03_METHOD_SPEC.md` cho phép thử như một nhánh riêng).

---

## 7. Sắp xếp và gộp các biên quá gần nhau

**Làm gì.**

```python
crossings = sorted(crossings)
merged_bounds = [crossings[0]]
for val in crossings[1:]:
    if val - merged_bounds[-1] > tolerance:      # tolerance = 1e-4
        merged_bounds.append(val)
```

Về mặt toán học: sắp tăng dần tập $\mathcal{B}$ thành $c_0 \le c_1 \le \dots \le c_{P}$, rồi dựng dãy biên $b_0 < b_1 < \dots < b_M$ bằng quy tắc

$$b_0 = c_0, \qquad c_p \text{ được giữ lại } \iff c_p - b_{\text{cuối}} > \tau, \quad \tau = 10^{-4}$$

trong đó $b_{\text{cuối}}$ là biên cuối cùng đã được giữ tại thời điểm xét $c_p$. Số bin của lớp là $M$.

**Vì sao.**
- Giao điểm được thêm theo thứ tự mức $k$ và theo sườn trái/phải, nên chưa tăng dần. Biên bin phải đơn điệu để `searchsorted` hoạt động.
- Có thể có giao điểm trùng hoặc rất sát nhau (đường cong dốc đứng cắt nhiều mức cùng chỗ; đổi dấu tại điểm lưới trùng 0; giao điểm sát `min`/`max`). Giữ chúng sẽ sinh bin độ rộng gần 0, rỗng mẫu và dễ gây lỗi số.
- Nếu sau gộp còn dưới 2 biên, quay về 1 bin (fallback).

**Lưu ý.** So sánh với biên cuối **đã giữ**, nên khi hai điểm sát nhau thì giữ điểm nhỏ, bỏ điểm lớn. Hệ quả: nếu có giao điểm cách `max_val` dưới `tolerance`, chính `max_val` bị bỏ và biên cuối nhỏ hơn `max_val` một chút. Bước 8 vẫn gán đúng nhờ `np.clip` và chặn chỉ số ở `n_bins − 1`. Số bin cuối cùng phụ thuộc dữ liệu và `tolerance`.

**So với paper.** Paper không bàn tới bước này; đây là xử lý kỹ thuật của nhóm → **[DIỄN GIẢI]**. `03_METHOD_SPEC.md` mục 6: "sort/dedup" là một phần của thiết kế đề xuất.

---

## 8. Gán mẫu vào bin

**Làm gì.**

```python
v   = np.clip(val, merged_bounds[0], merged_bounds[-1])
idx = np.searchsorted(merged_bounds, v, side="right") - 1
idx = min(max(0, idx), n_bins - 1)
```

Mẫu có khoảng cách chuẩn hóa $\hat{r}$ thuộc bin $j$ ($j = 0, \dots, M-1$) theo quy tắc:

$$\hat{r} \in \begin{cases} [\,b_j,\ b_{j+1}\,) & j = 0, \dots, M-2 \\[2pt] [\,b_{M-1},\ b_M\,] & j = M-1 \quad \text{(bin cuối đóng cả hai đầu)} \end{cases}$$

Trước khi gán, $\hat{r}$ được kẹp vào $[b_0, b_M]$ (`np.clip`) để an toàn số học.

**Vì sao.**
- Nửa khoảng `[b_j, b_{j+1})` bảo đảm mỗi mẫu thuộc **đúng một** bin, không bị đếm hai lần ở biên.
- `side="right"` + `−1` là cách chuẩn để tìm bin chứa `v`; riêng $v = b_M$ cho chỉ số $M$ nên phải chặn về $M-1$ để bin cuối đóng.
- `np.clip` bảo vệ khỏi sai số số học (và trường hợp `max_val` bị gộp mất ở bước 7).
- Độ phủ 100% mẫu train là điều kiện kiểm chứng đã ghi trong D05.

**So với paper.** Paper không nêu quy ước biên → **[DIỄN GIẢI]**.

---

## 9. Kết quả trả về

**Làm gì.** `PartitionResult` là `dataclass(frozen=True)` với: `boundaries`, `bin_assignments`, `n_bins_per_class`, `fallback_classes`.

**Vì sao.**
- Bất biến (`frozen`) để các bước sau không vô tình sửa biên hay nhãn bin.
- `n_bins_per_class` và `fallback_classes` được giữ lại vì số bin **không cố định** và có lớp bị fallback; đây là thông tin cần báo cáo khi so sánh các lần chạy.
- `boundaries` giữ lại để diễn giải, vẽ hình (notebook 02) và kiểm tra tính đơn điệu.

**So với paper.** Không có tương ứng trong paper; chỉ là cấu trúc dữ liệu của nhóm.

---

## 10. Sau `partition_by_class`: bin cục bộ được gộp giữa các lớp

`bin_assignments` là chỉ số **cục bộ** của từng lớp. Ở [`scoring.py`](../src/dddcl/curriculum/scoring.py) (strategy `ddcl_density`, `sp_ddcl_density`):

```python
bin_counts  = Counter(bin_assignments)                        # đếm theo chỉ số bin, GỘP mọi lớp
sorted_bins = sorted(bin_counts, key=lambda b: (-bin_counts[b], b))   # nhiều mẫu trước, hòa thì chỉ số nhỏ trước
# trong mỗi bin: sắp theo khoảng cách tăng dần, hòa thì theo sample_id
```

Tức bin 0 của lớp A và bin 0 của lớp B **cùng một nhóm**. Các bin nhiều mẫu nhất được dạy trước (được coi là "dễ").

**Vì sao.** Paper nói nhóm có nhiều mẫu thì dễ; cách gộp theo chỉ số cục bộ (`pool_by_local_bin_index_v1`) là cách đơn giản nhất để có một danh sách nhóm chung cho cả tập mà không cần định nghĩa thêm tiêu chí so sánh giữa lớp. Thứ tự trong bin theo khoảng cách tăng là lựa chọn thêm của nhóm.

**So với paper.** Việc xếp nhóm từ nhiều mẫu đến ít là **[KHỚP]**. Quy tắc **gộp theo chỉ số cục bộ** thì paper không mô tả → **[DIỄN GIẢI]** (A05). Vì số bin và vị trí biên khác nhau giữa các lớp, "bin 3 của lớp A" và "bin 3 của lớp B" không bảo đảm tương đương về vị trí/khoảng cách. Quy tắc này ảnh hưởng tới **mọi** kết quả DDCL-Density, kể cả khi SMOTE tắt, và nên được ghi như giả định áp dụng chung.

---

## 11. Hạn chế và điểm cần lưu ý

1. **Hai sườn không đối xứng; biên ngoài cùng không nằm trên một mức cắt.** Hai biên ngoài cùng luôn là $\hat r_{\min}$ và $\hat r_{\max}$ (thêm vào ở đầu), không phải giao điểm với mức $t_k$ nào, nên chiều cao KDE tại đó là giá trị tùy ý. Ngoài ra, nếu $\hat f(\hat r_{\min}) > t_k$ (đường cong tại mép trái dữ liệu vẫn cao hơn mức cắt) thì giao điểm trái của mức $t_k$ nằm ngoài khoảng dữ liệu và bị bỏ, trong khi sườn phải (đuôi dài hơn) vẫn có giao điểm. Kết quả: sườn phải có thể nhiều biên hơn sườn trái, và các biên "tương ứng" hai bên không luôn ghép cặp theo cùng tung độ. Với phân bố lệch phải (khoảng cách tới tâm luôn $\ge 0$ nên mép trái bị chặn bởi 0), hiện tượng này gần như chắc chắn xảy ra ở mức thấp nhất $k=1$.
2. **Dữ liệu nhiều đỉnh (multimodal).** Dò đổi dấu cho đủ giao điểm nên không lỗi, nhưng không có xử lý riêng: bin vẫn là khoảng liền trên trục khoảng cách, thung lũng giữa hai đỉnh chỉ thêm biên, và số bin tăng lên. Thí nghiệm E1 (Phase G, tập `synthetic_multimodal`) dành để kiểm tra giới hạn này; hiện còn PENDING.
3. **Số bin không cố định.** Phụ thuộc hình dạng KDE và `tolerance`; mỗi lớp có thể khác nhau, nên bước gộp theo chỉ số cục bộ (mục 10) mang tính áp đặt.
4. **Bin rỗng.** Có thể xảy ra nếu hai biên liên tiếp bao quanh một khoảng không có mẫu. Điều này không sai (Figure 2a của paper cũng có nhóm 0 mẫu) nhưng cần để ý khi xếp thứ tự và SMOTE.
5. **`n_levels` lệch giữa mặc định và pipeline.** Mặc định hàm là 6, `runner.py` và các script `tools/run_*` hard-code 5, và các trường `curriculum.partition_policy`, `curriculum.bandwidth` trong A04 chưa được runner đọc (xem `13_CODE_VS_PAPER.md` §6.4).
6. **Chưa kiểm chứng với mã tác giả.** Cách chia quantile thật của tác giả (có thể là `qcut`, mức mật độ khác, hoặc cách khác) chưa được xác nhận. Việc đọc repo tác giả (https://github.com/ECOLS-research-group/DDCL) là cách kiểm chứng đáng tin nhất.

## 12. Tóm tắt một câu cho từng quyết định

| Quyết định | Lý do ngắn gọn | Quan hệ với paper |
|---|---|---|
| Chia riêng từng lớp | Khoảng cách tính tới tâm riêng của lớp | Khớp khung |
| KDE Gauss + Scott | Mật độ trơn, ít tham số, không dò trên test | Khớp ý tưởng, diễn giải chi tiết |
| Cắt mức mật độ ngang, không `qcut` | Figure 2a có nhóm kích thước lệch, nhóm trống | Diễn giải (A04), suy luận từ hình |
| Nội suy tuyến tính giao điểm | Biên mịn, rẻ | Diễn giải |
| Gộp biên sát nhau (`tolerance`) | Tránh bin rộng bằng 0 | Diễn giải |
| Bin nửa khoảng, bin cuối đóng | Mỗi mẫu đúng một bin | Diễn giải |
| Fallback một bin | Lớp nhỏ/hằng số vẫn chạy được, có ghi vết | Bổ sung của nhóm |
| Gộp bin cục bộ giữa các lớp | Cách đơn giản nhất để có danh sách nhóm chung | Diễn giải (A05) |
