# Ledger ban đầu: evidence và điều chưa biết

Mỗi Agent cập nhật thêm evidence thực, không đổi quan sát chưa chạy thành verified runtime.

| ID | Nguồn | Điều đã xác nhận/quan sát | Hệ quả |
|---|---|---|---|
| P01 | Dynamic §4, §5 | Static được mở rộng bằng SPL và six-NN ensemble | Bài chính Dynamic, Static là dependency |
| P02 | Dynamic §5.1 | Adam .001, lambda .4/.6/growth1.1, 200epoch,30runs,70/20/10 | Giá trị paper, không đủ mọi implementation detail |
| P03 | Dynamic §5.2 | Ensemble600epoch/member,30runs, StandardNN+SVM | Thêm matched6NN control khi critique fairness |
| P04 | Dynamic Table1 / text | Hàng Static dùng lại từ [16] | Không khẳng định cùng splits/seeds cho paired significance paper-vs-team |
| U01 | Hai PDF | Quantile boundaries và normalization không mô tả đầy đủ | Team assumption explicit, independent partition |
| U02 | Dynamic Algorithm2 | Return placement và điều khiển pace thiếu rõ | Loop policy/version công khai, test behavior |
| U03 | Dynamic ensemble | Tie handling không rõ | Train-prior tie assumption, log tie rate |
| O01 | Khảo sát dataset.py tác giả | KDE horizontal-level intersections, target6 có thể sinh hơn6 bins | Observation để audit phase F, không copy code |
| O02 | Khảo sát dataset.py | Point branch sort_index thay vì sort distance trong đoạn ghép | Suspected mismatch, cần runtime trace |
| O03 | Khảo sát curriculum_lib.py | Split val .2 rồi test .1 phần còn lại | Gần72/20/8, khác text70/20/10 |
| O04 | Khảo sát distribution.py | Prepare curriculum trước split trong hàm train | Nguy cơ leakage; cần quantify qua audit |
| O05 | Khảo sát custom_callback.py | Mutate training-data container trong model.fit | Cần xác minh optimizer dùng subset thật hay không |
| D01 | CSV tác giả đã đọc | Diagnostic569x31inputs, ID tồn tại | Track clean drop ID, separate ID ablation |
| D02 | CSV tác giả | Cancer457x9, chưa xác minh nguồn | Block dataset này khi chưa trace nguồn |
| D03 | CSV tác giả | Digits standalone3823, ensemble_test1797 | Ghi đúng official vs random protocol |
| D04 | UCI BUPA | Selector không phải disease diagnosis label | Xác minh target trước interpret |
| D05 | UCI Musk1 | 476conformations/92molecules | Có thể dùng group split stress test riêng |

Observations O01–O05 từ khảo sát commit `c72c34478f3cf523eb43016f19057bd7ba64c2f7`, chưa chạy runtime. Đây không phải chứng minh paper numbers sai. Gói không kèm source tác giả. Khi audit, thêm URL/path/line/command/output hash đúng phiên bản.

## Mẫu quyết định

ID; câu hỏi; paper section; các lựa chọn; lựa chọn nhóm; lý do; nguồn phân loại; config/protocol affected; tests; validation evidence; chưa biết; ngày/commit. Mọi thay đổi sau final freeze cần protocol version mới và rerun affected experiments, không sửa config history.
