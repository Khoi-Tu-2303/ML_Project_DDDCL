# Nguồn và phạm vi xác nhận

## Nguồn chính trong gói

- `references/papers/dynamic-ddcl.pdf`: Dynamic Data Distribution-based Curriculum Learning, Chaudhry & Sharma, Information Sciences702(2025)121924; DOI https://doi.org/10.1016/j.ins.2025.121924 . Bài lab chính.
- `references/papers/static-ddcl-arxiv-v2.pdf`: Data Distribution-based Curriculum Learning, arXiv2402.07352v2, bản nền của Static. Bản journal DOI https://doi.org/10.1109/ACCESS.2024.3465793 . Không dùng metadata DOI placeholder trong preprint làm DOI final.
- `references/course/`: sáu file người dùng cung cấp, giữ nguyên nội dung. Đây là yêu cầu của giảng viên.

## Implementation tác giả — chỉ audit sau bản độc lập

https://github.com/ECOLS-research-group/DDCL ; khảo sát tại commit `c72c34478f3cf523eb43016f19057bd7ba64c2f7` (2025-01-31). Source trong paper trỏ về đây; README xác nhận cả Static/Dynamic và environment TensorFlow2.17/Python3.12.7. Chưa có runtime benchmark trong kit.

## UCI và hosted SDK

- Haberman: https://archive.ics.uci.edu/dataset/43/haberman+s+survival ; 306/3/2, raw nhãn1=sống>=5năm,2=chết<5năm.
- New-Thyroid: https://archive.ics.uci.edu/dataset/102/thyroid+disease ; chỉ file new-thyroid.data215/5/3 trong collection, không dùng tổng7200 của collection.
- Banknote: https://archive.ics.uci.edu/dataset/267/banknote+authentication ;1372/4/2.
- Diagnostic: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic ;30features+ID.
- Musk1: https://archive.ics.uci.edu/dataset/74/musk+version+1 ;166numeric features ngoài name columns.
- BUPA: https://archive.ics.uci.edu/dataset/60/liver+disorders ; cảnh báo selector không là disease label.
- Optical: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits .
- ISOLET: https://archive.ics.uci.edu/dataset/54/isolet .
- HF tokens: https://huggingface.co/docs/hub/security-tokens .
- HF SDK0.26.2 download: https://huggingface.co/docs/huggingface_hub/v0.26.2/guides/download .
- HF upload: https://huggingface.co/docs/huggingface_hub/guides/upload .
- HF dataset cards: https://huggingface.co/docs/hub/datasets-cards .
- PyTorch randomness: https://pytorch.org/docs/stable/notes/randomness.html .

Chuẩn bị kit ngày2026-11-05: đã đọc PDFs, course requirements, source paths ở phiên bản nêu trên; kiểm tra các archive3pilot khi xây công cụ. Không có HF repo thực của người dùng, token, upload HF hoặc W&B run public được tạo bởi kit. Không có kết quả train final. Kiểm tra thực tế mới nhất xem `KIT_VALIDATION.md` và `ENVIRONMENT_STATUS.md`.
