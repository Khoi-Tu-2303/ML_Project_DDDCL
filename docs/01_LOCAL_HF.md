# Local Windows, môi trường và Hugging Face

## 1. Mô hình vận hành

Code và notebook nằm trong Git repo nhóm. Dataset gốc/canonical nằm trên HF Hub. Khi chạy, `hf_hub_download` lấy file theo commit cụ thể về cache máy local; CPU/GPU local huấn luyện. **HF Hub lưu dữ liệu, không tự cung cấp máy huấn luyện.** File public không gated thường đọc được mà không có token. Upload hoặc đọc private/gated cần quyền thích hợp.

Chọn HF làm đường chính; không cần thêm Kaggle token. Nếu tìm được bản Kaggle công khai khớp hơn, Agent có thể hỗ trợ adapter nhưng phải pin phiên bản và kiểm tra nội dung; không thay đổi dataset chỉ vì tên giống.

## 2. Cài môi trường

Mục tiêu tham chiếu: Windows x86_64, Python **3.11.9**, PyTorch, CPU deterministic. Bản tác giả là TensorFlow; backend khác phải được ghi trong deviation analysis. Môi trường TensorFlow tham chiếu nằm riêng ở phase F.

PowerShell, chạy từ checkout nhóm:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --user --name ddcl-local --display-name "DDDCL Local"
.\.venv\Scripts\python.exe -m jupyterlab
```

Khi chạy thí nghiệm, ưu tiên ` .\tools\start_notebook.ps1 ` để đặt PYTHONHASHSEED và thread limits trước khi kernel khởi tạo. Nếu máy chặn script, đặt trong PowerShell hiện tại `$env:PYTHONHASHSEED="0"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"`, rồi chạy Jupyter bằng executable ở trên; không cần đổi policy toàn máy. Seed NumPy/Torch vẫn phải được runner đặt riêng.

`py -3.11` có thể trỏ đến patch khác: kiểm tra phiên bản thực tế, dùng 3.11.9 nếu cần đối chiếu môi trường tham chiếu. Không cần activate PowerShell nếu gọi trực tiếp executable như trên. Trong VS Code chọn interpreter `.venv` và kernel `DDDCL Local`.

`requirements.in` pin dependency trực tiếp. `requirements.txt` là bản resolve transitive cho Windows/Python 3.11 có hashes; môi trường khác dùng lock riêng. Lock đã resolve không đồng nghĩa tất cả tests/huấn luyện đã chạy trên Windows. Agent phase A phải cài, `pip check`, import, tiny train và xuất environment report trên máy thật.

Nếu cập nhật dependency: tạo venv mới, dùng công cụ resolve được ghi phiên bản trong `ENVIRONMENT_STATUS.md`, regenerate lock, kiểm tra lại; không nâng package âm thầm. Không cài thêm CUDA/GPU vào môi trường tham chiếu rồi gộp kết quả với CPU. Ghi GPU/driver/CUDA nếu dùng profile riêng.

## 3. Tài khoản và token HF

1. Đăng nhập https://huggingface.co/join hoặc tài khoản đã có.
2. Vào https://huggingface.co/settings/tokens . Tạo token fine-grained có quyền cần thiết trên đúng dataset repo; upload cần write. Token đọc là đủ cho private read. Dữ liệu public non-gated thường không cần token để download.
3. Đăng nhập trên máy local bằng SDK. Cách này mở prompt nhập token, không để token trong command history:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import login; login(add_to_git_credential=False)"
```

Không cần `git clone` dataset HF. Token lưu trong cache đăng nhập của HF. Không gửi token cho AI/chat. Nếu dùng `.env`, copy `.env.example` sang `.env` rồi điền bằng editor local; đừng dùng token trong cell hoặc lệnh được chia sẻ. `HF_TOKEN` nếu có sẽ ưu tiên hơn token đăng nhập; khi lỗi quyền, kiểm tra biến này có bị cũ không mà không in token.

## 4. Lấy dataset từ nguồn public hoặc chuẩn bị release của nhóm

Thầy ưu tiên bản HF/Kaggle đã public. Phase B phải tìm và kiểm tra bản khớp; ghi URL/revision/số dòng/feature/nhãn. Không tuyên bố đã kiểm tra nếu chưa tải file.

Nếu không tìm được bản khớp được xác minh, công cụ đi kèm giúp tạo release pilot từ nguồn UCI gốc. Đây là bước chuẩn bị một lần, không phải cách loader train lén quay lại UCI:

```powershell
.\.venv\Scripts\python.exe tools\prepare_pilot_release.py --out hf_release
```

Công cụ tải 3 archive UCI, kiểm tra shape/nhãn, tạo CSV canonical, provenance, manifest với SHA256 và dataset card. Không tính scaler, KDE hoặc SMOTE, không huấn luyện. Mỗi dòng có row_id; label mapping được ghi lại (có thể khác mã số tác giả). Giữ raw archive làm evidence local, không đưa vào Git. Nếu rerun khác nội dung thì fail; chỉ ghi đè khi chủ động dùng `--overwrite`.

Chỉ dùng đường bootstrap này khi có lý do đáp ứng quy tắc nguồn dữ liệu của thầy và có quyền phân phối phù hợp. Công cụ có nguồn/license trong provenance; Agent kiểm tra lại trang nguồn trước phát hành. Không gán CC-BY cho bản CSV của tác giả chỉ dựa trên license UCI.

## 5. Upload một lần và khóa version

Thay `YOUR_HF_USERNAME` bằng username thực; không phải GitHub username nếu hai tài khoản khác nhau:

```powershell
.\.venv\Scripts\python.exe tools\publish_hf_release.py --repo-id YOUR_HF_USERNAME/dynamic-ddcl-benchmarks --folder hf_release --data-config configs/data_hf.json
```

Lệnh này **tạo/cập nhật dataset repo PUBLIC** và upload allowlist `manifest.json`, `README.md`, `data/*.csv`, `provenance/*.json`. Nó không upload `.env`, code, PDF hoặc toàn bộ project. Không có `delete_patterns`, không xóa file cũ. Không gọi từ notebook train. Chạy upload là hành động publish; người dùng quyết định bằng việc chạy command này. SDK hỗ trợ create_repo/upload_folder, không cần Git LFS thủ công.

Sau upload, script ghi `configs/data_hf.json` với commit SHA cụ thể và manifest checksum. Kiểm tra URL dataset, rồi commit config này cùng code; token không nằm trong config. Không tự viết `revision=main` cho run cuối.

## 6. Download từ cloud trong code

```powershell
.\.venv\Scripts\python.exe tools\check_hf_data.py --config configs/data_hf.json --dataset haberman
```

Trong Python/notebook (SDK Hub, không cần thư viện `datasets` để đọc CSV):

```python
from huggingface_hub import hf_hub_download
import pandas as pd
path = hf_hub_download(
    repo_id="YOUR_HF_USERNAME/dynamic-ddcl-benchmarks",
    repo_type="dataset",
    revision="FULL_DATASET_COMMIT_SHA",  # lấy từ config sau upload
    filename="data/haberman.csv",
)
df = pd.read_csv(path)
```

Đây là ví dụ API. Loader dự án phải kiểm tra cả manifest/file checksum/schema; dùng `tools/hf_data.py` làm công cụ hỗ trợ, rồi đưa interface phù hợp vào `src/`. Không chỉnh sửa file bên trong HF cache. `HF_HUB_OFFLINE=1` chỉ hoạt động nếu đúng revision đã cache; không dùng nó để né lần kiểm tra download sạch.

## 7. W&B và Git

Đăng nhập W&B bằng `python -m wandb login` hoặc API key local, không dán vào notebook. Run cuối phải sync vào **một project public**; offline chỉ là chế độ phát triển hoặc chờ sync, không đạt rubric cuối.

Trước commit: `git status`, kiểm tra diff và notebook outputs. Đưa code, docs, configs, lockfiles, metadata nhỏ vào Git. Không commit `.venv`, raw data/cache, token, model lớn. PDF nguồn giữ local; không bắt buộc push PDF vào repo public.

```powershell
git add AGENTS.md 00_START_HERE.md docs tools configs requirements.in requirements.txt .python-version .env.example .gitignore ENVIRONMENT_STATUS.md
git diff --cached --stat
git commit -m "Add Dynamic DDCL implementation specification and HF data workflow"
git push
```

Nếu repo chưa có upstream cho branch, làm theo branch hiện tại và remote đã kiểm tra; không mặc định force push main.
