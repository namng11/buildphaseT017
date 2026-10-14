# HƯỚNG DẪN CÀI ĐẶT & CHẠY LOCAL DỰ ÁN SEGVISION QA
*(Tài liệu chuẩn dành cho Thành viên Nhóm và AI Coding Agent: Cursor, Antigravity, Copilot, Cline)*

---

## 1. Giới thiệu Dự án
**SegVision QA** là công cụ Mini-CVAT kiêm nền tảng kiểm duyệt và đánh giá mô hình **Semantic Segmentation** trên tập dữ liệu xe tự hành **BDD100K (19 classes)**.
- **Backend:** Python FastAPI + PyTorch + Hugging Face Transformers (SegFormer-B0).
- **Frontend:** Single Page App (HTML5 Canvas + Tailwind CSS Dark Mode).
- **Chức năng chính:**
  1. *Chế độ 1 (Inference):* Kéo thả ảnh đường phố, AI tự vẽ mask trong ~300ms, thanh trượt so sánh Split Curtain, bộ lọc ẩn/hiện 19 lớp.
  2. *Chế độ 2 (QA Benchmark):* Nạp ảnh Ground Truth để tính toán thời gian thực chỉ số **mIoU**, **per-class IoU** và sinh ra **Bản đồ lỗi (Error Map)** xanh/đỏ trực quan.
  3. *Tích hợp sẵn 100 ảnh mẫu:* Tự động nạp kiểm thử ngay chỉ bằng 1 click chuột.

---

## 2. Cấu trúc Thư mục Dự án

```text
AAA_buildphase/
├── run.py                          # 🚀 Script khởi chạy vạn năng & tự động giải nén zip
├── run.bat                         # ⚡ 1-Click khởi chạy cho Windows
├── run.sh                          # ⚡ 1-Click khởi chạy cho Linux / macOS
├── benchmark_100.py                # 📊 Đánh giá mIoU hàng loạt 100 ảnh mẫu
├── requirements.txt                # Thư viện Python
├── .gitignore                      # Chặn rò rỉ file nén và dữ liệu nặng lên Git
├── README.md                       # Giới thiệu & Hướng dẫn tổng quan
├── SETUP_GUIDE.md                  # File hướng dẫn kỹ thuật này
│
├── segvision_app/                  # Mã nguồn ứng dụng
│   ├── backend/
│   │   ├── constants.py            # Bảng màu & 19 nhãn chuẩn BDD100k
│   │   ├── evaluator.py            # Thuật toán tính IoU & sinh Error Heatmap
│   │   ├── model_service.py        # Quản lý nạp model SegFormer & inference
│   │   └── main.py                 # FastAPI REST API server
│   ├── frontend/
│   │   ├── index.html              # Giao diện Dark-mode Canvas
│   │   └── app.js                  # Logic tương tác client & Canvas
│   └── demo_samples/               # 5 cặp ảnh mẫu có sẵn theo Git
│
├── [CẦN TẢI TỪ DRIVE / ZALO VỀ THƯ MỤC GỐC]:
├── segformer_bdd100k_best.zip      # File zip chứa trọng số model fine-tune (~14MB)
└── bdd100k_sample_100.zip          # File zip chứa 100 cặp ảnh + mask BDD100k (~19MB)
```

---

## 3. CƠ CHẾ TỰ ĐỘNG THÔNG MINH (AUTO-SETUP)
Khi bạn đặt 2 file `segformer_bdd100k_best.zip` và `bdd100k_sample_100.zip` vào thư mục gốc của dự án, script `run.py` sẽ **tự động kiểm tra và giải nén** đúng vị trí cấu trúc:
- `segformer_bdd100k_best.zip` ➡️ giải nén thành thư mục `./segformer_bdd100k_best/`
- `bdd100k_sample_100.zip` ➡️ giải nén thành thư mục `./content/bdd100k_sample_100/`

*(Bạn KHÔNG cần phải tự mở phần mềm giải nén hay copy thủ công!)*

---

## 4. HƯỚNG DẪN DÀNH CHO AI CODING AGENT (CURSOR / ANTIGRAVITY / COPILOT)

> **Prompt mẫu gửi cho Agent:**
> ```text
> Tôi vừa clone repo SegVision QA về máy và đã dán 2 file zip 'segformer_bdd100k_best.zip' và 'bdd100k_sample_100.zip' vào thư mục gốc dự án.
> Bạn hãy:
> 1. Đọc file run.py và requirements.txt.
> 2. Thiết lập môi trường Python ảo và cài đặt dependencies nếu chưa có.
> 3. Khởi chạy server web bằng lệnh 'python run.py'.
> 4. Kiểm tra đảm bảo server chạy thành công trên http://127.0.0.1:8000 và phản hồi cho tôi.
> ```

---

## 5. HƯỚNG DẪN KHỞI CHẠY THỦ CÔNG (CHO THÀNH VIÊN)

### Bước 1: Tạo môi trường ảo (Khuyên dùng)
*Trên Windows (PowerShell):*
```bash
python -m venv .venv
.venv\Scripts\activate
```

*Trên Linux / macOS:*
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi chạy máy chủ
- **Cách đơn giản nhất:** 
  - Windows: Nhấp đúp vào `run.bat` (hoặc chạy `python run.py`).
  - Linux / Mac: Chạy `./run.sh` (hoặc `python3 run.py`).
- Máy chủ sẽ xuất hiện thông báo:
  ```text
  [SegVision QA] May chu Web Backend dang khoi dong...
  [URL Truy cap] -> http://127.0.0.1:8000
  ```

### Bước 4: Mở trình duyệt Web
Mở Chrome, Edge hoặc Firefox và truy cập:  
👉 **`http://127.0.0.1:8000`**

- Bạn sẽ thấy khay **100 ảnh mẫu**. Nhấp vào bất kỳ ảnh nào để xem AI nhận diện mặt đường, vỉa hè, xe cộ, tòa nhà trong nháy mắt!
- Chuyển sang tab **"2. Kiểm duyệt & Đo mIoU (QA Benchmark)"** để xem điểm số mIoU và bản đồ phân tích lỗi (Error Map).

---

## 6. CHẠY BENCHMARK TOÀN BỘ 100 ẢNH MẪU (CLI)
Để kiểm tra độ ổn định và tính mIoU trên toàn bộ 100 ảnh:
```bash
python benchmark_100.py
```
Hệ thống sẽ chạy qua 100 ảnh, tính tổng giao/hợp và xuất file `benchmark_100_results.csv`.

---

## 7. HƯỚNG DẪN ĐẨY GIT (CHO TRƯỞNG NHÓM)

File `.gitignore` đã chặn sẵn các file `.zip`, weights nặng và dữ liệu tạm:
```bash
# 1. Khởi tạo kho git (nếu chưa khởi tạo)
git init

# 2. Thêm toàn bộ mã nguồn
git add .

# 3. Kiểm tra xem các file .zip có bị đẩy nhầm không (phải KHÔNG có tên file .zip xuất hiện)
git status

# 4. Tạo commit đầu tiên
git commit -m "feat: SegVision QA Semantic Segmentation Web App (BDD100K 19 classes)"

# 5. Đẩy lên GitHub
git branch -M main
git remote add origin https://github.com/<tai-khoan>/<ten-repo>.git
git push -u origin main
```
