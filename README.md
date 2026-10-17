# 🚗 SegVision QA - Semantic Segmentation (BDD100K 19 Classes)
> **Đề tài M33:** Fine-tune mô hình Transformer (**SegFormer-B0**) phân vùng ngữ nghĩa 19 lớp vật thể trên tập dữ liệu xe tự hành **BDD100K**.  
> Tích hợp công cụ **Mini-CVAT & Nền tảng Đánh giá Kiểm duyệt Chất lượng (QA Benchmark & Error Mapping)**.

---

## 🌟 Tính Năng Nổi Bật

1. **AI Model Fine-tuned Đỉnh Cao (SegFormer-B0 v2):**
   - Huấn luyện trực tiếp trên tập dữ liệu xe tự hành **BDD100K** (19 nhãn chuẩn).
   - Đạt mIoU vượt trội **56.77%** (V2 Final Checkpoint), so tài A/B trực tiếp với bản v1 (44.22%) và baseline Cityscapes.
   - Suy luận mượt mà trên cả CPU và GPU với tensor nội suy bilinear float32 độ nét cao (1280x720).

2. **Soi Lớp Trực Quan (Live Class Mask Hover Inspector):**
   - Rà chuột trực tiếp lên bất kỳ vị trí nào trên ảnh mask để hiển thị ngay tooltip thông minh: Icon, Tên Tiếng Anh, Tên Tiếng Việt (*Ô tô con, Mặt đường, Vỉa hè, Tòa nhà, Cây cối, Người đi bộ...*), Mã ID lớp và Tọa độ pixel (X, Y).
   - Tự động highlight dòng tương ứng ở thanh danh mục 19 lớp bên phải giúp phân biệt các lớp có tông màu tương tự nhau.
   - Bảng HUD Inspector tiện lợi cố định tại góc dưới màn hình làm việc.

3. **Chế độ 1: Dự đoán & Phân tích Tương tác (Inference Mode):**
   - Thanh trượt so sánh **Split Curtain Slider** đối chiếu trực tiếp giữa ảnh thực tế và mask AI.
   - Đối chứng A/B **Model Comparison** chia màn hình trực quan giữa 2 phiên bản mô hình.
   - Hiển thị viễn trắc **ADAS Telemetry & Phân tích Hành lang An toàn**.
   - Bảng lọc **19 Classes** bật/tắt linh hoạt từng lớp vật thể và tính toán phân bố diện tích pixel.

4. **Chế độ 2: Kiểm duyệt & Đo lường Chất lượng (QA Benchmark Mode):**
   - Nạp ảnh nhãn thật (**Ground Truth Mask**) để chấm điểm **mIoU** và **per-class IoU** tức thì.
   - Sinh **Bản đồ lỗi (Visual Error Map)**: Vùng xanh lá (Dự đoán đúng), Vùng đỏ (Dự đoán sai).
   - Xuất báo cáo kết quả đánh giá ra file **CSV** và tải file mask chuẩn BDD100K **PNG**.
   - Báo cáo kiểm toán mô hình chi tiết với biểu đồ radar, ma trận nhầm lẫn và khuyến nghị kỹ thuật.

5. **Chế độ 3: Gán nhãn AI & Sửa cọ Tương tác (Co-Pilot Annotation):**
   - Chỉnh sửa, tinh chỉnh mask bằng bút cọ thông minh (Smart Brush / Eraser) theo cơ chế Human-in-the-loop.
   - Hỗ trợ hoàn tác (Undo), đổi kích thước cọ và xuất mask huấn luyện TrainId.

6. **Tích hợp sẵn 100 Bộ Ảnh Mẫu Kiểm Thử (1-Click Test):**
   - Click chọn bất kỳ ảnh nào trong khay mẫu để nạp tự động cả ảnh gốc lẫn nhãn chuẩn.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
AAA_buildphase/
├── run.py                          # 🚀 Script khởi chạy vạn năng & tự động giải nén
├── run.bat                         # ⚡ 1-Click khởi chạy dành riêng cho Windows
├── run.sh                          # ⚡ 1-Click khởi chạy cho Linux / macOS
├── benchmark_100.py                # 📊 Script chạy benchmark đánh giá hàng loạt 100 ảnh mẫu
├── requirements.txt                # Thư viện Python phụ thuộc
├── .gitignore                      # Chặn rò rỉ file nén lớn và weights lên Git
├── README.md                       # Tài liệu tổng quan dự án
├── SETUP_GUIDE.md                  # Hướng dẫn kỹ thuật chi tiết
│
├── segvision_app/                  # Mã nguồn ứng dụng Web SegVision QA
│   ├── backend/
│   │   ├── main.py                 # FastAPI server & REST endpoints
│   │   ├── model_service.py        # Module nạp model SegFormer & inference
│   │   ├── evaluator.py            # Thuật toán tính IoU & sinh Error Heatmap
│   │   └── constants.py            # Danh mục 19 classes & bảng màu HEX/RGB
│   ├── frontend/
│   │   ├── index.html              # Giao diện Canvas Dark-Mode cao cấp
│   │   └── app.js                  # Logic tương tác chuột & gọi API
│   └── demo_samples/               # 5 cặp ảnh mẫu có sẵn theo Git (dùng ngay khi clone)
│
├── [TẢI TỪ DRIVE / ZALO]:
├── segformer_bdd100k_best.zip      # File zip chứa trọng số model fine-tune (~14MB)
└── bdd100k_sample_100.zip          # File zip chứa 100 cặp ảnh + mask BDD100k (~19MB)
```

---

## ⚡ Hướng Dẫn Dành Cho Thành Viên Nhóm

### Cách 1: Sử dụng AI Coding Agent (Cursor / Antigravity / Copilot / Cline) — KHUYÊN DÙNG ⭐
Sau khi clone repository về máy và tải 2 file zip (`segformer_bdd100k_best.zip` và `bdd100k_sample_100.zip`) bỏ vào thư mục gốc của dự án, bạn chỉ cần mở Agent và copy lệnh sau:

> *"Tôi vừa tải dự án SegVision QA và đã để 2 file zip model và sample ở thư mục gốc. Bạn hãy đọc file run.py, tạo môi trường ảo, cài requirements.txt và khởi chạy server web giúp tôi."*

Agent của bạn sẽ tự động làm hết mọi thứ và trả về link `http://127.0.0.1:8000` mà không gặp bất kỳ lỗi nào!

---

### Cách 2: Khởi chạy Nhanh bằng 1 Cú Click (Dành cho Người dùng)

1. Tải 2 file `.zip` do nhóm gửi và dán vào thư mục dự án:
   - `segformer_bdd100k_best.zip`
   - `bdd100k_sample_100.zip`
   *(Bạn không cần giải nén thủ công, script `run.py` sẽ tự giải nén khi khởi động!)*

2. Cài đặt thư viện (chỉ cần làm lần đầu):
   ```bash
   pip install -r requirements.txt
   ```

3. Khởi động Web App:
   - **Trên Windows:** Nhấp đúp chuột vào file `run.bat` (hoặc gõ `python run.py`).
   - **Trên macOS / Linux:** Gõ `./run.sh` (hoặc `python3 run.py`).

4. Mở trình duyệt Web truy cập:
   👉 **`http://127.0.0.1:8000`**

---

### Cách 3: Chạy Kiểm Thử Đánh Giá Benchmark Hàng Loạt (CLI)
Để chạy kiểm thử tự động toàn bộ 100 cặp ảnh và xuất bảng điểm mIoU chi tiết:
```bash
python benchmark_100.py
```
Kết quả thống kê mIoU từng lớp và thời gian xử lý sẽ được in trực tiếp trên màn hình và xuất thành file `benchmark_100_results.csv`.

---

## 🛡️ Hướng Dẫn Đẩy Code Lên Git Dành Cho Trưởng Nhóm

Hệ thống đã cấu hình sẵn file `.gitignore` để tự động loại trừ các file dữ liệu nặng (`*.zip`, thư mục dataset và weights lớn), giúp repo Git nhẹ nhàng (~2MB) và tải cực nhanh:

```bash
git init
git add .
git commit -m "feat: SegVision QA web app with SegFormer-B0 on BDD100K 19 classes"
git branch -M main
git remote add origin <LINK_REPO_GITHUB_CUA_BAN>
git push -u origin main
```
Sau đó bạn tải 2 file zip (`segformer_bdd100k_best.zip` và `bdd100k_sample_100.zip`) gửi vào nhóm Zalo / Google Drive để các bạn tải về đặt vào thư mục gốc là xong!
