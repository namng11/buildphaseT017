# TÀI LIỆU PHÂN TÍCH, THIẾT KẾ VÀ KẾ HOẠCH XÂY DỰNG WEB "MINI-CVAT & MODEL EVALUATION TOOL"

Tài liệu này đóng vai trò là bản thiết kế kỹ thuật (Technical Specification & Architecture Blueprint) chi tiết để chuẩn bị bắt tay vào code ứng dụng web kiểm duyệt và đánh giá mô hình Semantic Segmentation trên tập dữ liệu BDD100K.

---

## 1. Mục tiêu và Định vị Sản phẩm

* **Tên ứng dụng dự kiến:** **SegVision QA (Mini-CVAT for Semantic Segmentation)**
* **Mục đích:** Không dừng lại ở một ứng dụng hiển thị ảnh thông thường (Viewer), hệ thống được thiết kế như một **Công cụ kiểm duyệt chất lượng gán nhãn và đo đạc hiệu năng mô hình (Model Evaluation & Annotation QA Tool)** chuẩn công nghiệp.
* **Người dùng mục tiêu:** 
  * Giảng viên / Hội đồng chấm đồ án: Đánh giá trực quan sự khác biệt trước và sau fine-tune, soi từng chi tiết lỗi (Error Heatmap).
  * Kỹ sư AI / Người gán nhãn: Kiểm tra chất lượng mô hình, đo đạc chỉ số IoU thời gian thực trên từng ảnh, xuất kết quả sang các định dạng chuẩn.

---

## 2. Kiến trúc Hệ thống (System Architecture)

Ứng dụng được xây dựng theo mô hình **Client - Server nhẹ (Local Decoupled Architecture)**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   GIAO DIỆN NGƯỜI DÙNG (FRONTEND SPA)                  │
│  - HTML5 Canvas tương tác cao (Hỗ trợ Zoom / Pan / Split Slider)       │
│  - Dark Theme chuẩn CVAT / Photoshop (Tailwind CSS + Lucide Icons)     │
│  - Bộ lọc 19 lớp (Layer Toggle), Thanh kéo Opacity, Bảng Metric IoU    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST API (JSON / FormData)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    MÁY CHỦ XỬ LÝ (BACKEND FASTAPI)                     │
│  ├── Controller (Routes): Quản lý upload ảnh, điều phối luồng          │
│  ├── Model Service: Nạp SegFormer-B0, tiền xử lý, chạy inference       │
│  ├── Metrics Engine: Tính toán mIoU, Per-class IoU, sinh Error Map     │
│  └── Export Service: Xuất ảnh mask màu, file PNG ID (0-18), JSON       │
└────────────────────────────────────────────────────────────────────────┘
```

### Công nghệ Lựa chọn:
* **Backend:** **Python 3.10+ & FastAPI** (Tốc độ xử lý bất đồng bộ cao, nhẹ gấp 3 lần Flask/Django, tài liệu Swagger tự động sinh).
* **AI Core:** **PyTorch & Hugging Face Transformers** (Nạp SegFormer-B0, chạy mượt mà cả trên CPU lẫn GPU rời).
* **Frontend:** **Vanilla HTML5 + Modern JavaScript (ES6) + Canvas API + Tailwind CSS** (Không cần cài đặt Node.js/npm cồng kềnh, chỉ cần mở trình duyệt là chạy, tối ưu hiệu năng render ma trận ảnh).

---

## 3. Thiết kế Tính năng Chi tiết (Feature Specifications)

Hệ thống hoạt động với **2 chế độ làm việc chính**:

### Chế độ 1: Dự đoán & Trực quan hóa (Inference Mode)
* **Đầu vào:** Chỉ cần 01 ảnh đường phố gốc (JPG/PNG).
* **Chức năng:**
  1. **Thanh trượt So sánh 2 nửa (Split Curtain Slider):** Người dùng kéo thanh chia đôi màn hình: nửa trái là ảnh gốc thật, nửa phải là lớp màu AI đã phân vùng.
  2. **Bảng điều khiển 19 Classes:** Cho phép tích chọn/bỏ chọn từng nhãn riêng lẻ (Ví dụ: tắt nhãn *Road* để giấu mặt đường, chỉ soi ranh giới *Xe* và *Người*).
  3. **Thanh trượt Opacity:** Điều chỉnh độ trong suốt của lớp phủ mask từ 0% (hiện ảnh gốc) đến 100% (chỉ hiện màu mask).
  4. **Thống kê diện tích (Pixel Distribution):** Tự động tính toán tỷ lệ % diện tích từng lớp xuất hiện trong bức ảnh.

### Chế độ 2: Kiểm duyệt & Đánh giá Ground Truth (Evaluation & QA Mode) — *Tính năng trọng tâm*
* **Đầu vào:** 01 ảnh gốc + 01 ảnh nhãn chuẩn Ground Truth (Mask PNG).
* **Chức năng:**
  1. **Tính toán IoU thời gian thực (Live IoU Calculator):**
     * Tính toán ngay lập tức **mIoU toàn ảnh** (Ví dụ: `78.2%`).
     * Xuất bảng **IoU chi tiết cho từng lớp xuất hiện trong ảnh** (Ví dụ: *Road: 93.1%, Car: 85.4%, Sidewalk: 67.2%...*).
  2. **Bản đồ phát hiện lỗi (Error / Difference Heatmap):**
     * Tự động sinh ra lớp bản đồ màu phản ánh độ sai lệch:
       * **Màu Xanh lá (True Positive):** Vùng AI nhận diện đúng 100% với nhãn thật.
       * **Màu Đỏ (False Positive):** Vùng AI nhận diện thừa (cắt lấn sang đối tượng khác).
       * **Màu Xanh dương / Vàng (False Negative):** Vùng AI bỏ sót nhãn.
  3. **So sánh Trước vs Sau Fine-tune (Model Benchmark Toggle):**
     * Nút chọn nhanh giữa **Pretrained Baseline** (Mô hình gốc Cityscapes) và **Fine-tuned Model** (Mô hình nhóm tự train trên BDD100k) để chứng minh hiệu quả huấn luyện.

---

## 4. Thiết kế Giao diện (UI/UX Wireframe)

Giao diện được bố trí theo phong cách công cụ chuyên nghiệp (Professional Annotation Tool):

```text
+---------------------------------------------------------------------------------------------------+
|  [Logo] SegVision QA - BDD100K Tool   | Model: [SegFormer-B0 Fine-tuned v] | Hardware: CPU (210ms)|
+---------------------------------------------------------------------------------------------------+
| BẢNG ĐIỀU KHIỂN TRÁI  |                       KHUNG VẼ CHÍNH (CANVAS)          | BẢNG CHỈ SỐ PHẢI |
|                       |                                                        |                  |
| 1. Upload ảnh:        |   [Ảnh gốc]   | [Thanh trượt] |   [Lớp Mask AI]        | 🎯 CHỈ SỐ IOU:   |
|   [ Chọn ảnh gốc ]    |               |               |                        | mIoU: 78.4%      |
|   [ Chọn Ground Truth]|                                                        | Accuracy: 92.1%  |
|                       |   (Hỗ trợ cuộn chuột Zoom in/out, kéo chuột Pan ảnh)   |                  |
| 2. Chế độ xem:        |                                                        | 📊 PER-CLASS IOU:|
|   (o) Split Slider    +--------------------------------------------------------+ - Road:    94.2% |
|   ( ) Mask Overlay    | THANH CÔNG CỤ: [Zoom 100%] [Fit Screen] [Bản đồ lỗi]   | - Car:     86.5% |
|   ( ) Error Heatmap   +--------------------------------------------------------+ - Sidewalk:68.1% |
|                       |                                                        | - Sky:     95.0% |
| 3. Độ mờ (Opacity):   |                                                        |                  |
|   [======o=====] 60%  |                                                        | 🎨 19 NHÃN (LỌC):|
|                       |                                                        | [v] Road (Tím)   |
| [ Xuất Mask PNG ]     |                                                        | [v] Car (Xanh)   |
| [ Xuất Báo cáo CSV ]  |                                                        | [ ] Building     |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Thiết kế Thuật toán & API Contract

### A. Công thức tính IoU thời gian thực (Metrics Logic)
Với 2 ma trận kích thước $(H, W)$: $P$ (Dự đoán của AI) và $G$ (Ground Truth nhãn thật), với mỗi lớp $c \in [0..18]$:
$$\text{Intersection}_c = \sum (P == c \land G == c \land G \neq 255)$$
$$\text{Union}_c = \sum ((P == c \lor G == c) \land G \neq 255)$$
$$\text{IoU}_c = \frac{\text{Intersection}_c}{\text{Union}_c}$$
$$\text{mIoU} = \frac{1}{|C_{\text{present}}|} \sum_{c \in C_{\text{present}}} \text{IoU}_c$$

### B. Danh sách API Endpoints (FastAPI)
1. `GET /api/classes`
   * Trả về danh sách 19 lớp, ID tương ứng và mã màu chuẩn RGB/HEX của BDD100k.
2. `POST /api/predict`
   * Nhận: File ảnh (`image`).
   * Trả về: Ảnh mask màu Base64, bảng phân bố % pixel, thời gian suy luận (latency ms).
3. `POST /api/evaluate`
   * Nhận: File ảnh (`image`) + File nhãn thật (`mask`).
   * Trả về: Ảnh dự đoán, ảnh Ground Truth, ảnh Error Map, chỉ số mIoU, danh sách per-class IoU.

---

## 6. Kế hoạch Triển khai (4 Bước)

| Bước | Nội dung công việc | Kết quả đầu ra |
| :--- | :--- | :--- |
| **Bước 1** | Khởi tạo cấu trúc thư mục backend & dịch vụ AI (`model_service.py`, `evaluator.py`). | Nạp được SegFormer, tính toán ma trận IoU và sinh bản đồ lỗi chính xác qua hàm test. |
| **Bước 2** | Xây dựng các API Endpoint bằng FastAPI (`main.py`). | API `/predict` và `/evaluate` phản hồi kết quả trong vòng < 300ms. |
| **Bước 3** | Thiết kế giao diện Web Single Page (`index.html`, Canvas JS). | Giao diện Dark-mode hoàn chỉnh, thanh trượt Split Slider, bật/tắt 19 lớp, hiển thị biểu đồ IoU. |
| **Bước 4** | Kiểm thử đầu-cuối (End-to-End Testing) với dữ liệu mẫu BDD100K. | Chạy thử nghiệm mượt mà trên trình duyệt `http://localhost:8000`, sẵn sàng nạp model fine-tune. |
