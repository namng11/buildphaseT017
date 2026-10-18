# BÁO CÁO TỔNG KẾT DỰ ÁN BUILDPHASE (ĐỘI P-017)

* **Đề tài:** Fine-tune mô hình Semantic Segmentation trên tập dữ liệu xe tự hành BDD100K (19 classes) & Xây dựng nền tảng kiểm duyệt chất lượng gán nhãn SegVision QA.
* **Thời gian thực hiện:** Giai đoạn Buildphase
* **Mã nguồn công cụ:** [https://github.com/AI20K-Build-Phase-Cohort-4A/P-017/tree/main/source-tool/segvision-qa](https://github.com/AI20K-Build-Phase-Cohort-4A/P-017/tree/main/source-tool/segvision-qa)

---

## 1. Tóm tắt những gì nhóm đã thực sự làm được

Trong đợt buildphase này, nhóm tập trung làm trọn vẹn hai phần việc chính:

1. **Huấn luyện mô hình AI (SegFormer-B0):**
   - Lấy mô hình nền `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, tiến hành tiền xử lý nhãn và fine-tune trực tiếp trên tập dữ liệu đường phố BDD100K (19 lớp vật thể theo chuẩn Cityscapes/BDD100K).
   - Qua các lượt tinh chỉnh siêu tham số, mô hình đạt mức **56.77% mIoU** ở bản checkpoint v2 (tăng đáng kể so với bản v1 44.22% và bản baseline ban đầu bị lệch miền dữ liệu).
   - Xuất và lưu trữ checkpoint định dạng Hugging Face / SafeTensors gọn nhẹ (~14.9 MB).

2. **Xây dựng ứng dụng Web cục bộ SegVision QA (FastAPI + Vanilla JS/HTML Canvas):**
   - Viết trọn vẹn backend bằng FastAPI xử lý pipeline: nạp ảnh -> tiền xử lý -> suy luận PyTorch -> nội suy bilinear float32 -> trích xuất mask màu, viễn trắc ADAS, đo IoU và sinh bản đồ lỗi.
   - Frontend không dùng framework cồng kềnh (không Node.js/npm) để ai kéo code về cũng chỉ cần chạy 1 lệnh là dùng được ngay trên trình duyệt qua cổng `http://127.0.0.1:8000`.
   - Tích hợp 3 chế độ tương tác:
     + **Chế độ 1 (Inference & A/B Test):** Kéo thanh rèm chia đôi màn hình so sánh ảnh gốc vs mask, hoặc so sánh trực tiếp kết quả giữa 2 model v1 vs v2; phân tích hành lang an toàn ADAS.
     + **Chế độ 2 (QA Benchmark):** Nạp ảnh Ground Truth để máy tính tự chấm điểm mIoU từng lớp ngay lập tức, tự vẽ bản đồ lỗi trực quan (vùng đoán đúng xanh lá, đoán sai màu đỏ), xuất file CSV điểm số.
     + **Chế độ 3 (Co-Pilot Annotation):** Dùng cọ thông minh (Smart Brush) để người dùng sửa trực tiếp các viền mask bị model dự đoán lem, sau đó xuất ra file ảnh nhãn TrainId chuẩn để tái huấn luyện.
     + **Tính năng soi lớp (Hover Inspector):** Di chuột đến bất kỳ điểm ảnh nào trên mask thì popup hiện ngay tên lớp bằng tiếng Việt + tiếng Anh + icon + ID + tọa độ (X, Y), đồng thời highlight dòng lớp đó ở bảng bên phải.
   - Chuẩn bị sẵn bộ 100 cặp ảnh mẫu kiểm thử để bấm 1 nút là nạp cả ảnh gốc lẫn nhãn thật vào test ngay.

---

## 2. Các Pain Points thực tế và Cách nhóm giải quyết

Khi bắt tay vào làm đồ án này, nhóm nhận thấy một số vấn đề rất thực tế sau đây và tập trung giải quyết:

### Pain point 1: Mô hình có sẵn từ Cityscapes chạy trên BDD100K rất tệ
* **Thực tế:** Bộ dữ liệu Cityscapes chủ yếu chụp đường phố nước Đức ban ngày, trời trong. BDD100K lại gồm camera hành trình thực tế tại Mỹ với đủ loại thời tiết: trời mưa, ban đêm, ngược sáng, tuyết, đường đông xe. Khi lấy nguyên model Cityscapes sang chạy BDD100K thì bị "domain shift", mask bị vỡ và lem nhem, mIoU chỉ quanh quẩn dưới 40%.
* **Cách giải quyết:** Nhóm giữ lại backbone Mix-Transformer B0 của SegFormer vì nó nhẹ, nhưng huấn luyện lại Decoder Head với dữ liệu BDD100K. Nhờ đó mIoU kéo lên được 44.22% (v1) và hoàn thiện ở mức 56.77% (v2). Các lớp lớn như Mặt đường (Road), Xe con (Car), Tòa nhà (Building), Cây cối (Vegetation), Bầu trời (Sky) đều đạt IoU từ 80% - 94%.

### Pain point 2: Kiểm thử AI thông thường chỉ xem số trên terminal, rất mù mờ
* **Thực tế:** Thường khi train xong một model segmentation, mọi người chỉ nhìn số mIoU chung chung trong notebook. Nhưng khi người lái xe hoặc kỹ sư QA cần xem thì không biết model đang bị ngu ở góc nào, đoán nhầm người đi bộ thành cột điện hay đoán nhầm vỉa hè thành mặt đường.
* **Cách giải quyết:** Nhóm làm hẳn tính năng **Visual Error Heatmap**. Khi nạp nhãn chuẩn vào, hệ thống so sánh từng pixel và phủ màu: điểm nào đúng thì hiện xanh lá mờ, điểm nào đoán sai thì lập tức tô màu đỏ cảnh báo. Nhìn vào là biết ngay model hay bị sai ở mép viền của xe hoặc các biển báo ở xa.

### Pain point 3: Bảng màu 19 lớp xe tự hành quá giống nhau, người dùng không phân biệt nổi
* **Thực tế:** Chuẩn bảng màu 19 classes của Cityscapes/BDD100K có nhiều màu rất sát nhau: xe con, xe tải, xe buýt đều là tông xanh dương/tím; mặt đường, vỉa hè, tường rào đều là các sắc độ xám xịt; cây cối và cỏ bãi đất đều màu xanh lá. Người test nhìn vào không thể nhớ hết mã màu.
* **Cách giải quyết:** Nhóm bổ sung bộ **Live Hover Inspector**. Người dùng chỉ việc rê chuột vào bức ảnh, hệ thống dùng một canvas ngầm đọc giá trị màu RGB của pixel dưới con trỏ với độ trễ gần như bằng 0, tra ngược vào bảng mã màu và bắn popup thông báo tiếng Việt rõ ràng: "🚗 Car (Ô tô con) - ID 13" hay "🚏 Sidewalk (Vỉa hè) - ID 1", đồng thời bảng danh mục lớp bên tay phải tự cuộn tới và sáng lên.

### Pain point 4: Gán nhãn thủ công (Data Annotation) từ đầu quá tốn thời gian
* **Thực tế:** Ngồi vẽ polygon từng chiếc xe, từng khúc đường trên công cụ như CVAT rất mệt và lâu.
* **Cách giải quyết:** Nhóm làm chế độ **Co-Pilot**: Cho model chạy trước để tạo khung mask thô (tiết kiệm được khoảng 70-80% thời gian vẽ tay ban đầu), người dùng chỉ việc cầm cọ (Brush/Eraser) tỉa lại những chỗ lem nhem rồi bấm xuất file mask TrainId.

---

## 3. Kết quả thực nghiệm đạt được

| Mô hình / Phiên bản | Tập dữ liệu | mIoU trung bình | Kích thước file | Thời gian xử lý (CPU i5/i7) |
|---|---|---|---|---|
| **SegFormer-B0 Baseline** | Cityscapes Pretrained | ~38.5% (khi test trên BDD) | ~14.9 MB | ~450ms |
| **SegFormer-B0 Fine-tuned v1** | BDD100K Split v1 | 44.22% | ~14.9 MB | ~480ms |
| **SegFormer-B0 Fine-tuned v2 (Bản nộp)** | BDD100K Final (24 epochs) | **56.77%** | **14.9 MB** | **~500ms** |

* **Đánh giá theo từng lớp (Per-class IoU của bản v2):**
  - Nhóm lớp tốt nhất: *Sky (94.9%)*, *Road (93.1%)*, *Car (85.3%)*, *Vegetation (83.4%)*, *Building (81.0%)*.
  - Nhóm lớp trung bình: *Sidewalk (58.2%)*, *Bus (54.1%)*, *Truck (49.6%)*, *Person (46.3%)*.
  - Nhóm lớp thấp: *Pole (31.2%)*, *Traffic Sign (28.7%)*, *Bicycle (22.4%)*, *Rider (19.8%)*.

---

## 4. Những hạn chế và yếu điểm lớn nhất của Buildphase (Nói thẳng thắn)

Dù đã hoàn thành đầy đủ các yêu cầu đề ra, nhóm tự nhận thấy dự án vẫn còn những điểm yếu và hạn chế kỹ thuật rõ rệt sau:

### 1. Vấn đề giới hạn tài nguyên và triển khai lên Cloud (Yếu điểm lớn nhất)
- Mô hình PyTorch khi chạy ở độ phân giải gốc 1280x720 và thực hiện nội suy tensor float32 cần lượng RAM từ 800MB - 1.2GB lúc cao điểm suy luận.
- Khi nhóm thử đưa lên server miễn phí của Render (bị giới hạn cứng 512MB RAM), hệ thống lập tức bị tràn bộ nhớ và sập dịch vụ (Out of Memory - Restart).
- Để giải quyết tình huống trên Render, nhóm đã thử phương án chuyển đổi sang ONNX Runtime và hạ độ phân giải, nhưng việc này lại làm cho mask bị răng cưa, chất lượng đường viền không được mềm và đẹp như lúc chạy PyTorch gốc.
- **Thực tế:** Vì thế, bản tối ưu nhất, mượt nhất của dự án hiện tại là bản chạy **Local** (trên máy cá nhân có từ 4GB RAM trở lên hoặc có GPU), chưa thể có một bản Cloud Free vừa chạy PyTorch phân giải cao vừa không tốn tiền nâng cấp RAM.

### 2. Mô hình SegFormer-B0 quá nhẹ nên nhận diện vật thể nhỏ rất kém
- Nhóm chọn bản **B0** (chỉ có ~3.7 triệu tham số) với mục tiêu ưu tiên tốc độ suy luận nhanh trên CPU thông thường và dung lượng file nhẹ.
- Nhưng sự đánh đổi là các vật thể có diện tích nhỏ hoặc mỏng manh trên đường như: *Cột đèn (Pole)*, *Biển báo (Traffic Sign)*, *Người đi xe máy/xe đạp (Rider/Bicycle)* có IoU rất thấp (chỉ dưới 30%). Khi xe ở xa tít cuối đường, model gần như bỏ sót hoặc nhận nhầm vào nền trời/tòa nhà.
- Nhóm đã thử nghiệm cấu hình cho mô hình lớn hơn là SegFormer-B1 HD trên Google Colab, tuy nhiên do tài nguyên GPU T4 miễn phí bị giới hạn thời gian (timeout ngắt phiên) nên chưa kịp huấn luyện xong bản B1 HD để đưa vào sản phẩm cuối cùng.

### 3. Tập dữ liệu huấn luyện mới dừng ở mức Subset mẫu
- Tập BDD100K đầy đủ có tới 70.000 ảnh train và 10.000 ảnh val độ phân giải cao.
- Vì không có server GPU riêng và chỉ tận dụng Google Colab miễn phí, nhóm buộc phải trích xuất một tập dữ liệu con (subset) để huấn luyện trong khoảng 20-25 epochs. Nếu có điều kiện train trên toàn bộ 70.000 ảnh với batch size lớn hơn, chắc chắn mIoU của các lớp khó sẽ còn cao hơn nữa.

### 4. Công cụ vẽ cọ (Smart Brush) trên Web còn ở mức cơ bản
- Chế độ sửa nhãn mới chỉ dừng ở mức vẽ pixel 2D trên HTML5 Canvas (vẽ hình tròn theo bán kính chuột).
- Nó chưa có tính năng thông minh cao cấp như tự bắt dính viền vật thể (Snapping), chưa hỗ trợ vẽ Polygon vector nhiều điểm như CVAT chuyên nghiệp, và chưa tích hợp được mô hình SAM (Segment Anything) do SAM quá nặng để chạy real-time trên máy không có card màn hình rời.

### 5. Ứng dụng Web là dạng Single-user (Cục bộ), chưa có Database quản lý
- Toàn bộ trạng thái ảnh, mask, lịch sử vẽ hiện đang lưu trên bộ nhớ RAM của trình duyệt người dùng.
- Chưa có hệ thống tài khoản (User Authentication), chưa có cơ sở dữ liệu (PostgreSQL/MongoDB) để lưu trữ dự án dài hạn hay cho phép nhiều người cùng vào gán nhãn một bộ ảnh như một hệ thống thương mại hoàn chỉnh.

---

## 5. Hướng phát triển nếu tiếp tục thực hiện

1. **Về mô hình:** Huấn luyện hoàn tất SegFormer-B1 hoặc SegFormer-B2 trên toàn bộ tập dữ liệu BDD100K để cải thiện dứt điểm IoU của nhóm vật thể nhỏ (biển báo, người đi xe máy).
2. **Về triển khai Cloud:** Tối ưu hóa mô hình bằng kỹ thuật Lượng tử hóa (INT8 Quantization qua ONNX/TensorRT) kết hợp chia tile ảnh thông minh để ứng dụng có thể chạy êm ru trên các gói Cloud 512MB RAM mà không bị vỡ hạt mask.
3. **Về tính năng gán nhãn:** Bổ sung thuật toán bắt cạnh viền thông minh (Contour snap) và xuất nhãn trực tiếp sang định dạng chuẩn của CVAT / COCO JSON.

---

## 6. Kết luận

Nhìn chung trong giai đoạn Buildphase này, nhóm Đội T-017 đã giải quyết được trọn vẹn mục tiêu cốt lõi: **đưa được một mô hình Transformer thực tế lên tập dữ liệu xe tự hành BDD100K, kéo độ chính xác mIoU từ mức không dùng được lên 56.77%, và dựng được một ứng dụng web kiểm duyệt SegVision QA chạy mượt mà, trực quan với đầy đủ tính năng đo lỗi và hỗ trợ gán nhãn**. 

Những thiếu sót còn lại chủ yếu đến từ rào cản tài nguyên phần cứng miễn phí (RAM cloud và GPU train), nhóm đã ghi nhận chi tiết và có phương án kỹ thuật rõ ràng để khắc phục nếu tiếp tục phát triển ở các giai đoạn sau.
