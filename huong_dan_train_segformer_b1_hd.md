# HƯỚNG DẪN HUẤN LUYỆN SEGFORMER-B1 ĐỘ PHÂN GIẢI GỐC 720P (1280x720) TRÊN GOOGLE COLAB

Tài liệu này hướng dẫn chi tiết quy trình huấn luyện nâng cao mô hình **SegFormer-B1** giữ nguyên **độ phân giải gốc 1280x720** trên tập dữ liệu **BDD100K (19 classes)** để đạt chỉ số mIoU vượt trội (>56% - 60%) và xóa bỏ hiện tượng các lớp hiếm bị 0.00%.

---

## 1. Mục tiêu và Cải tiến Kỹ thuật so với bản B0 cũ

| Tiêu chí | Bản B0 cũ (Đã chạy) | Bản B1 Nâng cấp (Kế hoạch này) |
| :--- | :--- | :--- |
| **Kiến trúc mô hình** | SegFormer-B0 (3.7M tham số) | **SegFormer-B1 (14M tham số)** |
| **Trọng số khởi điểm** | `nvidia/mit-b0` (Đầu 19 nhãn ngẫu nhiên) | **`nvidia/segformer-b1-finetuned-cityscapes-1024-1024`** (Đã học sẵn 19 nhãn đường phố) |
| **Độ phân giải ảnh** | Bị nén về 512 x 512 (mờ chi tiết) | **Giữ nguyên gốc 1280 x 720 (Cực nét)** |
| **Các lớp nhỏ & hiếm** | `rider`, `motorcycle`, `bicycle` bị 0.00% | **Có điểm số cao ngay từ Epoch 1** |
| **mIoU kỳ vọng** | 44.22% | **~56% – 60%+** |
| **Tài nguyên VRAM T4** | ~5 GB VRAM | **~7.5 GB VRAM** (Rất an toàn trong 15GB của T4) |

---

## 2. Các bước thực hiện trên Google Colab (Copy - Paste)

### BƯỚC 0: Chuẩn bị Colab
1. Mở [Google Colab](https://colab.research.google.com/) -> Bấm **New notebook**.
2. Menu **Runtime** -> **Change runtime type** -> Chọn **T4 GPU** -> Bấm **Save**.
3. Kéo thả file **`kaggle.json`** vào khung quản lý file bên trái màn hình.

---

### BƯỚC 1: Kết nối Google Drive & Cài đặt thư viện (Cell 1)
```python
from google.colab import drive
import torch

drive.mount('/content/drive')
!pip install -q transformers datasets evaluate accelerate albumentations

print("✅ GPU T4 đã sẵn sàng:", torch.cuda.get_device_name(0))
```

---

### BƯỚC 2: Tải và Giải nén Dữ liệu BDD100K 10K (Cell 2)
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Tải và giải nén dữ liệu BDD100K Semantic Segmentation
!kaggle datasets download -d solesensei/solesensei_bdd100k
!unzip -q solesensei_bdd100k.zip -d /content/bdd100k_data
print("✅ Giải nén dữ liệu hoàn tất!")
```

---

### BƯỚC 3: Dataset 1280x720 & Nạp SegFormer-B1 Cityscapes (Cell 3)
```python
import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

id2label = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
}
label2id = {v: k for k, v in id2label.items()}

# GIỮ NGUYÊN ĐỘ PHÂN GIẢI GỐC BDD100K 1280x720
processor = SegformerImageProcessor(
    do_reduce_labels=False, 
    size={"height": 720, "width": 1280}
)

class BDD100kDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_dir = mask_dir
        self.processor = processor

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{stem}_train_id.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, f"{stem}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        encoded = self.processor(image, mask, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoded.items()}

BASE = "/content/bdd100k_data/bdd100k_seg/bdd100k/seg"
train_dataset = BDD100kDataset(f"{BASE}/images/train", f"{BASE}/labels/train", processor)
val_dataset = BDD100kDataset(f"{BASE}/images/val", f"{BASE}/labels/val", processor)

# CHECKPOINT ĐÃ PRETRAIN 19 LỚP ĐƯỜNG PHỐ CỦA NVIDIA
MODEL_CHECKPOINT = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=19,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
print(f"✅ Đã nạp {len(train_dataset)} ảnh train (1280x720), {len(val_dataset)} ảnh val.")
print("✅ Khởi tạo SegFormer-B1 Pretrained Cityscapes thành công!")
```

---

### BƯỚC 4: Huấn luyện 8 Epochs & Lưu Model vào Drive (Cell 4)
```python
import evaluate, numpy as np
import torch
from transformers import TrainingArguments, Trainer

metric = evaluate.load("mean_iou")

# Hàm tính mIoU theo cụm 16 ảnh để chống tràn RAM khi chạy ảnh HD 720p
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    num_samples = logits.shape[0]
    preds = np.zeros(labels.shape, dtype=np.uint8)
    
    chunk_size = 16
    with torch.no_grad():
        for i in range(0, num_samples, chunk_size):
            chunk_logits = torch.from_numpy(logits[i : i + chunk_size])
            chunk_h, chunk_w = labels.shape[1], labels.shape[2]
            chunk_upsampled = torch.nn.functional.interpolate(
                chunk_logits, size=(chunk_h, chunk_w), mode="bilinear", align_corners=False
            )
            preds[i : i + chunk_size] = chunk_upsampled.argmax(dim=1).cpu().numpy().astype(np.uint8)
            del chunk_logits, chunk_upsampled

    metrics = metric.compute(
        predictions=preds, references=labels, num_labels=19, ignore_index=255, reduce_labels=False
    )
    
    print("\n" + "="*45)
    print("📊 BẢNG PER-CLASS IOU (ĐỘ PHÂN GIẢI GỐC 1280x720):")
    print("="*45)
    for i, name in id2label.items():
        val = metrics["per_category_iou"][i] * 100 if not np.isnan(metrics["per_category_iou"][i]) else 0.0
        print(f"Lớp {i:02d} ({name:<15}): {val:.2f}%")
    print("="*45)
    print(f">> ⭐ mIoU Trung Bình: {metrics['mean_iou']*100:.2f}%\n" + "="*45)
    
    return {"mean_iou": metrics["mean_iou"]}

# Lưu vào Google Drive
OUTPUT_DIR = "/content/drive/MyDrive/segformer_bdd100k_b1_hd_best"

training_args = TrainingArguments(
    output_dir="./results_b1_hd",
    learning_rate=5e-5,
    num_train_epochs=8,                  # 8 Epochs ảnh 720p là tối ưu nhất
    per_device_train_batch_size=2,       # Batch size 2 cho ảnh 1280x720
    gradient_accumulation_steps=4,       # Tích lũy 4 steps (tương đương batch=8)
    per_device_eval_batch_size=2,
    eval_accumulation_steps=10,
    fp16=True,                           # Tăng tốc GPU T4
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    load_best_model_at_end=True,
    logging_steps=50,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("🚀 Bắt đầu huấn luyện SegFormer-B1 ở độ phân giải 1280x720...")
trainer.train()

# Lưu model hoàn chỉnh về Google Drive
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"🎉 Hoàn tất! Model HD tốt nhất đã được lưu tại: {OUTPUT_DIR}")
```

---

## 3. Sản phẩm đầu ra & Cách sử dụng sau khi train xong

1. **Bảng số liệu mIoU và per-class IoU chính thức:** Xuất hiện ở cuối Epoch 8 -> Chụp ảnh lại để đưa vào báo cáo và slide nghiệm thu đề tài.
2. **File trọng số:** Nằm trong Google Drive tại thư mục `MyDrive/segformer_bdd100k_b1_hd_best` (nặng chỉ ~54MB).
3. **Nạp vào Web SegVision QA:**
   * Tải thư mục `segformer_bdd100k_b1_hd_best` từ Google Drive về máy tính cá nhân.
   * Đặt vào thư mục dự án `d:\AAA_buildphase\segformer_bdd100k_b1_hd_best`.
   * Web sẽ tự động chuyển sang chạy bằng chính mô hình B1 HD do bạn huấn luyện!
