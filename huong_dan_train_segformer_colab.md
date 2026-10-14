# HƯỚNG DẪN HUẤN LUYỆN SEGFORMER-B0 VỚI BDD100K TRÊN GOOGLE COLAB

Tài liệu này hướng dẫn chi tiết quy trình fine-tune mô hình **SegFormer-B0** trên tập dữ liệu **BDD100K 10K Semantic Segmentation (19 classes)** để lấy chỉ số **mIoU**, **per-class IoU** và lưu file trọng số tốt nhất về Google Drive.

---

## 0. Chuẩn bị trước khi chạy
1. Truy cập [Google Colab](https://colab.research.google.com/) và tạo một Notebook mới.
2. Đổi phần cứng sang GPU: Chọn menu **Runtime** -> **Change runtime type** -> chọn **T4 GPU** -> bấm **Save**.
3. Tải sẵn file `kaggle.json` từ tài khoản Kaggle của bạn về máy tính để chuẩn bị nạp vào Colab.

---

## BƯỚC 1: Kết nối Google Drive & Kiểm tra GPU
Dán vào ô Code thứ 1:
```python
from google.colab import drive
import torch

drive.mount('/content/drive')

if torch.cuda.is_available():
    print("✅ Đã kết nối GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ Chưa bật GPU! Hãy vào Runtime -> Change runtime type -> T4 GPU")
```

---

## BƯỚC 2: Cài đặt thư viện cần thiết
Dán vào ô Code thứ 2:
```bash
!pip install -q transformers datasets evaluate accelerate albumentations
print("✅ Cài đặt thư viện thành công!")
```

---

## BƯỚC 3: Tải và Giải nén Dữ liệu BDD100K từ Kaggle
*(Kéo thả file `kaggle.json` vào khung Files bên trái của Colab trước khi chạy ô này)*

Dán vào ô Code thứ 3:
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Tải bộ dữ liệu BDD100K Semantic Segmentation (10k ảnh)
!kaggle datasets download -d solesensei/solesensei_bdd100k

# Giải nén
!unzip -q solesensei_bdd100k.zip -d /content/bdd100k_data
print("✅ Giải nén dữ liệu hoàn tất!")
```

---

## BƯỚC 4: Khởi tạo Dataset PyTorch (Đọc trực tiếp ảnh và mask PNG)
Dán vào ô Code thứ 4:
```python
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor

# Danh sách 19 nhãn chuẩn của BDD100k (nhãn 255 là ignore)
id2label = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
}
label2id = {v: k for k, v in id2label.items()}

# Chuẩn hóa ảnh về 512x512 để train nhanh và nhẹ trên T4 GPU
processor = SegformerImageProcessor(do_reduce_labels=False, size={"height": 512, "width": 512})

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
        
        # Tìm file mask tương ứng
        mask_path = os.path.join(self.mask_dir, f"{stem}_train_id.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, f"{stem}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        encoded = self.processor(image, mask, return_tensors="pt")
        for k, v in encoded.items():
            encoded[k] = v.squeeze(0)
        return encoded

BASE = "/content/bdd100k_data/bdd100k_seg/bdd100k/seg"
train_dataset = BDD100kDataset(f"{BASE}/images/train", f"{BASE}/labels/train", processor)
val_dataset = BDD100kDataset(f"{BASE}/images/val", f"{BASE}/labels/val", processor)

print(f"✅ Đã nạp thành công: {len(train_dataset)} ảnh Train | {len(val_dataset)} ảnh Val")
```

---

## BƯỚC 5: Khởi tạo mô hình SegFormer-B0
Dán vào ô Code thứ 5:
```python
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=19,
    id2label=id2label,
    label2id=label2id
)
print("✅ Khởi tạo SegFormer-B0 thành công! Tổng tham số:", sum(p.numel() for p in model.parameters()))
```

---

## BƯỚC 6: Hàm đo đạc chỉ số mIoU và per-class IoU
Dán vào ô Code thứ 6:
```python
import evaluate
import numpy as np

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        
        # Upsample về 512x512
        upsampled_logits = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = upsampled_logits.argmax(dim=1).numpy()
        
        # Tính toán mIoU (bỏ qua nhãn 255)
        metrics = metric.compute(
            predictions=preds,
            references=labels,
            num_labels=19,
            ignore_index=255,
            reduce_labels=False
        )
        
        # In bảng Per-class IoU chi tiết
        per_class_iou = metrics["per_category_iou"]
        print("\n" + "="*45)
        print("📊 BẢNG PER-CLASS IOU (19 CLASSES)")
        print("="*45)
        for i, class_name in id2label.items():
            score = per_class_iou[i] * 100 if not np.isnan(per_class_iou[i]) else 0.0
            print(f"Lớp {i:02d} ({class_name:<15}): {score:.2f}%")
        print("="*45)
        print(f"⭐ mIoU Trung Bình: {metrics['mean_iou']*100:.2f}%\n")
            
        return {"mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
```

---

## BƯỚC 7: Huấn luyện & Tự động lưu checkpoint tốt nhất vào Google Drive
Dán vào ô Code thứ 7:
```python
from transformers import TrainingArguments, Trainer

# Thư mục lưu trọng số model trên Google Drive cá nhân
OUTPUT_DIR = "/content/drive/MyDrive/segformer_bdd100k_best"

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=6e-5,
    num_train_epochs=12,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,                          # Bật mixed precision tăng tốc x2
    evaluation_strategy="epoch",         # Đánh giá sau mỗi epoch
    save_strategy="epoch",
    save_total_limit=1,
    metric_for_best_model="mean_iou",    # Giữ bản có mIoU cao nhất
    greater_is_better=True,
    load_best_model_at_end=True,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("🚀 Bắt đầu quá trình huấn luyện...")
trainer.train()

# Lưu model và processor vào Drive
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"🎉 HOÀN TẤT! Model tốt nhất đã được lưu tại: {OUTPUT_DIR}")
```

---

## BƯỚC 8: Trực quan hóa ảnh kết quả (Ảnh gốc vs Ground Truth vs Dự đoán AI)
Dán vào ô Code thứ 8 để chụp ảnh kết quả đưa vào Slide:
```python
import matplotlib.pyplot as plt

sample = val_dataset[0]
image_tensor = sample["pixel_values"].unsqueeze(0).to(model.device)
label_mask = sample["labels"].numpy()

model.eval()
with torch.no_grad():
    outputs = model(pixel_values=image_tensor)
    pred_mask = outputs.logits.argmax(dim=1)[0].cpu().numpy()

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Tái tạo ảnh RGB gốc
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
raw_img = sample["pixel_values"].permute(1, 2, 0).numpy()
raw_img = np.clip((raw_img * std + mean) * 255, 0, 255).astype(np.uint8)

axs[0].imshow(raw_img)
axs[0].set_title("1. Ảnh gốc (Input)")
axs[0].axis("off")

axs[1].imshow(label_mask, cmap="tab20")
axs[1].set_title("2. Ground Truth (Nhãn thật BDD100K)")
axs[1].axis("off")

axs[2].imshow(pred_mask, cmap="tab20")
axs[2].set_title("3. SegFormer Dự đoán (AI Prediction)")
axs[2].axis("off")

plt.tight_layout()
plt.show()
```
