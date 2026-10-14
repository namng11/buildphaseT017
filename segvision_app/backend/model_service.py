"""Model Service for SegFormer-B0 Semantic Segmentation.
Handles model loading, inference, overlay generation, and latency profiling.
"""
import time
import os
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from .constants import COLOR_PALETTE, CLASS_NAMES, IGNORE_INDEX
from .evaluator import numpy_to_base64_png


class SegFormerService:
    def __init__(self, model_path_or_name: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SegFormerService] Thiet bi: {self.device}")
        
        # Kiểm tra xem có model fine-tune local không
        local_custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "segformer_bdd100k_best"))
        if model_path_or_name is None:
            if os.path.exists(local_custom_path) and os.path.exists(os.path.join(local_custom_path, "config.json")):
                model_path_or_name = local_custom_path
                self.model_display_name = "SegFormer-B0 (BDD100K Fine-tuned)"
            else:
                model_path_or_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
                self.model_display_name = "SegFormer-B0 (Cityscapes Baseline)"
        else:
            self.model_display_name = "Custom Model"

        self.model_name = model_path_or_name
        self.load_model(model_path_or_name)

    def load_model(self, path_or_name: str, display_name: Optional[str] = None):
        """Nạp model từ Hugging Face hoặc thư mục weights local."""
        print(f"[SegFormerService] Dang nap model: {path_or_name}...")
        self.processor = SegformerImageProcessor.from_pretrained(path_or_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(path_or_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = path_or_name
        if display_name:
            self.model_display_name = display_name
        print(f"[SegFormerService] Nap model thanh cong: {self.model_display_name}")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Thực hiện phân vùng ảnh và trả về:
        - raw_pred: Ma trận 2D chứa class_id từng pixel
        - overlay_base64: Ảnh mask màu theo chuẩn BDD100k
        - pixel_distribution: Bảng phân bố % diện tích các lớp
        - latency_ms: Thời gian xử lý chi tiết (tiền xử lý, suy luận, hậu xử lý)
        """
        orig_w, orig_h = image.size
        
        # 1. Tiền xử lý
        t0 = time.perf_counter()
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        t_pre = (time.perf_counter() - t0) * 1000

        # 2. Suy luận (Inference)
        t1 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits # (1, 19, H_feat, W_feat)
        t_inf = (time.perf_counter() - t1) * 1000

        # 3. Hậu xử lý (Upsample về kích thước ảnh gốc & argmax)
        t2 = time.perf_counter()
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # Tạo ảnh mask màu
        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for class_id in range(len(COLOR_PALETTE)):
            color_mask[pred_seg == class_id] = COLOR_PALETTE[class_id]

        # Tính toán phân bố phần trăm pixel
        total_pixels = orig_w * orig_h
        distribution = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            count = int(np.sum(pred_seg == class_id))
            if count > 0:
                distribution.append({
                    "id": class_id,
                    "name": class_name,
                    "color_hex": f"#{COLOR_PALETTE[class_id][0]:02x}{COLOR_PALETTE[class_id][1]:02x}{COLOR_PALETTE[class_id][2]:02x}",
                    "pixels": count,
                    "percentage": round((count / total_pixels) * 100, 2)
                })

        # Sắp xếp lớp chiếm nhiều diện tích nhất lên đầu
        distribution.sort(key=lambda x: x["pixels"], reverse=True)
        t_post = (time.perf_counter() - t2) * 1000

        return {
            "pred_mask": pred_seg,
            "color_mask_base64": numpy_to_base64_png(color_mask),
            "distribution": distribution,
            "width": orig_w,
            "height": orig_h,
            "latency": {
                "preprocess_ms": round(t_pre, 1),
                "inference_ms": round(t_inf, 1),
                "postprocess_ms": round(t_post, 1),
                "total_ms": round(t_pre + t_inf + t_post, 1),
                "device": str(self.device).upper()
            }
        }
