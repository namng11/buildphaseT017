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

# Khóa 1 luồng để tiết kiệm RAM trên container Render Free 512MB
torch.set_num_threads(1)

from .constants import COLOR_PALETTE, CLASS_NAMES, IGNORE_INDEX
from .evaluator import numpy_to_base64_png


class SegFormerService:
    def __init__(self, model_path_or_name: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SegFormerService] Thiet bi: {self.device}")
        
        # Đường dẫn các checkpoint (Hỗ trợ linh hoạt Local, Docker, và Cloud)
        candidate_final_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "segformer_bdd100k_final_best", "model")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segformer_bdd100k_final_best", "model")),
            os.path.abspath(os.path.join(".", "segformer_bdd100k_final_best", "model")),
            "/app/segformer_bdd100k_final_best/model",
        ]
        candidate_v1_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "segformer_bdd100k_best")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segformer_bdd100k_best")),
            os.path.abspath(os.path.join(".", "segformer_bdd100k_best")),
            "/app/segformer_bdd100k_best",
        ]

        resolved_final = next((p for p in candidate_final_paths if os.path.exists(p) and os.path.exists(os.path.join(p, "config.json"))), None)
        resolved_v1 = next((p for p in candidate_v1_paths if os.path.exists(p) and os.path.exists(os.path.join(p, "config.json"))), None)

        if model_path_or_name is None:
            if resolved_final:
                model_path_or_name = resolved_final
                self.model_display_name = "SegFormer-B0 (BDD100K Fine-tuned v2 - 56.77% mIoU)"
            elif resolved_v1:
                model_path_or_name = resolved_v1
                self.model_display_name = "SegFormer-B0 (BDD100K Fine-tuned v1 - 44.22% mIoU)"
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
        - adas_telemetry: Hành lang an toàn & khoảng cách mét thực tế
        - pixel_distribution: Bảng phân bố % diện tích các lớp
        - latency_ms: Thời gian xử lý chi tiết (tiền xử lý, suy luận, hậu xử lý)
        """
        orig_w, orig_h = image.size
        
        # Giới hạn kích thước ảnh nếu quá lớn để không bị tràn 512MB RAM trên Render Free
        max_dim = 1024
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            image = image.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.BILINEAR)
            orig_w, orig_h = image.size

        # 1. Tiền xử lý
        t0 = time.perf_counter()
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        t_pre = (time.perf_counter() - t0) * 1000

        # 2. Suy luận (Inference)
        t1 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits # (1, 19, H_feat, W_feat)
        t_inf = (time.perf_counter() - t1) * 1000

        # 3. Hậu xử lý (Tối ưu RAM: interpolate ở kích thước nhỏ gọn rồi argmax sang uint8)
        t2 = time.perf_counter()
        target_interp_h = min(orig_h, 512)
        target_interp_w = min(orig_w, 512)
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(target_interp_h, target_interp_w),
            mode="bilinear",
            align_corners=False
        )
        pred_small = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        del inputs, outputs, logits, upsampled_logits
        import gc
        gc.collect()

        if (target_interp_w, target_interp_h) != (orig_w, orig_h):
            pred_seg = np.array(Image.fromarray(pred_small).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST))
        else:
            pred_seg = pred_small
        t_post = (time.perf_counter() - t2) * 1000
        
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

        # 4. Trích xuất thông số hình học ADAS (Hành lang lái & khoảng cách mét)
        from .evaluator import compute_adas_telemetry
        adas_info = compute_adas_telemetry(pred_seg, np.array(image))

        t_post = (time.perf_counter() - t2) * 1000

        return {
            "pred_mask": pred_seg,
            "color_mask_base64": numpy_to_base64_png(color_mask),
            "adas_telemetry": adas_info,
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

    def predict_with_model(self, model, processor, image: Image.Image) -> np.ndarray:
        orig_w, orig_h = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
        target_interp_h = min(orig_h, 512)
        target_interp_w = min(orig_w, 512)
        upsampled = torch.nn.functional.interpolate(
            logits, size=(target_interp_h, target_interp_w), mode="bilinear", align_corners=False
        )
        pred_small = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        del inputs, outputs, logits, upsampled
        import gc
        gc.collect()

        if (target_interp_w, target_interp_h) != (orig_w, orig_h):
            res = np.array(Image.fromarray(pred_small).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST))
        else:
            res = pred_small
        return res

    def predict_ab_comparison(self, image: Image.Image) -> Dict[str, Any]:
        """
        Đòn 1: Chạy song song cả model Baseline (hoặc v1) và model Fine-tuned v2
        để so sánh đối chứng A/B trên cùng một bức ảnh.
        """
        orig_w, orig_h = image.size
        
        # 1. Chạy model hiện tại (v2 56.77% của Tài)
        pred_b = self.predict(image)
        mask_b_arr = pred_b["pred_mask"]
        
        # 2. Lazy load model đối chứng (v1 hoặc baseline)
        if not hasattr(self, "_baseline_model") or self._baseline_model is None:
            candidate_v1_paths = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "segformer_bdd100k_best")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segformer_bdd100k_best")),
                os.path.abspath(os.path.join(".", "segformer_bdd100k_best")),
                "/app/segformer_bdd100k_best",
            ]
            v1_path = next((p for p in candidate_v1_paths if os.path.exists(p) and os.path.exists(os.path.join(p, "config.json"))), None)
            baseline_target = v1_path if v1_path else "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
            self._baseline_processor = SegformerImageProcessor.from_pretrained(baseline_target)
            self._baseline_model = SegformerForSemanticSegmentation.from_pretrained(baseline_target).to(self.device)
            self._baseline_model.eval()
            self._baseline_name = "SegFormer-B0 v1 (44.22% mIoU - Trước)" if baseline_target == v1_path else "SegFormer-B0 Baseline (Cityscapes)"

        mask_a_arr = self.predict_with_model(self._baseline_model, self._baseline_processor, image)
        
        # Tạo color mask A
        color_mask_a = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for cid in range(len(COLOR_PALETTE)):
            color_mask_a[mask_a_arr == cid] = COLOR_PALETTE[cid]
            
        return {
            "model_a_name": getattr(self, "_baseline_name", "Mô hình cũ (v1)"),
            "model_a_mask_base64": numpy_to_base64_png(color_mask_a),
            "model_b_name": self.model_display_name,
            "model_b_mask_base64": pred_b["color_mask_base64"],
            "adas_telemetry": pred_b.get("adas_telemetry"),
            "distribution": pred_b["distribution"],
            "latency": pred_b["latency"],
            "improvements": [
                {"class_name": "Rider (Người lái xe)", "before": "0.00%", "after": "44.62%", "diff": "+44.62%"},
                {"class_name": "Motorcycle (Xe máy)", "before": "0.00%", "after": "45.22%", "diff": "+45.22%"},
                {"class_name": "Bicycle (Xe đạp)", "before": "0.00%", "after": "41.58%", "diff": "+41.58%"},
                {"class_name": "Person (Người đi bộ)", "before": "30.20%", "after": "58.67%", "diff": "+28.47%"}
            ],
            "miou_stats": {
                "before": "44.22%",
                "after": "56.77%",
                "delta": "+12.55%"
            }
        }

