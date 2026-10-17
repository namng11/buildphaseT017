"""Model Service for SegFormer-B0 Semantic Segmentation.
Supports both ultra-lightweight ONNX Runtime (default for cloud deployment <512MB RAM)
and PyTorch (for GPU / training / local fallback).
"""
import time
import os
import gc
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

# Thử nạp ONNX Runtime (công nghệ suy luận tối ưu RAM <300MB cho Render Free)
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# Thử nạp PyTorch nếu có sẵn (chế độ dự phòng khi chạy local)
try:
    import torch
    torch.set_num_threads(1)
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .constants import COLOR_PALETTE, CLASS_NAMES, IGNORE_INDEX
from .evaluator import numpy_to_base64_png


class SegFormerService:
    def __init__(self, model_path_or_name: Optional[str] = None):
        self.use_onnx = False
        self.device = "CPU"
        
        # Đường dẫn file ONNX tối ưu
        self.onnx_v2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "segformer_b0.onnx"))
        self.onnx_v1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "segformer_b0_v1.onnx"))
        
        # Kiểm tra ưu tiên sử dụng ONNX Runtime để tiết kiệm RAM tối đa (<300MB)
        if HAS_ONNX and os.path.exists(self.onnx_v2_path):
            self.use_onnx = True
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.onnx_session_v2 = ort.InferenceSession(self.onnx_v2_path, opts, providers=['CPUExecutionProvider'])
            self.model_display_name = "SegFormer-B0 (BDD100K Fine-tuned v2 - 56.77% mIoU [ONNX])"
            self.model_name = self.onnx_v2_path
            print(f"[SegFormerService] Khoi tao thanh cong voi ONNX Runtime (Tiet kiem RAM)")
            return

        # Nếu không có ONNX, nạp PyTorch thông thường
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[SegFormerService] ONNX khong kha dung. Chuyen sang PyTorch ({self.device})")
            
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
        else:
            raise RuntimeError("Khong tim thay onnxruntime va torch de chay mo hinh.")

    def load_model(self, path_or_name: str, display_name: Optional[str] = None):
        """Nạp model PyTorch từ Hugging Face hoặc thư mục local."""
        if not HAS_TORCH:
            print("[SegFormerService] Bo qua load_model PyTorch vi moi truong chay ONNX.")
            return
        print(f"[SegFormerService] Dang nap model PyTorch: {path_or_name}...")
        self.processor = SegformerImageProcessor.from_pretrained(path_or_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(path_or_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = path_or_name
        self.use_onnx = False
        if display_name:
            self.model_display_name = display_name
        print(f"[SegFormerService] Nap model thanh cong: {self.model_display_name}")

    def _preprocess_onnx(self, image: Image.Image) -> np.ndarray:
        """Tiền xử lý ảnh chuẩn BDD100K/ImageNet không cần thư viện transformers."""
        resized = image.resize((512, 512), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

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
        
        # Giới hạn kích thước ảnh nếu quá lớn (> 1920px Full HD)
        max_dim = 1920
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            image = image.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.BILINEAR)
            orig_w, orig_h = image.size

        # Chế độ ONNX Runtime (Siêu nhẹ, mượt mà chuẩn xác)
        if self.use_onnx:
            t0 = time.perf_counter()
            pixel_values = self._preprocess_onnx(image)
            t_pre = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            outputs = self.onnx_session_v2.run(['logits'], {'pixel_values': pixel_values})
            logits = outputs[0] # (1, 19, 128, 128)
            t_inf = (time.perf_counter() - t1) * 1000

            # Hậu xử lý mượt mà: Nội suy song tuyến tính (Bilinear) trên 19 kênh xác suất bằng OpenCV C++
            # Giúp viền mask mềm mịn, bo cong tự nhiên y hệt PyTorch gốc mà chỉ mất ~50ms và không tốn RAM
            t2 = time.perf_counter()
            try:
                import cv2
                logits_hwc = np.transpose(logits[0], (1, 2, 0)) # (128, 128, 19)
                logits_full = cv2.resize(logits_hwc, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                pred_seg = np.argmax(logits_full, axis=-1).astype(np.uint8)
                del logits_hwc, logits_full
            except Exception:
                pred_128 = np.argmax(logits, axis=1)[0].astype(np.uint8)
                pred_seg = np.array(Image.fromarray(pred_128).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST))
            
            del pixel_values, outputs, logits
            gc.collect()
            t_post = (time.perf_counter() - t2) * 1000
        else:
            # Chế độ PyTorch Eager (Dành cho Local)
            t0 = time.perf_counter()
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            t_pre = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model(**inputs)
                logits = outputs.logits
            t_inf = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            del inputs, outputs, logits, upsampled_logits
            gc.collect()
            t_post = (time.perf_counter() - t2) * 1000

        # Tạo ảnh mask màu
        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for class_id in range(len(COLOR_PALETTE)):
            color_mask[pred_seg == class_id] = COLOR_PALETTE[class_id]

        # Phân bố pixel
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

        distribution.sort(key=lambda x: x["pixels"], reverse=True)

        from .evaluator import compute_adas_telemetry
        adas_info = compute_adas_telemetry(pred_seg, np.array(image))

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
                "device": "ONNX-CPU" if self.use_onnx else str(self.device).upper()
            }
        }

    def predict_with_model(self, model, processor, image: Image.Image) -> np.ndarray:
        orig_w, orig_h = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        res = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        del inputs, outputs, logits, upsampled
        gc.collect()
        return res

    def predict_ab_comparison(self, image: Image.Image) -> Dict[str, Any]:
        """
        So sánh Đối chứng A/B giữa Model cũ (v1) và Model Fine-tuned mới (v2 56.77%).
        """
        orig_w, orig_h = image.size
        
        # 1. Chạy model hiện tại (v2)
        pred_b = self.predict(image)
        mask_b_arr = pred_b["pred_mask"]
        
        # 2. Chạy model đối chứng (v1)
        if self.use_onnx and os.path.exists(self.onnx_v1_path):
            if not hasattr(self, "onnx_session_v1") or self.onnx_session_v1 is None:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self.onnx_session_v1 = ort.InferenceSession(self.onnx_v1_path, opts, providers=['CPUExecutionProvider'])
            
            pixel_values = self._preprocess_onnx(image)
            out_v1 = self.onnx_session_v1.run(['logits'], {'pixel_values': pixel_values})[0]
            try:
                import cv2
                out_v1_hwc = np.transpose(out_v1[0], (1, 2, 0))
                out_v1_full = cv2.resize(out_v1_hwc, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                mask_a_arr = np.argmax(out_v1_full, axis=-1).astype(np.uint8)
                del out_v1_hwc, out_v1_full
            except Exception:
                pred_128_a = np.argmax(out_v1, axis=1)[0].astype(np.uint8)
                mask_a_arr = np.array(Image.fromarray(pred_128_a).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST))
            del pixel_values, out_v1
            gc.collect()
            model_a_name = "SegFormer-B0 v1 (44.22% mIoU - Trước)"
        elif HAS_TORCH:
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
            model_a_name = self._baseline_name
        else:
            mask_a_arr = mask_b_arr
            model_a_name = "Baseline Model"
        
        # Tạo color mask A
        color_mask_a = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for cid in range(len(COLOR_PALETTE)):
            color_mask_a[mask_a_arr == cid] = COLOR_PALETTE[cid]
            
        return {
            "model_a_name": model_a_name,
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
