"""Evaluator module for Semantic Segmentation.
Calculates IoU, per-class metrics, and generates visual Error/Difference Maps.
"""
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import io
import base64
from .constants import CLASS_NAMES, COLOR_PALETTE, IGNORE_INDEX


def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Chuyển đổi mảng numpy RGB thành chuỗi Base64 PNG để gửi về Web."""
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_mask_to_id_array(mask_img: Image.Image) -> np.ndarray:
    """
    Nhận diện định dạng của Ground Truth mask:
    Nếu là ảnh 1 kênh (L / P): chứa ID từ 0 đến 18 hoặc 255.
    Nếu là ảnh RGB: ánh xạ màu ngược lại ID (hoặc lấy kênh R).
    """
    if mask_img.mode in ("L", "P"):
        return np.array(mask_img, dtype=np.uint8)
    
    # Nếu là RGB, kiểm tra xem có phải kênh Red chứa ID không (format bitmask của BDD100k)
    arr = np.array(mask_img)
    if len(arr.shape) == 3:
        # Nếu kênh R, G, B bằng nhau (ảnh xám dạng RGB) hoặc kênh R chứa ID (<=19 hoặc 255)
        unique_r = np.unique(arr[:, :, 0])
        if np.all((unique_r <= 19) | (unique_r == 255)):
            return arr[:, :, 0]
        # Trường hợp ảnh mask màu bảng màu Cityscapes
        id_mask = np.full(arr.shape[:2], fill_value=IGNORE_INDEX, dtype=np.uint8)
        for class_id, color in enumerate(COLOR_PALETTE):
            matches = np.all(arr[:, :, :3] == color, axis=-1)
            id_mask[matches] = class_id
        return id_mask

    return np.array(mask_img, dtype=np.uint8)


def evaluate_segmentation(
    pred_mask: np.ndarray, 
    gt_mask: np.ndarray
) -> Dict[str, Any]:
    """
    So sánh ma trận dự đoán (pred_mask) và ma trận nhãn thật (gt_mask).
    Tính toán IoU từng lớp, mIoU toàn bức ảnh, và tạo ra Error Map (Bản đồ phân loại lỗi).
    """
    assert pred_mask.shape == gt_mask.shape, f"Kích thước lệch: pred={pred_mask.shape}, gt={gt_mask.shape}"
    h, w = pred_mask.shape
    total_valid_pixels = int(np.sum(gt_mask != IGNORE_INDEX))

    per_class_results = []
    valid_ious = []
    
    # Tạo bản đồ lỗi (Error Map RGB)
    # Mặc định: Vùng ignore (255) tô màu xám tối [40, 40, 40]
    error_map = np.full((h, w, 3), fill_value=40, dtype=np.uint8)

    # 1. Điểm đúng (True Positive): AI đoán trùng với Ground Truth -> Màu Xanh Lá
    correct_mask = (pred_mask == gt_mask) & (gt_mask != IGNORE_INDEX)
    error_map[correct_mask] = [46, 204, 113]  # Emerald Green

    # 2. Điểm sai (False Prediction): AI đoán khác Ground Truth -> Màu Đỏ
    wrong_mask = (pred_mask != gt_mask) & (gt_mask != IGNORE_INDEX)
    error_map[wrong_mask] = [231, 76, 60]     # Coral Red

    # Tính IoU cho từng lớp trong 19 lớp
    for class_id, class_name in enumerate(CLASS_NAMES):
        gt_class = (gt_mask == class_id)
        pred_class = (pred_mask == class_id) & (gt_mask != IGNORE_INDEX)

        intersection = int(np.sum(gt_class & pred_class))
        union = int(np.sum(gt_class | pred_class))

        gt_count = int(np.sum(gt_class))
        pred_count = int(np.sum(pred_class))

        is_present = (gt_count > 0) or (pred_count > 0)
        
        if union == 0:
            iou = 1.0 if not is_present else 0.0
        else:
            iou = float(intersection / union)

        if gt_count > 0:
            valid_ious.append(iou)

        per_class_results.append({
            "id": class_id,
            "name": class_name,
            "color_hex": f"#{COLOR_PALETTE[class_id][0]:02x}{COLOR_PALETTE[class_id][1]:02x}{COLOR_PALETTE[class_id][2]:02x}",
            "iou": round(iou * 100, 2),
            "gt_pixels": gt_count,
            "pred_pixels": pred_count,
            "gt_percentage": round((gt_count / max(1, total_valid_pixels)) * 100, 2),
            "present_in_gt": gt_count > 0
        })

    # mIoU chỉ tính trên các lớp thực sự xuất hiện trong ảnh Ground Truth
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    overall_accuracy = float(np.sum(correct_mask) / max(1, total_valid_pixels))

    return {
        "mean_iou": round(mean_iou * 100, 2),
        "overall_accuracy": round(overall_accuracy * 100, 2),
        "total_valid_pixels": total_valid_pixels,
        "correct_pixels": int(np.sum(correct_mask)),
        "wrong_pixels": int(np.sum(wrong_mask)),
        "classes": per_class_results,
        "error_map_base64": numpy_to_base64_png(error_map)
    }
