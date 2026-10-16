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


def compute_adas_telemetry(pred_mask: np.ndarray, orig_img_arr: np.ndarray) -> Dict[str, Any]:
    """
    Ước lượng khoảng cách đơn giác (Monocular Distance Proxy) và trích xuất hành lang lái an toàn.
    Ánh xạ từ 2D pixel sang tọa độ mét theo nguyên lý Ground-Plane Geometry (chuẩn ADAS Mobileye).
    """
    try:
        import cv2
    except ImportError:
        cv2 = None

    h, w = pred_mask.shape
    vis = orig_img_arr.copy()
    
    # 1. Phủ màu hành lang lái xe (Road corridor) - Dùng màu xanh lá chuẩn #16A34A [22, 163, 74]
    road_mask = (pred_mask == 0).astype(np.uint8)
    road_pixels = int(np.sum(road_mask))
    road_ratio = round((road_pixels / (h * w)) * 100, 1)
    
    vis[road_mask == 1] = (vis[road_mask == 1] * 0.72 + np.array([22, 163, 74]) * 0.28).astype(np.uint8)

    # Nếu môi trường Cloud chưa có OpenCV, trả về telemetry hành lang an toàn không bị crash
    if cv2 is None:
        return {
            "adas_overlay_base64": numpy_to_base64_png(vis),
            "objects": [],
            "object_count": 0,
            "closest_distance_m": None,
            "system_status": "SAFE" if road_ratio > 15 else "CAUTION",
            "status_color": "#16A34A" if road_ratio > 15 else "#F59E0B",
            "road_coverage_pct": road_ratio
        }

    # 2. Phát hiện và đo khoảng cách các vật thể chướng ngại vật
    OBSTACLE_CLASSES = {
        11: ("Person", (220, 38, 38)),      # Red
        12: ("Rider", (245, 158, 11)),      # Amber
        13: ("Car", (37, 99, 235)),         # Blue
        14: ("Truck", (124, 58, 237)),      # Purple
        15: ("Bus", (13, 148, 136)),        # Teal
        17: ("Motorcycle", (234, 88, 12)),  # Orange
        18: ("Bicycle", (16, 185, 129))     # Green
    }

    detected_objects = []
    min_dist = 999.0

    y_horizon = int(h * 0.48)
    h_cam = 1.35  # Độ cao camera kính lái tiêu chuẩn (mét)

    for cid, (cname, default_color) in OBSTACLE_CLASSES.items():
        obj_mask = (pred_mask == cid).astype(np.uint8)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 180:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            y_bottom = y + bh
            x_center = x + bw // 2

            # Ước tính khoảng cách Z mét bằng lượng giác mặt đất phẳng
            dy = max(6, y_bottom - y_horizon)
            dist_m = round((h_cam * (h * 0.90)) / dy, 1)
            dist_m = float(np.clip(dist_m, 1.5, 45.0))

            if dist_m < min_dist:
                min_dist = dist_m

            # Phân loại mức độ cảnh báo (Rõ ràng, không neon)
            if dist_m <= 4.5:
                tag_color = (220, 38, 38)   # Đỏ nguy hiểm
                risk_level = "BRAKE"
            elif dist_m <= 9.0:
                tag_color = (245, 158, 11)  # Vàng chú ý
                risk_level = "WARNING"
            else:
                tag_color = (37, 99, 235)   # Xanh an toàn
                risk_level = "SAFE"

            # Vẽ bounding box mảnh 1.5px
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), tag_color, 2)
            
            # Vẽ tag nhãn tinh gọn
            tag_text = f"{cname} {dist_m:.1f}m | {risk_level}"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x, max(0, y - th - 6)), (x + tw + 6, max(th + 6, y)), tag_color, -1)
            cv2.putText(vis, tag_text, (x + 3, max(th + 2, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            detected_objects.append({
                "class_id": cid,
                "class_name": cname,
                "distance_m": dist_m,
                "risk_level": risk_level,
                "bbox": [x, y, bw, bh]
            })

    # Đánh giá trạng thái chung của xe
    if min_dist <= 4.5:
        system_status = "CRITICAL BRAKE"
        status_color = "#DC2626"
    elif min_dist <= 9.0:
        system_status = "CAUTION"
        status_color = "#F59E0B"
    else:
        system_status = "SAFE"
        status_color = "#16A34A"

    return {
        "adas_overlay_base64": numpy_to_base64_png(vis),
        "objects": detected_objects,
        "object_count": len(detected_objects),
        "closest_distance_m": min_dist if min_dist < 999.0 else None,
        "system_status": system_status,
        "status_color": status_color,
        "road_coverage_pct": road_ratio
    }

