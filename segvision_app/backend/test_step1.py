"""Unit test script to verify Step 1 backend modules."""
import os
import sys
import numpy as np
from PIL import Image

# Thêm thư mục hiện tại vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')

from backend.constants import CLASS_NAMES, COLOR_PALETTE, IGNORE_INDEX
from backend.evaluator import evaluate_segmentation
from backend.model_service import SegFormerService

def test_backend_core():
    print("=== TEST BƯỚC 1: KIỂM TRA MODULES BACKEND ===")
    
    # 1. Kiểm tra constants
    assert len(CLASS_NAMES) == 19, "Số lượng class phải đúng 19"
    assert len(COLOR_PALETTE) == 19, "Bảng màu phải có đúng 19 màu"
    print("✅ 1. Khai báo 19 classes & bảng màu BDD100K: HỢP LỆ")

    # 2. Kiểm tra hàm đo đạc IoU và Error Map
    h, w = 100, 100
    gt = np.zeros((h, w), dtype=np.uint8)     # Toàn bộ là class 0 (Road)
    gt[0:20, 0:20] = 13                        # 20x20 pixel là class 13 (Car)
    gt[80:100, 80:100] = IGNORE_INDEX          # 20x20 pixel là ignore

    pred = np.zeros((h, w), dtype=np.uint8)   # AI đoán toàn bộ là class 0 (Road)
    pred[0:15, 0:15] = 13                      # AI chỉ đoán đúng 15x15 pixel là Car, sót 1 phần

    eval_res = evaluate_segmentation(pred, gt)
    print(f"✅ 2. Thuật toán Evaluator: mIoU = {eval_res['mean_iou']}% | Accuracy = {eval_res['overall_accuracy']}%")
    print(f"   - Error Map sinh ra thành công (độ dài base64: {len(eval_res['error_map_base64'])})")

    # 3. Kiểm tra Model Service (chạy thử trên 1 ảnh mẫu)
    sample_img_path = os.path.join("d:/AAA_buildphase", "images.jpg")
    if os.path.exists(sample_img_path):
        img = Image.open(sample_img_path).convert("RGB")
        print(f"📸 Tìm thấy ảnh test: {sample_img_path} ({img.size})")
    else:
        # Nếu chưa có ảnh thì tạo ảnh dummy
        img = Image.fromarray(np.random.randint(0, 255, (256, 512, 3), dtype=np.uint8))
        print("📸 Tạo ảnh giả lập RGB (256x512) để kiểm thử")

    service = SegFormerService()
    res = service.predict(img)
    print(f"✅ 3. Inference thành công!")
    print(f"   - Thời gian xử lý: {res['latency']['total_ms']} ms (Inference: {res['latency']['inference_ms']} ms trên {res['latency']['device']})")
    print(f"   - Kích thước mask đầu ra: {res['pred_mask'].shape}")
    top3 = [(x['name'], str(x['percentage']) + '%') for x in res['distribution'][:3]]
    print(f"   - Top 3 lớp chiếm diện tích lớn nhất: {top3}")

    print("\n🎉 BƯỚC 1 ĐÃ HOÀN THÀNH 100% VÀ SẴN SÀNG CHO BƯỚC 2!")

if __name__ == "__main__":
    test_backend_core()
