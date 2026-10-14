"""FastAPI Server for SegVision QA (Mini-CVAT & Model Evaluation Tool).
Provides REST API endpoints for Semantic Segmentation inference and Ground Truth evaluation.
"""
import io
import os
import sys
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import numpy as np

# Thêm thư mục gốc vào đường dẫn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from backend.constants import CLASS_NAMES, COLOR_PALETTE, HEX_PALETTE, IGNORE_INDEX
from backend.evaluator import evaluate_segmentation, decode_mask_to_id_array, numpy_to_base64_png
from backend.model_service import SegFormerService

app = FastAPI(
    title="SegVision QA - Mini-CVAT for BDD100K",
    description="Cong cu kiem duyet va danh gia mo hinh Semantic Segmentation tren BDD100K",
    version="1.0.0"
)

# Cấu hình CORS để giao diện web gọi API mượt mà
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo AI Model Service (SegFormer-B0)
print("[SegVision] Dang khoi dong AI Model Service...")
model_service = SegFormerService()

# Đường dẫn thư mục giao diện Frontend
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Đường dẫn thư mục demo samples
SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo_samples"))
os.makedirs(SAMPLES_DIR, exist_ok=True)
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

# Kiểm tra xem có tập dữ liệu 100 ảnh mẫu (bdd100k_sample_100) không
DATASET_100_DIR = None
for candidate in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "content", "bdd100k_sample_100")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bdd100k_sample_100")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bdd100k_sample_100")),
]:
    if os.path.exists(os.path.join(candidate, "images")):
        DATASET_100_DIR = candidate
        app.mount("/dataset_100", StaticFiles(directory=candidate), name="dataset_100")
        print(f"[SegVision] Da tim thay tap 100 anh mau tai: {candidate}")
        break


@app.get("/")
async def serve_index():
    """Phục vụ trang chủ ứng dụng Web."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "SegVision QA API is running. Frontend index.html not found yet."}


@app.get("/api/samples")
async def get_demo_samples():
    """Trả về danh sách các ảnh mẫu kèm mask có sẵn trong dự án."""
    # Ưu tiên lấy từ tập 100 ảnh nếu đã giải nén, nếu không dùng 5 ảnh demo có sẵn
    active_dir = DATASET_100_DIR if DATASET_100_DIR else SAMPLES_DIR
    mount_prefix = "/dataset_100" if DATASET_100_DIR else "/samples"

    img_dir = os.path.join(active_dir, "images")
    mask_dir = os.path.join(active_dir, "masks")
    if not os.path.exists(img_dir):
        return {"samples": [], "total": 0}
    
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
    samples = []
    for f in files:
        stem = os.path.splitext(f)[0]
        mask_f = f"{stem}.png"
        has_mask = os.path.exists(os.path.join(mask_dir, mask_f))
        samples.append({
            "name": stem,
            "image_url": f"{mount_prefix}/images/{f}",
            "mask_url": f"{mount_prefix}/masks/{mask_f}" if has_mask else None
        })
    return {"samples": samples, "total": len(samples), "source": "dataset_100" if DATASET_100_DIR else "demo_5"}


@app.get("/api/classes")
async def get_classes():
    """Trả về danh mục 19 classes, mã màu RGB và HEX chuẩn của BDD100k."""
    classes = []
    for i, name in enumerate(CLASS_NAMES):
        classes.append({
            "id": i,
            "name": name,
            "hex": HEX_PALETTE[i],
            "rgb": COLOR_PALETTE[i].tolist()
        })
    return {"classes": classes, "total": len(classes)}


@app.get("/api/model_info")
async def get_model_info():
    """Trả về thông tin model hiện tại đang nạp và các model khả dụng."""
    local_custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "segformer_bdd100k_best"))
    has_fine_tuned = os.path.exists(local_custom_path) and os.path.exists(os.path.join(local_custom_path, "config.json"))
    
    return {
        "current_model": model_service.model_display_name,
        "current_path": model_service.model_name,
        "device": str(model_service.device).upper(),
        "has_fine_tuned": has_fine_tuned
    }


@app.post("/api/switch_model")
async def switch_model(target: str = Form(...)):
    """Chuyển đổi giữa mô hình Fine-tuned của nhóm và Pretrained Baseline."""
    local_custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "segformer_bdd100k_best"))
    
    if target == "fine_tuned":
        if not os.path.exists(local_custom_path):
            raise HTTPException(status_code=404, detail="Không tìm thấy thư mục segformer_bdd100k_best")
        model_service.load_model(local_custom_path, display_name="SegFormer-B0 (BDD100K Fine-tuned)")
    else:
        baseline_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        model_service.load_model(baseline_name, display_name="SegFormer-B0 (Cityscapes Baseline)")

    return {
        "status": "success",
        "current_model": model_service.model_display_name,
        "model_name": model_service.model_name
    }


@app.post("/api/predict")
async def predict_image(image: UploadFile = File(...)):
    """
    Chế độ 1: Dự đoán & Trực quan hóa
    Nhận ảnh đầu vào -> Chạy SegFormer -> Trả về lớp mask màu và phân bố % pixel.
    """
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file ảnh: {str(e)}")

    # Chạy suy luận
    result = model_service.predict(pil_image)
    
    # Chuẩn bị ảnh gốc dạng Base64 để hiển thị trên web
    orig_base64 = numpy_to_base64_png(np.array(pil_image))

    return {
        "status": "success",
        "original_base64": orig_base64,
        "color_mask_base64": result["color_mask_base64"],
        "distribution": result["distribution"],
        "width": result["width"],
        "height": result["height"],
        "latency": result["latency"]
    }


@app.post("/api/evaluate")
async def evaluate_image(
    image: UploadFile = File(...),
    ground_truth: UploadFile = File(...)
):
    """
    Chế độ 2: Kiểm duyệt & Đánh giá Ground Truth (Benchmark/QA Mode)
    Nhận ảnh gốc + ảnh Ground Truth Mask -> Tính toán mIoU, per-class IoU và sinh Error Map.
    """
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        gt_bytes = await ground_truth.read()
        pil_gt = Image.open(io.BytesIO(gt_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi đọc file ảnh hoặc nhãn thật: {str(e)}")

    # 1. Chạy AI dự đoán
    pred_res = model_service.predict(pil_img)
    pred_mask = pred_res["pred_mask"]

    # 2. Xử lý ảnh nhãn Ground Truth (đảm bảo cùng kích thước với ảnh dự đoán)
    if pil_gt.size != pil_img.size:
        pil_gt = pil_gt.resize(pil_img.size, resample=Image.NEAREST)
    
    gt_id_mask = decode_mask_to_id_array(pil_gt)

    # 3. Tính toán ma trận so sánh IoU & sinh Bản đồ lỗi
    eval_res = evaluate_segmentation(pred_mask, gt_id_mask)

    # 4. Tạo ảnh màu trực quan cho Ground Truth
    gt_color = np.zeros((pil_img.height, pil_img.width, 3), dtype=np.uint8)
    for cid in range(len(COLOR_PALETTE)):
        gt_color[gt_id_mask == cid] = COLOR_PALETTE[cid]
    gt_color_base64 = numpy_to_base64_png(gt_color)

    return {
        "status": "success",
        "original_base64": numpy_to_base64_png(np.array(pil_img)),
        "color_mask_base64": pred_res["color_mask_base64"],
        "gt_color_base64": gt_color_base64,
        "error_map_base64": eval_res["error_map_base64"],
        "mean_iou": eval_res["mean_iou"],
        "overall_accuracy": eval_res["overall_accuracy"],
        "classes": eval_res["classes"],
        "distribution": pred_res["distribution"],
        "latency": pred_res["latency"]
    }


if __name__ == "__main__":
    import uvicorn
    print("[SegVision QA] Khoi dong may chu tai http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
