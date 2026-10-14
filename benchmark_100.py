"""
SegVision QA - Automated Benchmark on 100 Test Samples
Evaluates fine-tuned SegFormer-B0 on BDD100K 100 sample pairs (images + masks).
Outputs Dataset mIoU, per-class IoU, latency stats, and exports CSV.
"""
import os
import sys
import time
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "segvision_app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, ROOT_DIR)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from backend.constants import CLASS_NAMES, IGNORE_INDEX
from backend.evaluator import decode_mask_to_id_array
from backend.model_service import SegFormerService


def find_sample_dir():
    candidates = [
        os.path.join(ROOT_DIR, "content", "bdd100k_sample_100"),
        os.path.join(ROOT_DIR, "bdd100k_sample_100"),
        os.path.join(APP_DIR, "demo_samples")
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "images")) and os.path.exists(os.path.join(c, "masks")):
            return c
    return None


def run_benchmark():
    print("=" * 70)
    print("  * SEGVISION QA - BENCHMARK 100 SAMPLES (BDD100K 19 CLASSES)")
    print("=" * 70)

    sample_dir = find_sample_dir()
    if not sample_dir:
        print("[Loi] Khong tim thay thu muc chua anh va mask mau!")
        print("      Hay giai nen 'bdd100k_sample_100.zip' vao thu muc content/.")
        return

    img_dir = os.path.join(sample_dir, "images")
    mask_dir = os.path.join(sample_dir, "masks")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
    print(f"[Dataset] Thu muc: {sample_dir}")
    print(f"[Dataset] Tong so anh tim thay: {len(img_files)}")
    print("-" * 70)

    print("[Model] Dang khoi tao mo hinh SegFormer-B0...")
    service = SegFormerService()
    print(f"[Model] Dang su dung: {service.model_display_name}")
    print("-" * 70)

    num_classes = len(CLASS_NAMES)
    total_inter = np.zeros(num_classes, dtype=np.int64)
    total_union = np.zeros(num_classes, dtype=np.int64)
    total_correct = 0
    total_pixels = 0
    latencies = []

    print("[Tien trinh] Dang danh gia tung anh...")
    t_start = time.perf_counter()

    for i, fname in enumerate(img_files, 1):
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        if not os.path.exists(mask_path):
            continue

        img_path = os.path.join(img_dir, fname)
        pil_img = Image.open(img_path).convert("RGB")
        pil_gt = Image.open(mask_path)
        if pil_gt.size != pil_img.size:
            pil_gt = pil_gt.resize(pil_img.size, resample=Image.NEAREST)

        # Suy luan
        pred_res = service.predict(pil_img)
        pred_mask = pred_res["pred_mask"]
        gt_mask = decode_mask_to_id_array(pil_gt)

        latencies.append(pred_res["latency"]["total_ms"])

        # Tinh giao / hop tung lop
        valid_mask = (gt_mask != IGNORE_INDEX)
        total_correct += int(np.sum((pred_mask == gt_mask) & valid_mask))
        total_pixels += int(np.sum(valid_mask))

        for c in range(num_classes):
            p_c = (pred_mask == c) & valid_mask
            g_c = (gt_mask == c) & valid_mask
            inter = int(np.sum(p_c & g_c))
            union = int(np.sum(p_c | g_c))
            total_inter[c] += inter
            total_union[c] += union

        if i % 10 == 0 or i == len(img_files):
            print(f"  -> Da xu ly: {i:3d}/{len(img_files)} anh ({latencies[-1]:.1f} ms/anh)")

    t_total = time.perf_counter() - t_start
    print("-" * 70)

    # Tinh toan ket qua tong hop
    class_ious = []
    for c in range(num_classes):
        if total_union[c] > 0:
            iou = (total_inter[c] / total_union[c]) * 100
            class_ious.append((c, CLASS_NAMES[c], round(iou, 2), int(total_union[c])))
        else:
            class_ious.append((c, CLASS_NAMES[c], None, 0))

    valid_ious = [item[2] for item in class_ious if item[2] is not None]
    overall_miou = round(sum(valid_ious) / len(valid_ious), 2) if valid_ious else 0.0
    overall_acc = round((total_correct / total_pixels) * 100, 2) if total_pixels > 0 else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    # In bang ket qua
    print(f"\n{'ID':<4} | {'Ten Lop (Class)':<18} | {'IoU (%)':<10} | {'Dien tich (Pixels)':<18}")
    print("-" * 62)
    for cid, cname, iou, union_px in class_ious:
        iou_str = f"{iou:.2f}%" if iou is not None else "N/A (Khong co)"
        print(f"{cid:<4} | {cname:<18} | {iou_str:<10} | {union_px:<18,}")
    print("-" * 62)
    print(f"-> mIoU TOAN TAP DU LIEU (Mean IoU):    {overall_miou:.2f}%")
    print(f"-> TI LE CHINH XAC PIXEL (Accuracy):     {overall_acc:.2f}%")
    print(f"-> TOC DO XU LY TRUNG BINH:             {avg_latency:.1f} ms/anh")
    print(f"-> TONG THOI GIAN CHAY:                 {t_total:.1f} giay")
    print("=" * 70)

    # Xuat file CSV
    out_csv = os.path.join(ROOT_DIR, "benchmark_100_results.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Class_ID,Class_Name,IoU_Percent,Union_Pixels\n")
        for cid, cname, iou, union_px in class_ious:
            iou_val = iou if iou is not None else "N/A"
            f.write(f"{cid},{cname},{iou_val},{union_px}\n")
        f.write(f"\nMean_IoU,{overall_miou}%\n")
        f.write(f"Overall_Accuracy,{overall_acc}%\n")
        f.write(f"Avg_Latency_ms,{avg_latency}\n")

    print(f"[Xuat file] Da luu bao cao ket qua chi tiet vao: {out_csv}\n")


if __name__ == "__main__":
    run_benchmark()
