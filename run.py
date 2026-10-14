"""
SegVision QA - Universal Runner & Auto-Setup Script
Compatible with Windows, macOS, Linux.
Designed for Developers, Teammates, and AI Coding Agents (Cursor, Antigravity, Copilot, Cline, etc.)
"""
import os
import sys
import zipfile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "segvision_app")


def print_banner():
    print("=" * 68)
    print("  * SegVision QA - BDD100K Semantic Segmentation (19 Classes)")
    print("  * Universal Auto-Setup & Server Launcher")
    print("=" * 68)


def check_and_extract_zips():
    """Tự động kiểm tra và giải nén model weights và dữ liệu mẫu nếu có file zip."""
    # 1. Model weights
    model_dir = os.path.join(ROOT_DIR, "segformer_bdd100k_best")
    model_zip = os.path.join(ROOT_DIR, "segformer_bdd100k_best.zip")
    
    if not os.path.exists(model_dir) or not os.path.exists(os.path.join(model_dir, "config.json")):
        if os.path.exists(model_zip):
            print(f"[Auto-Setup] Tim thay '{os.path.basename(model_zip)}'. Dang tu dong giai nen...")
            try:
                with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                    zip_ref.extractall(ROOT_DIR)
                print("[Auto-Setup] -> Giai nen model weights thanh cong!")
            except Exception as e:
                print(f"[Auto-Setup] Loi khi giai nen model: {e}")
        else:
            print("[Auto-Setup] Luu y: Chua co thu muc 'segformer_bdd100k_best' hoac file zip.")
            print("             He thong se dung model baseline nvidia/segformer-b0-finetuned-cityscapes-1024-1024.")

    # 2. 100 Sample images dataset
    sample_dir1 = os.path.join(ROOT_DIR, "content", "bdd100k_sample_100")
    sample_dir2 = os.path.join(ROOT_DIR, "bdd100k_sample_100")
    sample_zip = os.path.join(ROOT_DIR, "bdd100k_sample_100.zip")
    
    has_sample = (
        (os.path.exists(sample_dir1) and os.path.exists(os.path.join(sample_dir1, "images"))) or
        (os.path.exists(sample_dir2) and os.path.exists(os.path.join(sample_dir2, "images")))
    )
    if not has_sample and os.path.exists(sample_zip):
        print(f"[Auto-Setup] Tim thay '{os.path.basename(sample_zip)}'. Dang tu dong giai nen vao ./content/...")
        try:
            os.makedirs(os.path.join(ROOT_DIR, "content"), exist_ok=True)
            with zipfile.ZipFile(sample_zip, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(ROOT_DIR, "content"))
            print("[Auto-Setup] -> Giai nen 100 anh mau thanh cong!")
        except Exception as e:
            print(f"[Auto-Setup] Loi khi giai nen 100 anh mau: {e}")


def check_dependencies():
    """Kiểm tra các thư viện Python cần thiết."""
    required = ["fastapi", "uvicorn", "torch", "transformers", "PIL", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[Canh bao] May ban chua cai dat day du cac thu vien: {missing}")
        print("[Huong dan] Vui long chay lenh sau truoc khi khoi dong:")
        print("            pip install -r requirements.txt\n")
        return False
    return True


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print_banner()
    check_and_extract_zips()

    # Đảm bảo đường dẫn segvision_app nằm trong sys.path
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    if not check_dependencies():
        sys.exit(1)

    print("\n[SegVision QA] May chu Web Backend dang khoi dong...")
    print("[URL Truy cap] -> http://127.0.0.1:8000")
    print("[Huong dan]    -> Mo trinh duyet truy cap dia chi tren de su dung.")
    print("[Thoat]        -> Nhan Ctrl + C trong Terminal nay de tat may chu.\n")

    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=APP_DIR)


if __name__ == "__main__":
    main()
