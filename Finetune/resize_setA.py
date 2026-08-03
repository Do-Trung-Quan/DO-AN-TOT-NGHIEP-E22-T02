"""
Resize set_A (Silicodata) tu 1024x1024 ve 256x256, giu nguyen cau truc
train/test + folder theo class de dung torchvision.ImageFolder.

INPUT:
  Silicodata_Updated_Feb2025/set_A_folder/train_images/{class}/*.jpg
  Silicodata_Updated_Feb2025/set_A_folder/test_images/{class}/*.jpg

OUTPUT:
  Finetune/images_256/train/{class}/*.jpg
  Finetune/images_256/test/{class}/*.jpg

Trong do {class} = silicosis | STB | TB | normal

CACH CHAY (tu thu muc goc project):
  python "Finetune/resize_setA.py"

THOI GIAN: ~1-2 phut cho 3044 anh.
IDEMPOTENT: chay lai chi resize file moi (skip cai da co).

LUU Y: convert ve grayscale ("L") roi luu, GIONG het pretrain. Khi train,
dataset se .convert("RGB") de khop dau vao ResNet18 (3 kenh).
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

# Console Windows mac dinh cp1252 khong in duoc ky tu tieng Viet trong path -> ep UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN")
SRC_BASE = ROOT / "Silicodata_Updated_Feb2025" / "set_A_folder"
OUT_BASE = ROOT / "Finetune" / "images_256"
TARGET_SIZE = 256

# Mapping: ten folder goc -> ten folder output (rut gon)
CLASS_MAP = {
    "folder_silicosis": "silicosis",
    "folder_STB": "STB",
    "folder_TB": "TB",
    "folder_normal": "normal",
}
SPLIT_MAP = {
    "train_images": "train",
    "test_images": "test",
}


def resize_one(args):
    src, dst = args
    try:
        if dst.exists():
            return "skip", src.name
        img = Image.open(src).convert("L")  # X-ray -> grayscale (giong pretrain)
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img.save(dst, "JPEG", quality=95)
        return "done", src.name
    except Exception as e:
        return "err", f"{src.name}: {e}"


def main():
    if not SRC_BASE.exists():
        print(f"Khong tim thay {SRC_BASE}")
        sys.exit(1)

    # Thu thap tat ca (src, dst) pairs
    pairs = []
    for src_split, out_split in SPLIT_MAP.items():
        for src_cls, out_cls in CLASS_MAP.items():
            src_dir = SRC_BASE / src_split / src_cls
            out_dir = OUT_BASE / out_split / out_cls
            if not src_dir.exists():
                print(f"  [warn] thieu folder: {src_dir}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in src_dir.glob("*.jpg"):
                pairs.append((img_path, out_dir / img_path.name))

    if not pairs:
        print("Khong tim thay anh nao!")
        sys.exit(1)

    print(f"Tong so anh can xu ly: {len(pairs)}")
    print(f"Output: {OUT_BASE}")

    # Thong ke phan bo
    print("\nPhan bo (sau khi resize):")
    from collections import Counter
    dist = Counter((p[1].parent.parent.name, p[1].parent.name) for p in pairs)
    for (split, cls), n in sorted(dist.items()):
        print(f"  {split}/{cls}: {n}")

    t0 = time.time()
    n_done, n_skip, n_err = 0, 0, 0
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(resize_one, p) for p in pairs]
        for fut in as_completed(futures):
            status, info = fut.result()
            if status == "done":
                n_done += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                if n_err <= 5:
                    print(f"  LOI: {info}")

    dt = time.time() - t0
    print(f"\nHOAN THANH sau {dt:.1f}s")
    print(f"  Resized: {n_done} | Skipped: {n_skip} | Errors: {n_err}")

    total_size = sum(p.stat().st_size for p in OUT_BASE.rglob("*.jpg"))
    print(f"  Tong dung luong output: {total_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
