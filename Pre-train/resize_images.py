"""
Resize toan bo anh NIH ChestX-ray14 tu 1024x1024 ve 256x256.

LY DO: Doc 112k file PNG 1024x1024 moi epoch = bottleneck I/O.
       Resize 1 lan, luu lai, train cac epoch sau doc file nho hon 16 lan.

INPUT:   Pre-train/images_*/images/*.png  (1024x1024, ~200-300KB/file)
OUTPUT:  Pre-train/images_256/*.png        (256x256, ~30-50KB/file)

THOI GIAN: ~20-40 phut tren CPU 8 core, chay song song.
DUNG LUONG THEM: ~5-6 GB (anh goc van giu, khong xoa).

IDEMPOTENT: chay nhieu lan se chi resize file moi (skip cai da co).
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from PIL import Image

ROOT = Path(r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN")
PRETRAIN_DIR = ROOT / "Pre-train"
OUT_DIR = PRETRAIN_DIR / "images_256"
TARGET_SIZE = 256


def resize_one(args):
    src, dst = args
    try:
        if dst.exists():
            return True, src.name, "skip"
        img = Image.open(src).convert("L")  # X-ray -> 1 kenh xam
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img.save(dst, "PNG", optimize=True)
        return True, src.name, "done"
    except Exception as e:
        return False, src.name, str(e)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Quet tat ca anh goc
    src_paths = []
    for sub in sorted(PRETRAIN_DIR.glob("images_*/images")):
        # bo qua thu muc images_256 neu vo tinh trung pattern
        if sub.parent.name == "images_256":
            continue
        src_paths.extend(sorted(sub.glob("*.png")))

    if not src_paths:
        print(f"Khong tim thay anh goc trong {PRETRAIN_DIR}/images_*/images")
        sys.exit(1)

    # Tao mapping src -> dst (flat folder, dung file name lam unique key)
    pairs = [(p, OUT_DIR / p.name) for p in src_paths]
    print(f"Tong so anh can xu ly: {len(pairs)}")
    print(f"Output: {OUT_DIR}")

    # Dem so anh da resize
    done_existing = sum(1 for _, d in pairs if d.exists())
    if done_existing == len(pairs):
        print(f"Tat ca {len(pairs)} anh da co san, khong can resize lai.")
        return
    print(f"Da co {done_existing}/{len(pairs)} anh, can resize them {len(pairs)-done_existing} anh.")

    t0 = time.time()
    n_done, n_skip, n_err = 0, 0, 0
    last_report = t0

    # Chay song song
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(resize_one, p) for p in pairs]
        for i, fut in enumerate(as_completed(futures), 1):
            ok, name, status = fut.result()
            if not ok:
                n_err += 1
                if n_err <= 5:
                    print(f"  LOI: {name}: {status}")
            elif status == "skip":
                n_skip += 1
            else:
                n_done += 1

            # Bao tien do moi 5 giay
            now = time.time()
            if now - last_report > 5:
                rate = (n_done + n_skip) / (now - t0)
                eta = (len(pairs) - i) / max(rate, 0.1)
                print(f"  [{i}/{len(pairs)}] done={n_done} skip={n_skip} "
                      f"err={n_err} rate={rate:.0f}/s eta={eta/60:.1f}min")
                last_report = now

    total_time = (time.time() - t0) / 60
    print(f"\nHOAN THANH sau {total_time:.1f} phut.")
    print(f"  Resized: {n_done}")
    print(f"  Skipped: {n_skip}")
    print(f"  Errors:  {n_err}")
    print(f"  Output:  {OUT_DIR}")

    # Kiem tra dung luong
    total_size = sum(p.stat().st_size for p in OUT_DIR.glob("*.png"))
    print(f"  Tong dung luong: {total_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
