"""Dong goi du lieu de upload len Google Drive RIENG TU cho colab/phase0_p01.ipynb.

Tao trong --out-dir:
  p01_images.zip        chi 1835 anh trong data/train va data/test (KHONG co info.csv, xlsx, thu muc sillicosis)
  p01_weights/          biovil_t_image_model_proj_size_128.pt, best_model_ce.pth, phase3_oof_{control,frozen}.npz
Ten file anh chua ho ten benh nhan: zip khong ma hoa, chi upload len Drive ca nhan, khong chia se, khong dua vao git.

  python -m phase0.make_colab_bundle --out-dir D:/colab_bundle
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from phase0.common import ROOT

WEIGHTS = [
    ROOT / "Pre-train BioViL-T/biovil_t_image_model_proj_size_128.pt",
    ROOT / "Finetune CrossEntropy/best_model_ce.pth",
    ROOT / "fusion/output/phase3_oof_control.npz",
    ROOT / "fusion/output/phase3_oof_frozen.npz",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for sub in ("train", "test") for p in (ROOT / "data" / sub).rglob("*.jpg"))
    assert len(imgs) == 1835, f"ky vong 1835 anh, thay {len(imgs)}"
    zpath = args.out_dir / "p01_images.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:     # jpg da nen, khong nen lai
        for p in imgs:
            z.write(p, p.relative_to(ROOT / "data").as_posix())
    print(f"da ghi {zpath} ({zpath.stat().st_size / 1e9:.2f} GB, {len(imgs)} anh)")

    wdir = args.out_dir / "p01_weights"
    wdir.mkdir(exist_ok=True)
    for w in WEIGHTS:
        if not w.exists():
            raise SystemExit(f"thieu {w}")
        if w.stat().st_size < 2000:
            raise SystemExit(f"{w.name} chi {w.stat().st_size} byte: co the la con tro Git LFS, khong phai trong so that")
        shutil.copy2(w, wdir / w.name)
        print(f"  {w.name}: {w.stat().st_size / 1e6:.1f} MB")
    print("\nUpload p01_images.zip len MyDrive/ va thu muc p01_weights/ len MyDrive/p01_weights/")


if __name__ == "__main__":
    main()
