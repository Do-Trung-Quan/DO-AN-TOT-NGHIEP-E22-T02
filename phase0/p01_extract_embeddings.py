"""P0.1 — trich lai embedding BioViL-T tren anh da xu ly (PG_update ban 3, muc 2).

CHAY TREN COLAB GPU, sau phase0/p01_preprocess.py va QA (phase0/p01_qa_summary.py dat cong).
Dung lai SetAClassifier/build_backbone cua imagefeat/extract_image_features.py de embedding so sanh
duoc voi embedding cu:
  - cung backbone (frozen) hoac cung checkpoint SetA (control)
  - cung 256 chieu dau cua vector 512 chieu
  - fp32, khong autocast
Chi doi dau vao (anh 448x448 da dung kich thuoc, khong Resize/CenterCrop them):
  V1 = cat theo hop bao phoi + che chu;  V2 = V1 + xoa pixel ngoai mask phoi;
  V0 = toan anh da che chu (chi lam vector du phong cho anh cham "hong").

Ra: image_features_{mode}_{variant}.parquet (img_id + img_feat_0..255) — doc duoc bang
phase0/p01_batch_noise.py --features.

  python -m phase0.p01_extract_embeddings --processed /content/p01_processed/p01_processed.npz --mode frozen \
      --variant V1 --out-dir /content/p01_processed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "imagefeat"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True, help="p01_processed.npz tu p01_preprocess")
    ap.add_argument("--mode", choices=["control", "frozen"], required=True)
    ap.add_argument("--variant", choices=["V0", "V1", "V2"], required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "Finetune CrossEntropy" / "best_model_ce.pth")
    ap.add_argument("--backbone", type=Path,
                    default=ROOT / "Pre-train BioViL-T" / "biovil_t_image_model_proj_size_128.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision.transforms import Compose, ToTensor
    from health_multimodal.image.data.transforms import ExpandChannels
    from extract_image_features import FEATURE_DIM_ALIVE, SetAClassifier, build_backbone, sha256_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(args.processed, allow_pickle=False)
    ids, imgs = z["img_id"], z[args.variant]
    print(f"P0.1 buoc 5 — embedding {args.mode} / {args.variant} | {len(ids)} anh {imgs.shape[1:]} | {device}")

    model = SetAClassifier(build_backbone(args.backbone)).to(device)
    if args.mode == "control":
        if not args.checkpoint.exists():
            raise SystemExit(f"Khong thay checkpoint {args.checkpoint}")
        missing, unexpected = model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
        if missing or unexpected:
            print(f"  [CANH BAO] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    tf = Compose([ToTensor(), ExpandChannels()])   # anh da dung kich thuoc: khong Resize/CenterCrop them
    chunks = []
    with torch.no_grad():
        for s in range(0, len(ids), args.batch_size):
            batch = torch.stack([tf(Image.fromarray(a)) for a in imgs[s:s + args.batch_size]])
            chunks.append(model.forward_embedding(batch.to(device)).float().cpu().numpy())
    full = np.concatenate(chunks).astype(np.float32)
    feats = full[:, :FEATURE_DIM_ALIVE]
    spread = feats.astype(np.float64).max(0) - feats.astype(np.float64).min(0)
    dead = int((spread == 0).sum())
    print(f"da trich {full.shape}; chieu hang so trong {FEATURE_DIM_ALIVE} chieu giu lai = {dead}")
    if dead:
        print("  [CANH BAO] co chieu hang so — van giu 256 chieu dau de so sanh duoc voi embedding cu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.DataFrame({"img_id": ids, "has_image": np.ones(len(ids), np.int8),
                                  "fold_id": np.full(len(ids), -1, np.int8)}),
                    pd.DataFrame(feats, columns=[f"img_feat_{d}" for d in range(FEATURE_DIM_ALIVE)])], axis=1)
    out = args.out_dir / f"image_features_{args.mode}_{args.variant}.parquet"
    df.to_parquet(out, index=False)
    (args.out_dir / f"image_features_{args.mode}_{args.variant}_manifest.json").write_text(json.dumps({
        "mode": args.mode, "variant": args.variant, "samples": int(len(ids)),
        "transform": {"V1": "cat theo hop bao mask phoi (torchxrayvision PSPNet, giu 2 thanh phan) + le 5%, "
                            "dem vuong, resize 448, OCR detect + to den hop chu",
                      "V2": "V1 + xoa pixel ngoai mask phoi",
                      "V0": "toan anh dem vuong, OCR detect + to den hop chu, resize 448"}[args.variant]
                     + "; ToTensor, ExpandChannels",
        "processed_config_hash": str(z["config_hash"]) if "config_hash" in z.files else None,
        "checkpoint_sha256": None if args.mode == "frozen" else sha256_file(args.checkpoint),
        "backbone_sha256": sha256_file(args.backbone), "feature_dim_kept": FEATURE_DIM_ALIVE,
        "dead_dims_in_kept": dead, "precision": "float32", "torch": torch.__version__}, indent=2), encoding="utf-8")
    print("da ghi", out)


if __name__ == "__main__":
    main()
