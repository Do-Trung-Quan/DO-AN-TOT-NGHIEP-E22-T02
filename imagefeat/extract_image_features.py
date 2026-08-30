"""
================================================================================
P2 — BO DAC TRUNG DOI CHUNG (control)  [~3 phut GPU, khong train lai]
================================================================================
Trich vector cho ca 1835 anh bang mot checkpoint DA CO SAN tu nhanh SetA
(vd Finetune CrossEntropy/best_model_ce.pth), khong fine-tune gi them.

MUC DICH:
  Cac checkpoint SetA duoc train tren BO DU LIEU SetA voi GroupShuffleSplit
  sach. Bo anh o day ("NEW DATA") la bo KHAC. Neu hai bo khong giao nhau thi
  model chua tung nhin thay bat ky anh nao trong 1835 anh -> ro ri = 0 theo
  dinh nghia, khong can cross-fitting.

  Dung lam DOI CHUNG: neu fusion tren bo cross-fit (P1) va bo doi chung (P2)
  cho ket qua cung chieu, do la bang chung ket qua khong den tu ro ri.

CANH BAO PHAI KIEM TRUOC KHI DUNG:
  Gia dinh "SetA va NEW DATA khong giao nhau" CHUA duoc kiem chung bang so.
  Neu co san danh sach anh cua SetA, chay --seta-index de script doi chieu
  patient_group va bao cao so nguoi trung. Khong co danh sach thi phai ghi ro
  trong bao cao rang day la gia dinh, khong phai su kien.

CHAY:
  python imagefeat/extract_image_features.py
  python imagefeat/extract_image_features.py --checkpoint "Finetune 4/best_model_biovilt_finetuned.pth"
================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from health_multimodal.image.data.transforms import ExpandChannels

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESIZE, CROP = 512, 448
FEATURE_DIM_FULL = 512
FEATURE_DIM_ALIVE = 256


class PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("L"))


class SetAClassifier(nn.Module):
    """Kien truc GIONG HET cac script SetA cu, de load duoc state_dict cua chung.

    head = Sequential(Dropout, Linear(512, 2)) — giu nguyen 512 o day vi
    checkpoint duoc luu voi shape do. Viec cat con 256 chieu dien ra sau,
    luc trich embedding.
    """

    def __init__(self, backbone, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(backbone.feature_size, 2))

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).img_embedding.flatten(1)


def build_backbone(backbone_path: Path):
    from health_multimodal.image.model.model import ImageModel
    from health_multimodal.image.model.types import ImageEncoderType
    if backbone_path.exists():
        return ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                          joint_feature_size=128, pretrained_model_path=str(backbone_path))
    from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
    return get_biovil_t_image_encoder()


def sha256_file(path: Path) -> str | None:
    import hashlib
    if not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=here / "output" / "image_index.parquet")
    parser.add_argument("--map", type=Path,
                        default=here / "output" / "LOCAL_ONLY_filename_map.csv")
    parser.add_argument("--checkpoint", type=Path,
                        default=repo / "Finetune CrossEntropy" / "best_model_ce.pth")
    parser.add_argument("--backbone", type=Path,
                        default=repo / "Pre-train BioViL-T" / "biovil_t_image_model_proj_size_128.pt")
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("P2 — BO DAC TRUNG DOI CHUNG (khong train lai)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}\nDevice={device}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Khong thay checkpoint: {args.checkpoint}")

    index = pd.read_parquet(args.index)
    mapping = pd.read_csv(args.map)
    frame = index.merge(mapping[["img_id", "image_path"]], on="img_id", how="left")
    frame = frame.sort_values("img_id").reset_index(drop=True)
    if frame["image_path"].isna().any():
        raise SystemExit("Thieu image_path — chay lai build_index.py.")
    print(f"Anh: {len(frame)}")

    model = SetAClassifier(build_backbone(args.backbone)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [CANH BAO] missing={len(missing)} unexpected={len(unexpected)} key")
        print("  Neu con so nay lon, checkpoint khong khop kien truc — dung lai kiem tra.")
    model.eval()

    eval_tf = Compose([Resize(RESIZE), CenterCrop(CROP), ToTensor(), ExpandChannels()])
    loader = DataLoader(PathDataset(frame["image_path"], eval_tf),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # fp32 tuyet doi — KHONG autocast. Day la artifact ban giao.
    chunks = []
    with torch.no_grad():
        for images in loader:
            chunks.append(model.forward_embedding(images.to(device)).float().cpu().numpy())
    full = np.concatenate(chunks, 0).astype(np.float32)
    print(f"Da trich: {full.shape}")

    spread_full = full.astype(np.float64).max(0) - full.astype(np.float64).min(0)
    dead_full = int((spread_full == 0).sum())
    print(f"Chieu hang so tren {FEATURE_DIM_FULL} chieu goc (float64) = {dead_full}")

    features = full[:, :FEATURE_DIM_ALIVE]
    spread = features.astype(np.float64).max(0) - features.astype(np.float64).min(0)
    dead = int((spread == 0).sum())
    print(f"Chieu hang so trong {FEATURE_DIM_ALIVE} chieu giu lai = {dead}")
    assert dead == 0, f"Con {dead} chieu hang so trong khoi song."

    output = pd.concat([
        pd.DataFrame({
            "img_id": frame["img_id"].to_numpy(),
            "has_image": np.ones(len(frame), dtype=np.int8),
            "fold_id": np.full(len(frame), -1, dtype=np.int8),   # -1 = khong cross-fit
        }),
        pd.DataFrame(features, columns=[f"img_feat_{d}" for d in range(FEATURE_DIM_ALIVE)]),
    ], axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "image_features_control.parquet"
    output.to_parquet(features_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "imagefeat/extract_image_features.py",
        "role": "BO DOI CHUNG — khong cross-fit, dung de kiem chung ket qua fusion",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "train_label_of_checkpoint": "bnn (nhanh SetA, GroupShuffleSplit seed 42)",
        "leakage_claim": "SetA va NEW DATA gia dinh khong giao nhau — CHUA kiem chung bang so",
        "transform": f"Resize({RESIZE}) -> CenterCrop({CROP}) -> ToTensor -> ExpandChannels",
        "extraction_precision": "float32 (autocast TAT)",
        "feature_dim_raw": FEATURE_DIM_FULL,
        "feature_dim_kept": FEATURE_DIM_ALIVE,
        "dead_dims_in_raw": dead_full,
        "samples": int(len(frame)),
        "torch": torch.__version__,
    }
    manifest_path = args.output_dir / "image_features_control_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da ghi -> {features_path}")
    print(f"Da ghi -> {manifest_path}")


if __name__ == "__main__":
    main()
