"""
================================================================================
P1 — CROSS-FITTING 5 FOLD  [buoc quan trong nhat]
================================================================================
Sinh vector dac trung OUT-OF-FOLD cho ca 1835 anh: moi anh duoc trich bang mot
model CHUA TUNG NHIN THAY no. Dua nhanh anh len ngang chuan nhanh timeseries.

    StratifiedGroupKFold(5)  groups = patient_group,  y = label_ketqua
    for k in 0..4:
        train tren 4 fold  ->  trich vector cho fold k
    ghep lai -> 1835 vector, moi vector out-of-fold, KHONG ro ri

KHAC BIET BAT BUOC so voi finetune_biovilt_silicosis.py:

  | Hang muc      | Script cu                      | Ban nay                       |
  |---------------|--------------------------------|-------------------------------|
  | Split         | train_test_split ngau nhien     | StratifiedGroupKFold theo BN  |
  | Assert        | tren file_name (vo dung)        | tren patient_group            |
  | Nhan          | ten thu muc normal/silicosis    | label_ketqua tu info.csv      |
  | Checkpoint    | val loss                        | val PR-AUC                    |
  | Trich xuat    | co autocast fp16                | fp32, TAT autocast            |
  | Dau ra        | 512 chieu (256 la rac)          | 256 chieu song + assert       |

VI SAO NHAN `ketqua` CHU KHONG PHAI `bnn`:
  bnn chi co 99 ca duong / 86 benh nhan duong. Chia 5 fold con ~17 benh nhan
  duong moi fold -> khong du de fine-tune ResNet50. ketqua co 462 duong
  (25.2%), hoc duoc. bnn van duoc mang theo trong image_index de danh gia
  probe va cho fusion dung lam nhan dich.

CHAY:
  python imagefeat/crossfit_finetune.py --smoke        # kiem tra nhanh truoc
  python imagefeat/crossfit_finetune.py                # chay that, ~1 buoi GPU
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import (CenterCrop, Compose, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation,
                                    Resize, ToTensor)

# Pipeline tien xu ly chuan BioViL-T: anh xam -> 3 kenh bang repeat,
# KHONG ImageNet normalize. Phai giong het luc fine-tune goc.
from health_multimodal.image.data.transforms import ExpandChannels

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESIZE, CROP = 512, 448
FEATURE_DIM_FULL = 512
# BioViL-T (RESNET50_MULTI_IMAGE) la model temporal: nua sau cua img_embedding la
# dac trung SAI KHAC so voi phim cu. Chi cap 1 anh -> nua sau roi ve hang so.
# Da do tren 1835 anh: dung 256 chieu 256..511 co ptp == 0 trong float64.
FEATURE_DIM_ALIVE = 256


class XrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, label_column: str):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        image = Image.open(row["image_path"]).convert("L")   # ExpandChannels can [1,H,W]
        return (self.transform(image),
                int(row[self.label_column]),
                int(row["label_bnn"]))


class BioViLTCrossFit(nn.Module):
    """Backbone BioViL-T + head chinh (ketqua) + head phu tuy chon (bnn)."""

    def __init__(self, backbone, dropout: float = 0.3, use_aux: bool = False):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(FEATURE_DIM_ALIVE, 2)
        self.head_aux = nn.Linear(FEATURE_DIM_ALIVE, 2) if use_aux else None

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        full = self.backbone(x).img_embedding.flatten(1)
        return full[:, :FEATURE_DIM_ALIVE]          # bo khoi chet ngay tu dau

    def forward(self, x: torch.Tensor):
        feature = self.dropout(self.forward_embedding(x))
        aux = self.head_aux(feature) if self.head_aux is not None else None
        return self.head(feature), aux

    def set_backbone_trainable(self, flag: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = flag


def build_backbone(backbone_path: Path):
    from health_multimodal.image.model.model import ImageModel
    from health_multimodal.image.model.types import ImageEncoderType
    if backbone_path.exists():
        print(f"    backbone LOCAL: {backbone_path}")
        return ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                          joint_feature_size=128, pretrained_model_path=str(backbone_path))
    print("    khong thay file local -> tai tu HuggingFace")
    from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
    return get_biovil_t_image_encoder()


def assert_no_group_overlap(frame: pd.DataFrame, left: np.ndarray, right: np.ndarray,
                            what: str) -> None:
    shared = (set(frame.iloc[left]["patient_group"])
              & set(frame.iloc[right]["patient_group"]))
    assert not shared, f"RO RI BENH NHAN ({what}): {len(shared)} nguoi nam o ca hai tap!"


def run_fold(fold: int, frame: pd.DataFrame, trainval_idx: np.ndarray,
             heldout_idx: np.ndarray, args, device, backbone_path: Path) -> np.ndarray:
    label_column = "label_ketqua"

    inner = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    trainval = frame.iloc[trainval_idx]
    rel_train, rel_val = next(inner.split(
        trainval, trainval[label_column], groups=trainval["patient_group"]))
    train_idx = trainval_idx[rel_train]
    val_idx = trainval_idx[rel_val]

    assert_no_group_overlap(frame, train_idx, val_idx, f"fold{fold} train/val")
    assert_no_group_overlap(frame, train_idx, heldout_idx, f"fold{fold} train/heldout")
    assert_no_group_overlap(frame, val_idx, heldout_idx, f"fold{fold} val/heldout")

    train_df = frame.iloc[train_idx]
    val_df = frame.iloc[val_idx]
    heldout_df = frame.iloc[heldout_idx]

    # Sampler can ca 2 lop co mat trong train (counts[v] se KeyError neu thieu).
    # Val cung phai co ca 2 lop, neu khong PR-AUC luon = 0 -> best_state khong bao
    # gio duoc cap nhat va model am tham giu trong so epoch cuoi. Bao loi ngay.
    if train_df[label_column].nunique() < 2:
        raise SystemExit(f"Fold {fold}: tap train chi co 1 lop. Kiem tra lai phan tang.")
    if val_df[label_column].nunique() < 2:
        raise SystemExit(
            f"Fold {fold}: tap val chi co 1 lop -> khong tinh duoc PR-AUC. "
            f"Giam --val-size hoac doi --seed."
        )

    train_tf = Compose([Resize(RESIZE), RandomHorizontalFlip(p=0.5),
                        RandomRotation(degrees=10), RandomCrop(CROP),
                        ToTensor(), ExpandChannels()])
    eval_tf = Compose([Resize(RESIZE), CenterCrop(CROP), ToTensor(), ExpandChannels()])

    counts = train_df[label_column].value_counts().sort_index()
    sample_weight = train_df[label_column].map(lambda v: 1.0 / counts[v]).tolist()
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(train_df), replacement=True)

    train_loader = DataLoader(XrayDataset(train_df, train_tf, label_column),
                              batch_size=args.batch_size, sampler=sampler,
                              drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(XrayDataset(val_df, eval_tf, label_column),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    heldout_loader = DataLoader(XrayDataset(heldout_df, eval_tf, label_column),
                                batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(f"    train {len(train_df)} | val {len(val_df)} | heldout {len(heldout_df)}"
          f" | duong(train) {int(train_df[label_column].sum())}")

    use_aux = args.aux_weight > 0
    model = BioViLTCrossFit(build_backbone(backbone_path), use_aux=use_aux).to(device)
    model.set_backbone_trainable(False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    best_state = None
    patience = 0
    stage = 1

    for epoch in range(args.epochs):
        if epoch == args.unfreeze_epoch and stage == 1:
            model.set_backbone_trainable(True)
            optimizer = torch.optim.Adam([
                {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": [p for n, p in model.named_parameters()
                            if not n.startswith("backbone.")], "lr": args.lr_head},
            ])
            stage, patience = 2, 0

        model.train()
        for images, labels, labels_aux in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # AMP chi dung khi HUAN LUYEN; luc trich xuat se tat hoan toan.
            with torch.autocast("cuda", enabled=use_amp):
                logits, logits_aux = model(images)
                loss = criterion(logits, labels)
                if logits_aux is not None:
                    loss = loss + args.aux_weight * criterion(logits_aux, labels_aux.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        probabilities, truths = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                with torch.autocast("cuda", enabled=use_amp):
                    logits, _ = model(images.to(device))
                probabilities.extend(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
                truths.extend(labels.numpy())
        # Chon checkpoint theo PR-AUC, KHONG theo val loss: dong chuan nhanh
        # timeseries va dung hon voi du lieu lech.
        score = average_precision_score(truths, probabilities) if len(set(truths)) > 1 else 0.0
        marker = ""
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            marker = " <- best"
        else:
            patience += 1
        print(f"    epoch {epoch + 1:3d}/{args.epochs} [S{stage}] val_pr_auc={score:.4f}{marker}")
        if patience >= args.patience:
            print(f"    [EARLY STOP] epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    best val PR-AUC = {best_score:.4f}")

    # ---- TRICH XUAT: fp32 tuyet doi, khong autocast ----
    model.eval()
    chunks = []
    with torch.no_grad():
        for images, _, _ in heldout_loader:
            chunks.append(model.forward_embedding(images.to(device)).float().cpu().numpy())
    features = np.concatenate(chunks, 0).astype(np.float32)
    assert features.shape == (len(heldout_df), FEATURE_DIM_ALIVE), features.shape
    return features


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=here / "output" / "image_index.parquet")
    parser.add_argument("--map", type=Path,
                        default=here / "output" / "LOCAL_ONLY_filename_map.csv")
    parser.add_argument("--backbone", type=Path,
                        default=repo / "Pre-train BioViL-T" / "biovil_t_image_model_proj_size_128.pt")
    parser.add_argument("--output-dir", type=Path, default=here / "output")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--unfreeze-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-weight", type=float, default=0.0,
                        help="Trong so head phu tren nhan bnn. 0 = tat (mac dinh).")
    parser.add_argument("--smoke", action="store_true",
                        help="Chay nhanh: 2 fold x 2 epoch, it anh.")
    args = parser.parse_args()

    if args.smoke:
        args.folds, args.epochs, args.unfreeze_epoch = 2, 2, 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"P1 — CROSS-FITTING {args.folds} FOLD  {'[SMOKE]' if args.smoke else ''}")
    print("=" * 80)
    print(f"Device={device} | nhan huan luyen=label_ketqua | aux_weight={args.aux_weight}")

    index = pd.read_parquet(args.index)
    mapping = pd.read_csv(args.map)
    frame = index.merge(mapping[["img_id", "image_path"]], on="img_id", how="left")
    if frame["image_path"].isna().any():
        raise SystemExit("Thieu image_path — chay lai build_index.py.")
    frame = frame.sort_values("img_id").reset_index(drop=True)

    if args.smoke:
        # Lay mau theo BENH NHAN (khong theo hang) de khong pha vo nhom, va giu
        # ca 2 lop de sampler/PR-AUC con y nghia.
        rng = np.random.default_rng(args.seed)
        by_patient = frame.groupby("patient_group")["label_ketqua"].max()
        positive = by_patient[by_patient == 1].index.to_numpy()
        negative = by_patient[by_patient == 0].index.to_numpy()
        keep = np.concatenate([
            rng.choice(positive, size=min(40, len(positive)), replace=False),
            rng.choice(negative, size=min(60, len(negative)), replace=False),
        ])
        frame = frame[frame["patient_group"].isin(set(keep))].reset_index(drop=True)
        print(f"[SMOKE] rut gon con {len(frame)} anh / {frame['patient_group'].nunique()} benh nhan")

    print(f"Anh: {len(frame)} | benh nhan: {frame['patient_group'].nunique()} | "
          f"ketqua duong: {int(frame['label_ketqua'].sum())}")

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(frame, frame["label_ketqua"], groups=frame["patient_group"]))

    features = np.full((len(frame), FEATURE_DIM_ALIVE), np.nan, dtype=np.float32)
    fold_id = np.full(len(frame), -1, dtype=np.int8)

    for fold, (trainval_idx, heldout_idx) in enumerate(splits[:args.folds]):
        print(f"\n{'-' * 80}\n[FOLD {fold}] heldout={len(heldout_idx)} anh")
        features[heldout_idx] = run_fold(
            fold, frame, trainval_idx, heldout_idx, args, device, args.backbone)
        fold_id[heldout_idx] = fold

    covered = fold_id >= 0
    print(f"\n{'=' * 80}\nDa trich {int(covered.sum())}/{len(frame)} anh")

    # ---- kiem tra chieu hang so bang FLOAT64 (float32 se bo sot) ----
    valid = features[covered].astype(np.float64)
    spread = valid.max(axis=0) - valid.min(axis=0)
    dead = int((spread == 0).sum())
    print(f"Chieu hang so (float64) = {dead}/{FEATURE_DIM_ALIVE}")
    if not args.smoke:
        assert dead == 0, f"Con {dead} chieu hang so — khong duoc bau giao bo vector nay."

    # Dung concat mot lan thay vi gan 256 cot le -> tranh phan manh DataFrame.
    output = pd.concat([
        pd.DataFrame({
            "img_id": frame["img_id"].to_numpy(),
            "has_image": covered.astype(np.int8),
            "fold_id": fold_id,
        }),
        pd.DataFrame(features, columns=[f"img_feat_{d}" for d in range(FEATURE_DIM_ALIVE)]),
    ], axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "image_features.parquet"
    output.to_parquet(features_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "imagefeat/crossfit_finetune.py",
        "backbone": "BioViL-T ImageModel / RESNET50_MULTI_IMAGE, joint_feature_size=128",
        "backbone_sha256": sha256_file(args.backbone),
        "train_label": "label_ketqua (ketqua == 1) — NHAN PROXY tu doc phim",
        "target_label_carried": "label_bnn — nhan dich, KHONG dung de huan luyen",
        "aux_weight_bnn": args.aux_weight,
        "split_strategy": f"StratifiedGroupKFold({args.folds}) groups=patient_group",
        "checkpoint_selection": "val PR-AUC (average_precision_score)",
        "transform": f"Resize({RESIZE}) -> CenterCrop({CROP}) -> ToTensor -> ExpandChannels",
        "extraction_precision": "float32 (autocast TAT)",
        "feature_dim_raw": FEATURE_DIM_FULL,
        "feature_dim_kept": FEATURE_DIM_ALIVE,
        "dropped_dims": "256..511 — hang so vi BioViL-T temporal chi nhan 1 anh",
        "samples": int(len(frame)),
        "patients": int(frame["patient_group"].nunique()),
        "seed": args.seed,
        "torch": torch.__version__,
        "warning_pii": "Khoa la img_id/file_hash. KHONG chua ten file hay ho ten.",
    }
    manifest_path = args.output_dir / "image_features_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Da ghi -> {features_path}")
    print(f"Da ghi -> {manifest_path}")
    print("\nBUOC TIEP: python imagefeat/qc_report.py")


def sha256_file(path: Path) -> str | None:
    import hashlib
    if not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
