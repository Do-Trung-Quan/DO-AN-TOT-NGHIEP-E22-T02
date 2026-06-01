"""
Pretrain ResNet18 voi BYOL tren NIH ChestX-ray14 (~112k anh, KHONG nhan).

BAT BUOC: chay 'python Pre-train/resize_images.py' TRUOC de pre-resize anh
ve 256x256 vao Pre-train/images_256/. Pretrain nay doc tu do, KHONG doc tu
images_001..images_012 nua (qua cham vi anh goc 1024x1024).

OUTPUT (sau khi chay xong):
  - Pre-train/xray_byol_backbone.pth   : state_dict cua ResNet18 (chi backbone)
  - Pre-train/pretrain_loss.csv        : loss tung epoch
  - Pre-train/pretrain_loss.png        : do thi loss (neu co matplotlib)
  - Pre-train/pretrain_log.txt         : log day du
  - Pre-train/pretrain_config.json     : hyperparameters da dung

CACH CHAY (mo PowerShell tai thu muc goc cua project):
  # Buoc 0: pre-resize (~30 phut, chay 1 lan)
  python "Pre-train/resize_images.py"

  # Buoc 1: test nhanh ~3-5 phut
  python "Pre-train/pretrain_byol.py" --smoke

  # Buoc 2: chay that — uoc ~5-8h cho 30 epoch full data
  python "Pre-train/pretrain_byol.py" --epochs 30 --batch-size 64

SAU KHI XONG:
  - File xray_byol_backbone.pth co the load thang vao torchvision ResNet18:
        m = torchvision.models.resnet18(weights=None)
        m.load_state_dict(torch.load("Pre-train/xray_byol_backbone.pth"), strict=False)
  - finetune_silicosis.py: PRETRAINED = ROOT / "Pre-train" / "xray_byol_backbone.pth"
"""
import os
import csv
import json
import time
import copy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT = Path(r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN")
PRETRAIN_DIR = ROOT / "Pre-train"
IMAGES_DIR = PRETRAIN_DIR / "images_256"   # anh da pre-resize 256x256 (chay resize_images.py truoc)
OUT_BACKBONE = PRETRAIN_DIR / "xray_byol_backbone.pth"
OUT_LOSS_CSV = PRETRAIN_DIR / "pretrain_loss.csv"
OUT_LOSS_PNG = PRETRAIN_DIR / "pretrain_loss.png"
OUT_LOG = PRETRAIN_DIR / "pretrain_log.txt"
OUT_CFG = PRETRAIN_DIR / "pretrain_config.json"


# ============================================================
# 1. DATASET: quet toan bo PNG trong Pre-train/images_*/images/
# ============================================================
class NIHChestXrayUnlabeled(Dataset):
    """Doc anh tu Pre-train/images_256/ (da resize 256x256 san)."""

    def __init__(self, img_dir: Path, transform, max_images: int | None = None):
        if not img_dir.exists():
            raise RuntimeError(
                f"Khong thay thu muc {img_dir}.\n"
                f"  -> Chay 'python Pre-train/resize_images.py' truoc."
            )
        paths = sorted(img_dir.glob("*.png"))
        if not paths:
            raise RuntimeError(
                f"Thu muc {img_dir} rong.\n"
                f"  -> Chay 'python Pre-train/resize_images.py' de resize anh."
            )
        if max_images:
            paths = paths[:max_images]
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        # BYOL: 2 view augment khac nhau cua cung 1 anh
        return self.transform(img), self.transform(img)


def make_transform(img_size: int = 224):
    """Augmentation phu hop cho X-ray (KHAC BYOL chuan):
       - BO RandomGrayscale (vo nghia vi X-ray vo n grayscale)
       - BO ColorJitter saturation/hue (vo nghia sau grayscale)
       - GIU brightness/contrast (mo phong khac biet exposure giua may chup)
       - GIAM GaussianBlur kernel 23->9 (nhanh hon ~5x, hieu qua tuong tu)
       - THEM RandomRotation +-7 do (mo phong tu the chup hoi khac nhau)
       - HEP scale RandomResizedCrop 0.5->0.7 (tranh crop mat het long nguc)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# 2. MODEL: BYOL = online (backbone+proj+pred) + target (EMA copy)
# ============================================================
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(self, feat_dim=512, hidden=1024, proj_dim=256):
        super().__init__()
        # Online network
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # bo fc
        self.projection = MLPHead(feat_dim, hidden, proj_dim)
        self.prediction = MLPHead(proj_dim, hidden, proj_dim)

        # Target network = ban sao online, KHONG co gradient
        self.backbone_t = copy.deepcopy(self.backbone)
        self.projection_t = copy.deepcopy(self.projection)
        for p in self.backbone_t.parameters():
            p.requires_grad = False
        for p in self.projection_t.parameters():
            p.requires_grad = False

    def forward_online(self, x):
        f = self.backbone(x).flatten(1)
        z = self.projection(f)
        return self.prediction(z)

    @torch.no_grad()
    def forward_target(self, x):
        f = self.backbone_t(x).flatten(1)
        return self.projection_t(f)

    @torch.no_grad()
    def update_target(self, m: float):
        """EMA: target = m*target + (1-m)*online"""
        for p_o, p_t in zip(self.backbone.parameters(), self.backbone_t.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)
        for p_o, p_t in zip(self.projection.parameters(), self.projection_t.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)


def byol_loss(p, z):
    """Negative cosine similarity (z duoc detach: target khong gradient)."""
    p = F.normalize(p, dim=-1)
    z = F.normalize(z.detach(), dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


# ============================================================
# 3. UTILITIES
# ============================================================
def log_line(msg: str):
    print(msg, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_backbone(model: BYOL, path: Path):
    """Luu chi backbone dang state_dict tuong thich torchvision ResNet18."""
    full = models.resnet18(weights=None)
    full_keys = [k for k in full.state_dict().keys() if not k.startswith("fc.")]
    bb_state = model.backbone.state_dict()
    bb_keys = list(bb_state.keys())
    assert len(full_keys) == len(bb_keys), \
        f"key mismatch: {len(full_keys)} vs {len(bb_keys)}"
    new_state = {fk: bb_state[bk] for fk, bk in zip(full_keys, bb_keys)}
    torch.save(new_state, path)
    log_line(f"[save] backbone -> {path.name} ({path.stat().st_size/1e6:.1f} MB)")


def cosine_momentum(ep: int, total: int, base: float = 0.996) -> float:
    import math
    return 1 - (1 - base) * (math.cos(math.pi * ep / total) + 1) / 2


# ============================================================
# 4. MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="64 an toan cho RTX 3060 6GB voi AMP")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--momentum-base", type=float, default=0.996)
    ap.add_argument("--max-images", type=int, default=None,
                    help="Gioi han so anh (de test). Mac dinh: dung het ~112k anh.")
    ap.add_argument("--save-every", type=int, default=5,
                    help="Save backbone moi N epoch")
    ap.add_argument("--no-amp", action="store_true",
                    help="Tat mixed precision (cham hon, dung nhieu VRAM hon)")
    ap.add_argument("--smoke", action="store_true",
                    help="Test nhanh: 1 epoch, 500 anh")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 1
        args.max_images = 500

    # Reset log + luu config
    OUT_LOG.write_text("", encoding="utf-8")
    OUT_CFG.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_line(f"[cfg] {json.dumps(vars(args))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_line(f"[env] device={device}, torch={torch.__version__}")
    if device.type == "cuda":
        log_line(f"[env] GPU={torch.cuda.get_device_name(0)}, "
                 f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Data
    tf = make_transform(args.img_size)
    ds = NIHChestXrayUnlabeled(IMAGES_DIR, tf, max_images=args.max_images)
    log_line(f"[data] tong so anh: {len(ds)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=(args.num_workers > 0),
    )

    # Model
    model = BYOL().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_line(f"[model] trainable params = {n_params/1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    log_line(f"[env] AMP = {use_amp}")

    history = []
    t0 = time.time()
    best_loss = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        m = cosine_momentum(ep - 1, args.epochs, args.momentum_base)

        loss_sum, n = 0.0, 0
        ep_start = time.time()
        for step, (v1, v2) in enumerate(loader):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    p1 = model.forward_online(v1)
                    p2 = model.forward_online(v2)
                    z1 = model.forward_target(v1)
                    z2 = model.forward_target(v2)
                    loss = 0.5 * (byol_loss(p1, z2) + byol_loss(p2, z1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                p1 = model.forward_online(v1)
                p2 = model.forward_online(v2)
                z1 = model.forward_target(v1)
                z2 = model.forward_target(v2)
                loss = 0.5 * (byol_loss(p1, z2) + byol_loss(p2, z1))
                loss.backward()
                optimizer.step()

            # EMA update target SAU khi optimizer step
            model.update_target(m)

            loss_sum += loss.item() * v1.size(0)
            n += v1.size(0)

            if step % 50 == 0:
                log_line(f"  [ep {ep} step {step}/{len(loader)}] "
                         f"loss={loss.item():.4f}")

        scheduler.step()
        avg = loss_sum / max(n, 1)
        history.append(avg)
        ep_time = (time.time() - ep_start) / 60
        eta = ep_time * (args.epochs - ep)
        log_line(f"[ep {ep:03d}/{args.epochs}] avg_loss={avg:.4f} "
                 f"momentum={m:.4f} time={ep_time:.1f}min eta={eta:.0f}min")

        # Save backbone
        if ep % args.save_every == 0 or ep == args.epochs or avg < best_loss:
            save_backbone(model, OUT_BACKBONE)
            if avg < best_loss:
                best_loss = avg

        # Luu loss CSV sau moi epoch (de neu sap may van con)
        with open(OUT_LOSS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["epoch", "loss"])
            for i, v in enumerate(history, 1):
                w.writerow([i, v])

    log_line(f"[done] tong thoi gian: {(time.time()-t0)/60:.1f} min, "
             f"best_loss={best_loss:.4f}")

    # Ve do thi
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(history) + 1), history, marker="o", ms=3)
        plt.xlabel("epoch"); plt.ylabel("BYOL loss"); plt.title("Pretrain loss")
        plt.grid(True); plt.tight_layout()
        plt.savefig(OUT_LOSS_PNG, dpi=130)
        log_line(f"[save] loss curve -> {OUT_LOSS_PNG.name}")
    except ImportError:
        log_line("[warn] khong co matplotlib, bo qua ve do thi")

    # Sanity check: forward 1 anh xem co ra vector hop le khong
    model.eval()
    with torch.no_grad():
        sample, _ = ds[0]
        feat = model.backbone(sample.unsqueeze(0).to(device)).flatten(1)
        log_line(f"[sanity] feature shape = {tuple(feat.shape)}, "
                 f"mean={feat.mean().item():.4f}, std={feat.std().item():.4f}")
        if feat.std().item() < 1e-3:
            log_line("[WARN] std rat thap -> co the model bi collapse!")


if __name__ == "__main__":
    main()
