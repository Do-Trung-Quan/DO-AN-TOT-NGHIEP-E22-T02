"""
Fine-tune ResNet18 (BYOL X-ray pretrained) tren set_A Silicodata: phan loai 4 lop
  silicosis | STB (silico-tuberculosis) | TB (tuberculosis) | normal

Stage 1 - Linear probe: freeze backbone, chi train fc.
Stage 2 - Full finetune: unfreeze het, LR backbone thap + LR head cao.

DU LIEU (da resize 256x256 boi resize_setA.py):
  Finetune/images_256/train/{class}/*.jpg   (2129 anh)
  Finetune/images_256/test/{class}/*.jpg    (915 anh)
Backbone pretrain:
  Pre-train/xray_byol_backbone.pth

OUTPUT:
  Finetune/runs/resnet18_setA_best.pth   : full model + class_to_idx + metrics
  Finetune/runs/backbone_setA.pth        : CHI backbone (de trich dac trung -> DGCNN)
  Finetune/runs/finetune_log.txt         : log
  Finetune/runs/confusion_matrix.csv     : confusion matrix tren test

CACH CHAY (tu thu muc goc project):
  python "Finetune/finetune_setA.py" --smoke                 # test nhanh
  python "Finetune/finetune_setA.py"                         # train day du
"""
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN")
DATA_DIR = ROOT / "Finetune" / "images_256"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
PRETRAINED = ROOT / "Pre-train" / "xray_byol_backbone.pth"
OUT_DIR = ROOT / "Finetune" / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LOG = OUT_DIR / "finetune_log.txt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_line(msg: str):
    print(msg, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------- Transforms ----------------
def build_transforms(img_size: int = 224):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    # Anh input da la 256x256 grayscale-luu-thanh-JPG; loader se convert RGB.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


# ---------------- Model ----------------
def build_model(num_classes: int, pretrained_path: Path):
    model = models.resnet18(weights=None)
    if pretrained_path.exists():
        state = torch.load(pretrained_path, map_location="cpu")
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        log_line(f"[model] load pretrain: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        log_line(f"[model] CANH BAO: khong thay {pretrained_path}, dung init ngau nhien!")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model, freeze: bool):
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = not freeze


def class_weight_tensor(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0  # tranh chia 0
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


# ---------------- Train / Eval ----------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    ys, ps, probs_all = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        ys.extend(y.numpy()); ps.extend(preds); probs_all.extend(probs)
    ys, ps, probs_all = np.array(ys), np.array(ps), np.array(probs_all)
    out = {
        "acc": accuracy_score(ys, ps),
        "f1_macro": f1_score(ys, ps, average="macro", zero_division=0),
        "f1_weighted": f1_score(ys, ps, average="weighted", zero_division=0),
    }
    try:
        out["auc_macro"] = roc_auc_score(
            ys, probs_all, multi_class="ovr", average="macro")
    except ValueError:
        out["auc_macro"] = float("nan")
    return out, ys, ps, probs_all


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total, loss_sum, correct = 0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def run_stage(stage, model, train_loader, val_loader, device, epochs,
              lr_head, lr_backbone, weight_decay, criterion, use_amp, num_classes):
    if lr_backbone is None:  # linear probe
        params = [{"params": model.fc.parameters(), "lr": lr_head}]
    else:
        backbone_params = [p for n, p in model.named_parameters()
                           if not n.startswith("fc.") and p.requires_grad]
        params = [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": model.fc.parameters(), "lr": lr_head},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_auc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val, _, _, _ = evaluate(model, val_loader, device, num_classes)
        scheduler.step()
        log_line(f"[{stage}] ep {ep:02d}/{epochs} | train_loss {tr_loss:.4f} acc {tr_acc:.3f} "
                 f"| val acc {val['acc']:.3f} f1_macro {val['f1_macro']:.3f} "
                 f"auc_macro {val['auc_macro']:.3f}")
        if val["auc_macro"] > best_auc:
            best_auc = val["auc_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc


def save_backbone_only(model, path):
    """Luu chi backbone (bo fc) de dung trich dac trung -> DGCNN."""
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}
    torch.save(state, path)
    log_line(f"[save] backbone (feature extractor) -> {path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs-probe", type=int, default=10)
    ap.add_argument("--epochs-finetune", type=int, default=30)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Ty le train tach ra lam val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 epoch moi stage, test nhanh")
    args = ap.parse_args()

    OUT_LOG.write_text("", encoding="utf-8")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_line(f"[env] device={device}, torch={torch.__version__}")
    log_line(f"[cfg] {json.dumps(vars(args))}")

    if args.smoke:
        args.epochs_probe = 1
        args.epochs_finetune = 1

    train_tf, eval_tf = build_transforms(args.img_size)

    # ImageFolder: class duoc sort alphabet -> STB,TB,normal,silicosis = 0,1,2,3
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)
    class_to_idx = full_train.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    log_line(f"[data] classes = {class_to_idx}")
    log_line(f"[data] train(full)={len(full_train)} test={len(test_ds)}")

    # Tach val tu train (stratified theo nhan)
    all_labels = [s[1] for s in full_train.samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)), test_size=args.val_frac,
        stratify=all_labels, random_state=args.seed)

    train_ds = Subset(full_train, train_idx)
    # val dung eval_tf: tao 1 ImageFolder rieng tro cung folder nhung transform eval
    val_base = datasets.ImageFolder(TRAIN_DIR, transform=eval_tf)
    val_ds = Subset(val_base, val_idx)

    train_labels = [all_labels[i] for i in train_idx]
    log_line(f"[split] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    from collections import Counter
    log_line(f"[split] train dist = {dict(sorted(Counter(train_labels).items()))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes, PRETRAINED).to(device)

    cw = class_weight_tensor(np.array(train_labels), num_classes, device)
    log_line(f"[loss] class weights = {[round(x,3) for x in cw.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=cw)

    use_amp = (device.type == "cuda") and not args.no_amp

    # Stage 1 - linear probe
    freeze_backbone(model, freeze=True)
    log_line("\n=== STAGE 1: linear probe (backbone frozen) ===")
    run_stage("probe", model, train_loader, val_loader, device,
              args.epochs_probe, args.lr_head, None,
              args.weight_decay, criterion, use_amp, num_classes)

    # Stage 2 - full finetune
    freeze_backbone(model, freeze=False)
    log_line("\n=== STAGE 2: full finetune ===")
    run_stage("finetune", model, train_loader, val_loader, device,
              args.epochs_finetune, args.lr_head, args.lr_backbone,
              args.weight_decay, criterion, use_amp, num_classes)

    # Test
    log_line("\n=== FINAL TEST ===")
    metrics, ys, ps, _ = evaluate(model, test_loader, device, num_classes)
    log_line(f"[test] acc={metrics['acc']:.4f} f1_macro={metrics['f1_macro']:.4f} "
             f"f1_weighted={metrics['f1_weighted']:.4f} auc_macro={metrics['auc_macro']:.4f}")
    target_names = [idx_to_class[i] for i in range(num_classes)]
    log_line("[test] classification_report:")
    log_line(classification_report(ys, ps, target_names=target_names, zero_division=0))
    cm = confusion_matrix(ys, ps)
    log_line(f"[test] confusion matrix [rows=true, cols=pred] order={target_names}:")
    log_line(str(cm))

    # Luu confusion matrix CSV
    import csv
    with open(OUT_DIR / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"pred_{n}" for n in target_names])
        for i, row in enumerate(cm):
            w.writerow([f"true_{target_names[i]}"] + list(row))

    # Luu full model + backbone-only
    torch.save({
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "test_metrics": metrics,
        "args": vars(args),
    }, OUT_DIR / "resnet18_setA_best.pth")
    log_line(f"[save] full model -> resnet18_setA_best.pth")
    save_backbone_only(model, OUT_DIR / "backbone_setA.pth")

    log_line("\n[done] Fine-tune hoan tat.")


if __name__ == "__main__":
    main()
