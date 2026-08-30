"""
================================================================================
CACH 1 - BUOC 2: Fine-tune BioViL-T (da hoc DAU HIEU tu set_B) tren set_A
================================================================================
Nap backbone tu Buoc 1 (biovilt_setb_findings_backbone.pth) -> fine-tune Co/Khong
tren set_A voi CrossEntropy (dung y thay).

SO SANH:
  - BioViL-T THUONG + CE      (Finetune CrossEntropy)       -> AUC 0.913
  - BioViL-T + hoc dau hieu set_B + CE  (file nay)          -> ?
  => "hoc dau hieu truoc" co giup phan loai tot hon khong?

KHAC Finetune CrossEntropy DUY NHAT 1 CHO: nguon backbone.
  Finetune CrossEntropy: backbone = BioViL-T goc (biovil_t_...pt)
  File nay:              backbone = BioViL-T da hoc dau hieu (Buoc 1)

CHAY:
  python "Finetune SetB Aux/finetune_setA_findingaware.py" --smoke
  python "Finetune SetB Aux/finetune_setA_findingaware.py"
================================================================================
"""

# 1) Nhập thư viện và thiết lập môi trường ban đầu
import os, re, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import (Compose, Resize, CenterCrop, RandomCrop,
                                     RandomHorizontalFlip, RandomRotation, ToTensor)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 2) Đọc tham số dòng lệnh để chạy thử nhanh hoặc huấn luyện đầy đủ
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--unfreeze-epoch", type=int, default=6)
args = parser.parse_args()

# 3) Cấu hình đường dẫn, tham số huấn luyện và môi trường chạy
EXCEL_PATH    = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune 2\SetA_Labels.xlsx"
IMAGE_DIR     = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune\images_256\train"
OUTPUT_DIR    = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune SetB Aux"
VANILLA_BACKBONE  = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Pre-train BioViL-T\biovil_t_image_model_proj_size_128.pt"
FINDINGS_BACKBONE = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\SetB Findings\biovilt_setb_findings_backbone.pth"
IMAGE_COLUMN  = "file_name"
LABEL_COLUMN  = "bnn"

BATCH_SIZE     = args.batch_size
EPOCHS         = 2 if args.smoke else args.epochs
UNFREEZE_EPOCH = 1 if args.smoke else args.unfreeze_epoch
LR_HEAD, LR_BACKBONE = 1e-4, 1e-5
RESIZE, CROP = 512, 448
PATIENCE = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

# 4) Bộ ghi log để lưu kết quả train vào file text
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, d):
        for x in self.s:
            if getattr(x, "closed", False): continue
            x.write(d); x.flush()
    def flush(self):
        for x in self.s:
            if getattr(x, "closed", False): continue
            x.flush()
_log = open(os.path.join(OUTPUT_DIR, "train_log_smoke.txt" if args.smoke else "train_log.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log)

print("=" * 80)
print(f"CACH 1 - BUOC 2: BioViL-T (hoc dau hieu set_B) + CE tren set_A  {'[SMOKE]' if args.smoke else ''}")
print("=" * 80)
print(f"Backbone (finding-aware): {FINDINGS_BACKBONE}")
print(f"Device={DEVICE} | AMP={USE_AMP} | Batch={BATCH_SIZE} | Epochs={EPOCHS}")

# 5) Tải dữ liệu từ Excel, khớp file ảnh và chia tập train/val/test
print("\n[1] Excel + match + split...")
df = pd.read_excel(EXCEL_PATH)
df[LABEL_COLUMN] = df[LABEL_COLUMN].map({"Khong": 0, "Co": 1})
all_files = {}
for root, _d, files in os.walk(IMAGE_DIR):
    for f in files:
        all_files[f.lower().strip()] = os.path.join(root, f)
df["image_path"] = df[IMAGE_COLUMN].apply(lambda x: all_files.get(str(x).strip().lower()))
df = df[df["image_path"].notna()].reset_index(drop=True)
df["patient_base"] = df[IMAGE_COLUMN].apply(
    lambda x: re.sub(r"_\d+\.(jpg|jpeg|png|bmp|tiff|tif)$", "", str(x).lower().strip()))
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_idx, tmp_idx = next(gss.split(df, df[LABEL_COLUMN], groups=df["patient_base"]))
train_df, temp_df = df.iloc[tr_idx].copy(), df.iloc[tmp_idx].copy()
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
v_idx, te_idx = next(gss2.split(temp_df, temp_df[LABEL_COLUMN], groups=temp_df["patient_base"]))
val_df, test_df = temp_df.iloc[v_idx].copy(), temp_df.iloc[te_idx].copy()
if args.smoke:
    train_df = train_df.sample(n=min(80, len(train_df)), random_state=0).reset_index(drop=True)
    val_df   = val_df.sample(n=min(40, len(val_df)), random_state=0).reset_index(drop=True)
    test_df  = test_df.sample(n=min(40, len(test_df)), random_state=0).reset_index(drop=True)
print(f"  Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
assert len(set(train_df["patient_base"]) & set(test_df["patient_base"])) == 0, "Leakage!"

# 6) Xây dựng transforms và dataset cho ảnh X-ray
from health_multimodal.image.data.transforms import ExpandChannels
train_tf = Compose([Resize(RESIZE), RandomHorizontalFlip(0.5), RandomRotation(10),
                    RandomCrop(CROP), ToTensor(), ExpandChannels()])
eval_tf  = Compose([Resize(RESIZE), CenterCrop(CROP), ToTensor(), ExpandChannels()])

class XrayDataset(Dataset):
    def __init__(self, dframe, transform):
        self.df = dframe.reset_index(drop=True); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        return self.transform(Image.open(row["image_path"]).convert("L")), int(row[LABEL_COLUMN])

counts = train_df[LABEL_COLUMN].value_counts().sort_index()
sample_w = train_df[LABEL_COLUMN].map(lambda l: 1.0/counts[l]).tolist()
sampler = WeightedRandomSampler(sample_w, num_samples=len(train_df)*2, replacement=True)
train_loader = DataLoader(XrayDataset(train_df, train_tf), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True, num_workers=0)
val_loader   = DataLoader(XrayDataset(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(XrayDataset(test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 7) Tạo backbone BioViL-T và mô hình classifier cho bài toán phân loại
print("\n[2] Building BioViL-T (nap backbone finding-aware)...")
from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.types import ImageEncoderType

def build_backbone():
    # dung backbone goc de dung kien truc, roi ghi de bang weight da hoc dau hieu
    bb = ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                    joint_feature_size=128, pretrained_model_path=VANILLA_BACKBONE)
    if os.path.exists(FINDINGS_BACKBONE):
        sd = torch.load(FINDINGS_BACKBONE, map_location="cpu")
        missing, unexpected = bb.load_state_dict(sd, strict=False)
        print(f"  Nap finding-aware backbone OK | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"  ⚠️ KHONG thay {FINDINGS_BACKBONE} -> dung backbone goc (chay Buoc 1 truoc!)")
    return bb

class BioViLTClassifier(nn.Module):
    def __init__(self, backbone, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.feature_size, num_classes))
    def forward(self, x):
        return self.head(self.backbone(x).img_embedding.flatten(1))
    def set_backbone_trainable(self, flag):
        for p in self.backbone.parameters(): p.requires_grad = flag

model = BioViLTClassifier(build_backbone()).to(DEVICE)
print(f"  feature_size = {model.backbone.feature_size}")
model.set_backbone_trainable(False)
for p in model.head.parameters(): p.requires_grad = True

# LOSS = CrossEntropy (dung y thay)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# 8) Huấn luyện mô hình theo 2 giai đoạn: head trước, sau đó mở backbone
print("\n[3] Training...")
print("=" * 80)
best_val = float("inf"); patience_ctr = 0; stage = 1
train_losses, val_losses = [], []
CKPT = os.path.join(OUTPUT_DIR, "best_model_findingaware.pth")

def run_epoch(loader, train=True):
    model.train(train); tot = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train: optimizer.zero_grad()
        with torch.set_grad_enabled(train), torch.autocast("cuda", enabled=USE_AMP):
            loss = criterion(model(x), y)
        if train:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        tot += loss.item()
    return tot / max(1, len(loader))

for epoch in range(EPOCHS):
    if epoch == UNFREEZE_EPOCH and stage == 1:
        print(f"\n[UNFREEZE] Epoch {epoch+1}: mo bang backbone -> Stage 2")
        model.set_backbone_trainable(True)
        optimizer = torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.head.parameters(), "lr": LR_HEAD}])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        stage = 2; patience_ctr = 0
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    train_losses.append(tr); val_losses.append(va)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} [S{stage}] | Train {tr:.4f} | Val {va:.4f}")
    if va < best_val:
        best_val = va; patience_ctr = 0
        torch.save(model.state_dict(), CKPT); print("  -> best saved")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n[EARLY STOP] epoch {epoch+1}"); break
    scheduler.step(va)

# 9) Đánh giá mô hình trên tập test và lưu kết quả
print("\nLoad best + test...")
model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
y_true, y_proba = [], []
with torch.no_grad():
    for x, y in test_loader:
        with torch.autocast("cuda", enabled=USE_AMP):
            p = torch.softmax(model(x.to(DEVICE)), dim=1)
        y_proba.extend(p[:, 1].float().cpu().numpy()); y_true.extend(y.numpy())
y_proba = np.array(y_proba); y_true = np.array(y_true)
roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
fpr, tpr, _ = roc_curve(y_true, y_proba)
_f, _t, _thr = roc_curve(y_true, y_proba)
opt = float(_thr[np.argmax(_t - _f)])
y_pred = (y_proba >= opt).astype(int)
cm = confusion_matrix(y_true, y_pred); tn, fp, fn, tp = cm.ravel()
sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
acc = (tp+tn)/cm.sum()

print("\n" + "=" * 80)
print("TEST (Youden's J) — BioViL-T finding-aware + CE")
print("=" * 80)
print(f"  ROC AUC            : {roc_auc:.4f}   (so voi BioViL-T thuong + CE = 0.913)")
print(f"  Accuracy           : {acc:.4f}")
print(f"  Sensitivity (Co)   : {sens:.3f}")
print(f"  Specificity (Khong): {spec:.3f}")
print(f"  F1 (Co)            : {f1_score(y_true, y_pred, zero_division=0):.4f}")
print("\n" + classification_report(y_true, y_pred, target_names=["Khong (0)", "Co (1)"], zero_division=0))

if not args.smoke:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BioViL-T finding-aware (set_B) + CE — set_A", fontsize=13, fontweight="bold")
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); ax[0].plot([0,1],[0,1],"k--")
    ax[0].set_title("ROC"); ax[0].legend(); ax[0].grid(alpha=0.3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1],
                xticklabels=["Khong","Co"], yticklabels=["Khong","Co"], annot_kws={"size":14})
    ax[1].set_title("Confusion Matrix"); ax[1].set_ylabel("True"); ax[1].set_xlabel("Predicted")
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "evaluation_findingaware.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); print(f"\nSaved -> {png}")
    import csv as _csv
    with open(os.path.join(OUTPUT_DIR, "confusion_matrix_findingaware.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f); w.writerow(["", "pred_Khong", "pred_Co"])
        w.writerow(["true_Khong", int(cm[0,0]), int(cm[0,1])])
        w.writerow(["true_Co",    int(cm[1,0]), int(cm[1,1])])
    print("Saved -> confusion_matrix_findingaware.csv")

print("\n" + "=" * 80)
print(f"DONE — finding-aware + CE | AUC={roc_auc:.4f} | Sens={sens:.3f} | Spec={spec:.3f}")
print("=" * 80)
sys.stdout.flush(); sys.stdout = sys.__stdout__; _log.close()
