"""
================================================================================
TRICH XUAT VECTOR DAC TRUNG ANH (512-d) tu model BioViL-T + CE (silicosis)
================================================================================
Model: best_model_biovilt_silicosis_ce.pth  (BioViL-T + head 512->2, AUC 0.8774)
Chay tren bo anh SDD (= NEW DATA), lay img_embedding (512) TRUOC head.

Bo du lieu goc train o C:\\Users\\ASUS\\Downloads\\archive (may khac); script
dung anh SDD co san tren may nay = "NEW DATA" (test 360 khop chinh xac).

XUAT RA (FinetuneNewData/vectors/):
  - train_vectors.npy (N,512) / test_vectors.npy
  - train_meta.csv / test_meta.csv  (fname, label)   [PII -> khong upload GitHub]
  - vectors_all.npz

CHAY:
  python "FinetuneNewData/export_vectors_silicosis_ce.py"
================================================================================
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- data SDD co san tren may nay ----
DATA_DIR   = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\NEW DATA"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
TEST_DIR   = os.path.join(DATA_DIR, "test")
HERE       = os.path.dirname(os.path.abspath(__file__))
VEC_DIR    = os.path.join(HERE, "vectors")
CKPT       = os.path.join(HERE, "best_model_biovilt_silicosis_ce.pth")
BACKBONE_PATH = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Pre-train BioViL-T\biovil_t_image_model_proj_size_128.pt"
CLASS_TO_LABEL = {"normal": 0, "silicosis": 1}
RESIZE, CROP, BATCH_SIZE = 512, 448, 8
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

os.makedirs(VEC_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

print("=" * 80)
print("TRICH XUAT VECTOR (512-d) — BioViL-T + CE (silicosis)")
print("=" * 80)
print(f"Checkpoint: {CKPT}\nDevice={DEVICE} | AMP={USE_AMP}")

def scan_dir(base):
    items = []
    for cls, lab in CLASS_TO_LABEL.items():
        d = os.path.join(base, cls)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                items.append({"path": os.path.join(d, f), "label": lab, "fname": f})
    return items

train_items = scan_dir(TRAIN_DIR)
test_items  = scan_dir(TEST_DIR)
print(f"\n[1] Scan: train={len(train_items)} | test={len(test_items)}")

from health_multimodal.image.data.transforms import ExpandChannels
eval_tf = Compose([Resize(RESIZE), CenterCrop(CROP), ToTensor(), ExpandChannels()])

class XrayDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return eval_tf(Image.open(it["path"]).convert("L")), int(it["label"])

print("\n[2] Building BioViL-T + load checkpoint...")
from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.types import ImageEncoderType

def build_backbone():
    if os.path.exists(BACKBONE_PATH):
        return ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                          joint_feature_size=128, pretrained_model_path=BACKBONE_PATH)
    from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
    return get_biovil_t_image_encoder()

class BioViLTClassifier(nn.Module):
    def __init__(self, backbone, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.feature_size, num_classes))
    def forward_embedding(self, x):
        return self.backbone(x).img_embedding.flatten(1)
    def forward(self, x):
        return self.head(self.forward_embedding(x))

model = BioViLTClassifier(build_backbone()).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()
print(f"  feature_size = {model.backbone.feature_size} | checkpoint loaded OK")

@torch.no_grad()
def extract(items, split):
    loader = DataLoader(XrayDataset(items), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    vecs, labs = [], []
    for x, y in loader:
        with torch.autocast("cuda", enabled=USE_AMP):
            emb = model.forward_embedding(x.to(DEVICE))
        vecs.append(emb.float().cpu().numpy()); labs.append(y.numpy())
    X = np.concatenate(vecs, 0).astype(np.float32)
    Y = np.concatenate(labs, 0).astype(np.int64)
    print(f"  {split}: X={X.shape} | silicosis={int(Y.sum())}")
    return X, Y

print("\n[3] Extract embeddings...")
X_tr, y_tr = extract(train_items, "train")
X_te, y_te = extract(test_items,  "test")

import csv
def save_meta(path, items, y):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["idx", "fname", "label"])
        for i, (it, lab) in enumerate(zip(items, y)):
            w.writerow([i, it["fname"], int(lab)])

np.save(os.path.join(VEC_DIR, "train_vectors.npy"), X_tr)
np.save(os.path.join(VEC_DIR, "test_vectors.npy"),  X_te)
save_meta(os.path.join(VEC_DIR, "train_meta.csv"), train_items, y_tr)
save_meta(os.path.join(VEC_DIR, "test_meta.csv"),  test_items,  y_te)
np.savez_compressed(os.path.join(VEC_DIR, "vectors_all.npz"),
                    X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                    fnames_train=np.array([it["fname"] for it in train_items]),
                    fnames_test=np.array([it["fname"] for it in test_items]))

print("\n" + "=" * 80)
print("DONE — vector da luu vao:", VEC_DIR)
print(f"  train_vectors.npy {X_tr.shape} | test_vectors.npy {X_te.shape}")
print("CANH BAO PII: fname co ten benh nhan -> KHONG upload GitHub.")
print("=" * 80)
