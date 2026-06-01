"""Kiem tra backbone hien tai co bi collapse khong.

Collapse = moi anh khac nhau cung ra cung 1 vector dac trung (vo dung).
Healthy = cac vector co std lon, vector cua anh khac nhau thi cosine sim thap.
"""
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import random

ROOT = Path(r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN")
BACKBONE = ROOT / "Pre-train" / "xray_byol_backbone.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = models.resnet18(weights=None)
m.load_state_dict(torch.load(BACKBONE, map_location=device), strict=False)
m.fc = torch.nn.Identity()
m.eval().to(device)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Lay 200 anh ngau nhien tu nhieu folder
all_paths = []
for sub in sorted((ROOT / "Pre-train").glob("images_*/images")):
    all_paths.extend(list(sub.glob("*.png"))[:50])
random.seed(0); random.shuffle(all_paths)
paths = all_paths[:200]

feats = []
with torch.no_grad():
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        f = m(x).flatten(1)
        feats.append(f.cpu())
feats = torch.cat(feats)   # [200, 512]

# Stats
per_dim_std = feats.std(dim=0)         # std qua 200 anh tai moi dim
mean_per_dim_std = per_dim_std.mean().item()
norm_mean = feats.norm(dim=1).mean().item()

# Pairwise cosine similarity
feats_n = feats / feats.norm(dim=1, keepdim=True)
cos_matrix = feats_n @ feats_n.T
# Lay similarity giua cac cap khac nhau (bo diagonal)
mask = ~torch.eye(len(feats), dtype=torch.bool)
mean_cos = cos_matrix[mask].mean().item()
max_cos = cos_matrix[mask].max().item()

print(f"Feature shape:           {tuple(feats.shape)}")
print(f"Mean per-dim std:        {mean_per_dim_std:.4f}")
print(f"Mean L2 norm:            {norm_mean:.2f}")
print(f"Mean pairwise cos sim:   {mean_cos:.4f}")
print(f"Max pairwise cos sim:    {max_cos:.4f}")
print()
print("=== DIAGNOSIS ===")
if mean_per_dim_std < 0.05:
    print("❌ COLLAPSED: std qua thap, moi anh ra cung 1 vector.")
elif mean_cos > 0.95:
    print("⚠️  PARTIAL COLLAPSE: cos sim trung binh > 0.95, anh khac nhau van rat giong.")
elif mean_cos > 0.85:
    print("⚠️  WEAK: feature it phan biet, fine-tune se kho.")
elif mean_cos < 0.5:
    print("✅ HEALTHY: feature da hoc tot, anh khac nhau co vector khac.")
else:
    print(f"🤔 OK NHUNG CHUA TOI UU: cos sim {mean_cos:.3f}, co the dung duoc.")
