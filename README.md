# DO-AN-TOT-NGHIEP-E22-T02

**Thesis topic:** A graph-based multimodal learning model for occupational lung disease diagnosis, combining chest X-ray images and clinical pulmonary function data.

---

## Branch `Chest-X-ray` — Image Branch

This branch contains the **image feature extraction** pipeline: pretrain a backbone on a large chest X-ray corpus, then fine-tune it for silicosis (occupational dust lung disease) detection. The resulting backbone produces image feature vectors that are later fed into a graph model (DGCNN) together with clinical data.

### Pipeline

```
Pretrain backbone (NIH ChestX-ray14)
        │
        ├── BYOL (self-supervised)   → ResNet18
        └── CheXNet (supervised)     → DenseNet121
        │
        ▼
Fine-tune binary classification (Co / Khong = silicosis / no-silicosis) on Silicodata set_A
        │
        ▼
Feature-extractor backbone → (next step) DGCNN + clinical data
```

Binary labels: **Co** = silicosis + silico-tuberculosis (STB); **Khong** = tuberculosis (TB) + normal.

---

## Folder structure

| Folder | Content |
|---|---|
| `Pre-train/` | BYOL self-supervised pretraining on NIH ChestX-ray14. Includes training/resize/collapse-check scripts, log, loss curve, the `xray_byol_backbone.pth` backbone, and `FINETUNE_GUIDE.md`. |
| `Pre-train CheXNET/` | CheXNet (DenseNet121) pretrained weights — **obtained from arnoweng/CheXNet** (see Credits). |
| `Finetune 2/` | Fine-tuning the **BYOL-ResNet18** backbone for binary classification. |
| `Finetune 3/` | Fine-tuning the **CheXNet-DenseNet121** backbone for binary classification. |

---

## Results (same 339-image test set, same split)

| Backbone | Pretraining | ROC AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| ResNet18 | BYOL (self-supervised) | 0.832 | 0.77 | 0.73 | 0.83 |
| **DenseNet121** | **CheXNet (supervised)** | **0.847** | **0.79** | **0.78** | **0.81** |

→ CheXNet (supervised pretraining on NIH disease labels) transfers better than BYOL for this task. The decision threshold is selected via **Youden's J** to balance sensitivity and specificity.

---

## Data

- **Pretraining:** [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) — ~112,000 public chest X-ray images.
- **Fine-tuning:** Silicodata (set_A) — 4-class (silicosis / STB / TB / normal), collapsed into a binary task.

> **Privacy note:** Real patient data (X-ray images + clinical survey records) is **NOT** included in this repository because it contains personal/medical information. Only public datasets are used for backbone pretraining and fine-tuning.

---

## Credits & sources

- **CheXNet weights (DenseNet121):** pretrained weights are taken from the reproduction [arnoweng/CheXNet](https://github.com/arnoweng/CheXNet). These weights are NOT trained by us; we only remap the state-dict keys to load them into torchvision and then fine-tune.
- **CheXNet (original paper):** Rajpurkar et al., *"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"*, 2017. https://arxiv.org/abs/1711.05225
- **BYOL:** Grill et al., *"Bootstrap Your Own Latent"*, NeurIPS 2020. https://arxiv.org/abs/2006.07733
- **NIH ChestX-ray14:** Wang et al., *"ChestX-ray8"*, CVPR 2017.

---

## How to run

```bash
# 1. BYOL pretraining (resize NIH images to 256x256 first)
python "Pre-train/resize_images.py"
python "Pre-train/pretrain_byol.py" --epochs 60 --batch-size 64

# 2. Fine-tune BYOL-ResNet18
python "Finetune 2/cnn-model.py"

# 3. Fine-tune CheXNet-DenseNet121
python "Finetune 3/cnn-model-chexnet.py"
```

See `Pre-train/FINETUNE_GUIDE.md` for detailed fine-tuning instructions.
