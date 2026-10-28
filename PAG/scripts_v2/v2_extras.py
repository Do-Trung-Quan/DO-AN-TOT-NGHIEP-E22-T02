"""PAG_V2 §10 — san pham ban giao: bang ket qua tong hop duy nhat, phien ban thu vien, hash file fold, config tung thi nghiem.
Doc cac CSV gop da co trong results/ (khong doc lai du lieu benh nhan, tru hash fold). Ghi results/tong_hop.
"""
import sys, json, hashlib, platform, importlib
import numpy as np, pandas as pd
from pag_lib import *

R = Report("tong_hop", "Tổng hợp — bảng kết quả duy nhất, phiên bản thư viện, hash fold, cấu hình thí nghiệm")

# 1) bang tong hop: moi dong mot mo hinh, cot = nhom x chi so
src = [RES / "tuan2_du_lieu_do_thi" / "chi_so_doi_chung.csv", RES / "tuan3_PG_T0_T2" / "chi_so.csv"]
D = pd.concat([pd.read_csv(f) for f in src if f.exists()])
D = D[D.nhom.isin(["có ảnh", "không ảnh", "toàn bộ"])]
cols = ["roc_auc", "roc_auc_ci", "pr_auc", "brier", "ece", "nhay_tai_dh80", "auc_trong_lo"]
W = D.set_index(["mo_hinh", "nhom"])[[c for c in cols if c in D.columns]].unstack("nhom")
W.columns = [f"{g} | {m}" for m, g in W.columns]
order = [f"{g} | {m}" for g in ("có ảnh", "không ảnh", "toàn bộ") for m in cols if f"{g} | {m}" in W.columns and W[f"{g} | {m}"].notna().any()]
W = W[order].reset_index()
R.table(W, "bang_tong_hop", "Bảng kết quả tổng hợp — dev OOF, xác suất đã hiệu chỉnh trong fold",
        note="*Mỗi dòng một mô hình; cột theo nhóm (có ảnh / không ảnh / toàn dev). `roc_auc` (+KTC 95% bootstrap theo người), `pr_auc` (diện tích dưới đường precision–recall), "
             "`brier` (sai số bình phương của xác suất), `ece` (sai lệch hiệu chỉnh), `nhay_tai_dh80` (độ nhạy khi độ đặc hiệu 0,80), `auc_trong_lo` (AUC trong từng lô chụp). "
             "Dòng “toàn bộ” chỉ tham khảo: gộp hai nhóm có tỉ lệ dương khác nhau (≈25% vs ≈8%) nên AUC bị đẩy lên. Kết quả ablation ở tuan5_ablation; giả thuyết ở tuan4.*")

# 2) phien ban thu vien
vers = dict(python=sys.version.split()[0], he_dieu_hanh=platform.platform())
for m in ("numpy", "pandas", "sklearn", "scipy", "torch", "xgboost", "torchxrayvision", "cv2", "health_multimodal", "torch_geometric"):
    try: vers[m] = getattr(importlib.import_module(m), "__version__", "có (không rõ phiên bản)")
    except Exception: vers[m] = "không cài"
import torch
vers["cuda"] = torch.version.cuda; vers["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không"
R.table(pd.DataFrame(list(vers.items()), columns=["thu_vien", "phien_ban"]), "phien_ban_thu_vien", "Phiên bản thư viện")

# 3) hash fold (chi hash, khong ghi id)
core = load_core()
def h(s): return hashlib.sha256(s.encode()).hexdigest()[:16]
lines = "\n".join(f"{i}:{f}:{d}:{s}" for i, f, d, s in sorted(zip(core.id.astype(str), core.fold2, core.dev2, core.split)))
fold_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in [CACHE / "core.pkl"] if p.exists()}
hs = [dict(doi_tuong="phân chia id → (fold2, dev2, split)", sha256_16=h(lines), so_hang=len(core),
           dev=int(core.dev2.sum()), test=int((core.split == "test").sum()), so_nguoi_dev=int(core.loc[core.dev2, "uid2"].nunique()))]
hs += [dict(doi_tuong=f"tệp {k}", sha256_16=v) for k, v in fold_files.items()]
R.table(pd.DataFrame(hs), "hash_fold", "Hash phân chia fold (để kiểm tra tái lập; không chứa id)",
        note=f"*Seed: chia/bootstrap {SEED}; mạng PG/T 3 seed (0, 1, 2) ở vòng ngoài, seed 0 ở dò cấu hình và embedding; không gian dò 30 cấu hình (seed 42).*")
R.save()

# 4) config tung thi nghiem
cfg = {"khong_gian_do": SPACE, "bien_the": {n: variant(n) for n in VARIANTS}, "bien_tien_su_bo_o_PG-nobnn": list(HIST_VARS),
       "cau_hinh_chon": {n: json.loads((CACHE / f"selected_{n}.json").read_text()) for n in ("PG", "T0", "T2") if (CACHE / f"selected_{n}.json").exists()},
       "anh": json.loads((CACHE / "img_v2_config.json").read_text(encoding="utf-8"))}
(R.d / "cau_hinh_thi_nghiem.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(pd.DataFrame(list(vers.items())).to_string(index=False, header=False)); print(pd.DataFrame(hs).to_string(index=False))
print("EXTRAS XONG")
