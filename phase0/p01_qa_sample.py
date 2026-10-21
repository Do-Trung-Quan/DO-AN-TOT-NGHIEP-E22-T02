"""P0.1 — chon anh cho QA truc quan (PG_update ban 3, muc 2 "QA").

50 anh ngau nhien PHAN TANG theo lo x nhan (seed 42) + MOI anh bi danh co o QA tu dong.
Moi tang (lo, nhan) co it nhat 1 anh, phan con lai chia theo ty le co tang. Nhan chi dung de rai
mau cho deu, khong ghi vao file cham va khong dung cho bat ky quyet dinh mo hinh nao.

Ra: p01_qa_cham_tay.csv (stt, img_id, nguon, lo, cac co, cot `diem` de trong, `ghi_chu`).
Nguoi cham dien `diem` = dat | mot_phan | hong, roi chay phase0/p01_qa_summary.py.

  python -m phase0.p01_qa_sample --qa /content/p01_processed/p01_mask_qa.csv --out /content/p01_processed/p01_qa_cham_tay.csv

Trong notebook: from phase0.p01_qa_sample import ve_luoi; ve_luoi(npz, bang_cham, tu=0, den=40)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from phase0.common import ROOT, load_config


def lot_label(dataset_dir: Path | None = None) -> pd.DataFrame:
    d = np.load((dataset_dir or ROOT / "fusion/output") / "fusion_dataset_ketqua_frozen.npz", allow_pickle=False)
    df = pd.DataFrame({"img_id": d["id"].astype(int), "lo": d["batch_date"].astype(str),
                       "nhan": d["y"].astype(int), "co_anh": d["has_image"].astype(bool)})
    return df[df.co_anh].drop(columns="co_anh")


def stratified(df: pd.DataFrame, n: int, seed: int) -> list[int]:
    """Chon n img_id phan tang theo (lo, nhan): moi tang >= 1, phan du chia theo co tang (du lon nhat)."""
    rng = np.random.default_rng(seed)
    groups = {k: g.img_id.to_numpy() for k, g in df.groupby(["lo", "nhan"])}
    keys = sorted(groups)
    size = np.array([len(groups[k]) for k in keys], float)
    if len(keys) <= n:
        alloc = np.ones(len(keys), int)
        rest = n - len(keys)
        cap = size - alloc
        share = rest * cap / cap.sum()
        add = np.floor(share).astype(int)
        order = np.argsort(-(share - add))
        for i in order[: rest - add.sum()]:
            add[i] += 1
        alloc += np.minimum(add, cap.astype(int))
    else:                                   # nhieu tang hon so mau: chon tang theo ty le co
        p = size / size.sum()
        pick = rng.choice(len(keys), n, replace=False, p=p)
        alloc = np.zeros(len(keys), int)
        alloc[pick] = 1
    out = []
    for k, a in zip(keys, alloc):
        out += list(rng.choice(groups[k], a, replace=False))
    return sorted(int(x) for x in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, required=True, help="p01_mask_qa.csv tu p01_preprocess")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config()["ban3"]["qa_truc_quan"]

    qa = pd.read_csv(args.qa)
    ll = lot_label(args.dataset_dir)
    df = ll[ll.img_id.isin(qa.img_id)]
    assert len(df) == len(qa), "QA va dataset lech so anh"
    rand = set(stratified(df, cfg["n_ngau_nhien"], cfg["seed"]))
    flagged = set(qa.loc[qa.co_bat_ky == 1, "img_id"])
    flag_cols = [c for c in qa.columns if c.startswith("co_") and c != "co_bat_ky"]

    rows = []
    for i in sorted(rand) + sorted(flagged - rand):
        r = qa.set_index("img_id").loc[i]
        nguon = "ngau_nhien+co" if (i in rand and i in flagged) else ("ngau_nhien" if i in rand else "co")
        rows.append({"img_id": i, "nguon": nguon, "lo": ll.set_index("img_id").loc[i, "lo"],
                     "co": ";".join(c[3:] for c in flag_cols if r[c] == 1), "diem": "", "ghi_chu": ""})
    out = pd.DataFrame(rows)
    out.insert(0, "stt", np.arange(1, len(out) + 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    n_l = df.groupby(["lo", "nhan"]).ngroups
    print(f"{n_l} tang (lo x nhan) | {len(rand)} anh ngau nhien | {len(flagged)} anh bi co "
          f"({len(flagged & rand)} trung) | tong can cham {len(out)}")
    print(f"ghi {args.out}  — dien cot 'diem' = dat | mot_phan | hong")


def ve_luoi(npz, bang: pd.DataFrame, tu: int = 0, den: int = 40, cot: int = 5):
    """Ve tung anh trong bang[tu:den]: V1 co duong vien mask (trai) va V2 (phai). Chi xem, KHONG luu hinh."""
    import matplotlib.pyplot as plt
    pos = {int(k): j for j, k in enumerate(npz["img_id"])}
    sub = bang.iloc[tu:den]
    hang = int(np.ceil(len(sub) / cot))
    fig, ax = plt.subplots(hang, 2 * cot, figsize=(4 * cot, 2.3 * hang))
    ax = np.atleast_2d(ax)
    for a in ax.ravel():
        a.axis("off")
    for n, (_, r) in enumerate(sub.iterrows()):
        j = pos[int(r.img_id)]
        a1, a2 = ax[n // cot, 2 * (n % cot)], ax[n // cot, 2 * (n % cot) + 1]
        a1.imshow(npz["V1"][j], cmap="gray")
        a1.contour(npz["mask"][j], levels=[0.5], colors="r", linewidths=0.6)
        a1.set_title(f"#{r.stt} {r.co if isinstance(r.co, str) else ''}"[:28], fontsize=7)
        a2.imshow(npz["V2"][j], cmap="gray")
    plt.tight_layout()
    plt.show()


def xem_anh(npz, img_ids, qa: pd.DataFrame | None = None):
    """Xem mot hay nhieu anh theo img_id: V0 (toan anh da che chu) | V1 + vien mask | V2. Chi xem, KHONG luu hinh."""
    import matplotlib.pyplot as plt
    pos = {int(k): j for j, k in enumerate(npz["img_id"])}
    img_ids = [img_ids] if np.isscalar(img_ids) else list(img_ids)
    fig, ax = plt.subplots(len(img_ids), 3, figsize=(10, 3.4 * len(img_ids)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(img_ids):
        j = pos[int(i)]
        for c, (k, t) in enumerate((("V0", "V0 toàn ảnh"), ("V1", "V1 cắt + che chữ"), ("V2", "V2 chỉ phổi"))):
            ax[r, c].imshow(npz[k][j], cmap="gray")
            ax[r, c].axis("off")
            ax[r, c].set_title(f"img_id {i} · {t}", fontsize=8)
        ax[r, 1].contour(npz["mask"][j], levels=[0.5], colors="r", linewidths=0.6)
        if qa is not None:
            q = qa.set_index("img_id").loc[int(i)]
            ax[r, 2].set_title(f"cờ: {';'.join(c[3:] for c in qa.columns if c.startswith('co_') and c != 'co_bat_ky' and q[c]) or 'không'}",
                               fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
