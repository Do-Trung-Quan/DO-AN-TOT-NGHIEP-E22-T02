"""P0.6 — Khao sat du lieu kham theo vung (PAG_V2 muc 2, cho thiet ke ban 2).

Khoi 6 vung x 7 dau hieu cua nhanh "ts" cu khong duoc luu thanh file; dung lai tu xlsx voi BA trang thai:
Co / Khong / thieu (NaN). Ma hoa cu (_binary_khong_or_nan_to_0_else_1) gop NaN va "Khong" thanh 0.

Ra:
  1. bang 42 o (dau hieu x vung): so Co, Khong, thieu; ty le duong tren toan 8030 luot va tren o khong thieu;
     tach theo ketqua va co/khong anh
  2. quy tac cua thay: duong < 1% o MOI dau hieu x vung -> gop 6 node kham thanh 1, bo E6, quay ve ban 1
  3. do doc lap voi viec doc phim (dai dien gian tiep, vi khong co cot thu tu phieu / nguoi ghi):
     lien he giua viec co ghi dau hieu kham va ketqua, khong che tinh (phan tang theo tinh)

Chay LOCAL (doc xlsx). Chi ghi so tong hop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase0.common import (OUT, cluster_boot_indices, load_config, read_merged_xlsx, write_json)

SIGNS = ["rungt", "rungg", "goduc", "vtam", "vtno", "vtrit", "vtngay"]
AGG = ["rungthan", "go", "ran", "ranam", "ranno", "ranrit", "ranngay", "riraopn", "khoangls"]


def state(series: pd.Series) -> pd.Series:
    """'co' -> 1, 'khong' -> 0, thieu -> NaN."""
    t = series.astype(str).str.strip().str.lower()
    return pd.Series(np.where(t == "co", 1.0, np.where(t == "khong", 0.0, np.nan)), index=series.index)


def mh_or(y, x, strata) -> float:
    """Ty so chenh Mantel-Haenszel cua x doi voi y, phan tang theo strata."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = ((x[m] == 1) & (y[m] == 1)).sum(); b = ((x[m] == 1) & (y[m] == 0)).sum()
        c = ((x[m] == 0) & (y[m] == 1)).sum(); d = ((x[m] == 0) & (y[m] == 0)).sum()
        n = a + b + c + d
        if n:
            num += a * d / n
            den += b * c / n
    return num / den if den else np.nan


def main():
    cfg = load_config()
    raw = read_merged_xlsx()
    meta = pd.read_parquet("output/ketqua/fusion_node_meta.parquet")
    raw = raw.merge(meta[["id", "patient_uid", "split"]], on="id", validate="1:1")
    N = len(raw)
    ket = raw["ketqua"].to_numpy()
    has = raw["has_image"].astype(bool).to_numpy()
    out = {"config_hash": cfg["_hash"], "n_luot_kham": int(N)}
    print("=" * 78)
    print("P0.6 — du lieu kham theo vung (6 vung x 7 dau hieu)")
    print("=" * 78)

    # ------------------------------------------------ 1. bang 42 o
    rows = []
    S = {}
    for sg in SIGNS:
        for z in range(1, 7):
            st = state(raw[f"{sg}{z}"])
            S[(sg, z)] = st
            co, kh = int((st == 1).sum()), int((st == 0).sum())
            rows.append({
                "dau_hieu": sg, "vung": z, "co": co, "khong": kh, "thieu": int(st.isna().sum()),
                "ty_le_co_tren_8030": co / N,
                "ty_le_co_tren_khong_thieu": co / (co + kh) if co + kh else None,
                "co_khi_ketqua1": int(((st == 1) & (ket == 1)).sum()),
                "co_khi_ketqua2": int(((st == 1) & (ket == 2)).sum()),
                "co_khi_co_anh": int(((st == 1) & has).sum()),
                "co_khi_khong_anh": int(((st == 1) & ~has).sum())})
    tab = pd.DataFrame(rows)
    out["bang_42_o"] = tab.to_dict("records")
    by_sign = tab.groupby("dau_hieu")[["co", "khong", "thieu"]].sum()
    print("\nTong theo dau hieu (6 vung cong lai):")
    print(by_sign.to_string())
    n_cells = len(tab) * N
    n_filled = int(tab[["co", "khong"]].to_numpy().sum())
    print(f"\nO khong thieu: {n_filled} / {n_cells} ({n_filled / n_cells:.4%}); trong do 'Co': {int(tab.co.sum())}")
    mx = tab.loc[tab.ty_le_co_tren_8030.idxmax()]
    print(f"Ty le 'Co' cao nhat tren 8030 luot: {mx.ty_le_co_tren_8030:.4%} (dau hieu {mx.dau_hieu}, vung {int(mx.vung)})")
    out["tom_tat"] = {"o_khong_thieu": n_filled, "tong_o": int(n_cells), "o_co": int(tab.co.sum()),
                      "ty_le_co_cao_nhat_8030": float(mx.ty_le_co_tren_8030),
                      "o_cao_nhat": f"{mx.dau_hieu}{int(mx.vung)}"}

    # ------------------------------------------------ 2. quy tac cua thay
    n_above = int((tab.ty_le_co_tren_8030 >= 0.01).sum())
    trigger = n_above == 0
    out["quy_tac_thay"] = {
        "dieu_kien": "ty le duong < 1% o MOI dau hieu x vung (mau so: tat ca 8030 luot kham)",
        "so_o_>=1%": n_above,
        "kich_hoat": bool(trigger),
        "hanh_dong_neu_kich_hoat": "gop 6 node kham thanh 1 node, bo canh E6, thiet ke quay ve ban 1",
        "bien_the_mau_so_khong_thieu": {
            "so_o_>=1%": int((tab.ty_le_co_tren_khong_thieu.fillna(0) >= 0.01).sum()),
            "ghi_chu": "tren o khong thieu thi ty le cao hon nhung chi dua tren it quan sat; khong dung de quyet dinh"}}
    print(f"\n[quy tac thay] so o co ty le 'Co' >= 1%: {n_above}/42 -> kich hoat = {trigger}")

    # ------------------------------------------------ cot tong hop theo luot
    agg = {}
    for c in AGG:
        t = raw[c].astype(str).str.strip().str.lower()
        agg[c] = {k: int(v) for k, v in t.value_counts().head(5).items()}
    out["cot_tong_hop_theo_luot"] = agg
    ran = state(raw["ran"])
    any_zone_ran = np.zeros(N, bool)
    for sg in ("vtam", "vtno", "vtrit", "vtngay"):
        for z in range(1, 7):
            any_zone_ran |= (S[(sg, z)] == 1).to_numpy()
    out["nhat_quan_ran_tong_hop_va_vung"] = {
        "ran_tong_hop_co": int((ran == 1).sum()),
        "co_it_nhat_mot_o_ran_theo_vung_co": int(any_zone_ran.sum()),
        "ran_tong_hop_co_nhung_khong_o_vung_nao": int(((ran == 1).to_numpy() & ~any_zone_ran).sum()),
        "o_vung_co_nhung_ran_tong_hop_khong": int((any_zone_ran & ~(ran == 1).to_numpy()).sum())}
    print("\n[nhat quan ran tong hop vs vung]", out["nhat_quan_ran_tong_hop_va_vung"])

    # ------------------------------------------------ 3. doc lap voi viec doc phim
    filled = np.zeros(N, bool)
    for st in S.values():
        filled |= st.notna().to_numpy()
    anypos = np.zeros(N, bool)
    for st in S.values():
        anypos |= (st == 1).to_numpy()
    prov = raw["tinh"].astype(str).to_numpy()
    y = (ket == 1).astype(int)
    uid = raw["patient_uid"].to_numpy()

    def block(name, x):
        x = x.astype(int)
        or_ = mh_or(y, x, prov)
        vals = []
        for idx in cluster_boot_indices(uid, cfg["n_boot"] // 4, cfg["seed"]):
            v = mh_or(y[idx], x[idx], prov[idx])
            if np.isfinite(v):
                vals.append(v)
        ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        return {"n_x1": int(x.sum()), "ty_so_chenh_MH_phan_tang_theo_tinh": float(or_), "ci": ci,
                "ghi_chu": "OR>1: co ghi/duong thi ketqua=1 nhieu hon; KTC cat 1 la chua phan biet duoc"}

    out["doc_lap_voi_doc_phim"] = {
        "co_it_nhat_mot_o_khong_thieu": {"so_luot": int(filled.sum()), **block("filled", filled)},
        "co_it_nhat_mot_o_Co": {"so_luot": int(anypos.sum()), **block("anypos", anypos)},
        "ghi_chu": ("Khong co cot thu tu trong phieu hay nguoi ghi, nen khong kiem tra truc tiep duoc. "
                    "Day chi la lien he gian tiep sau khi khong che tinh."),
    }
    f = out["doc_lap_voi_doc_phim"]
    for k in ("co_it_nhat_mot_o_khong_thieu", "co_it_nhat_mot_o_Co"):
        print(f"\n[{k}] {f[k]['so_luot']} luot | OR_MH {f[k]['ty_so_chenh_MH_phan_tang_theo_tinh']:.2f} "
              f"[{f[k]['ci'][0]:.2f}; {f[k]['ci'][1]:.2f}]")
    # co/khong anh
    out["luot_co_o_khong_thieu_theo_anh"] = {"co_anh": int((filled & has).sum()),
                                              "khong_anh": int((filled & ~has).sum())}
    print("\n[phan bo theo co/khong anh]", out["luot_co_o_khong_thieu_theo_anh"])
    out["ban_do_vung"] = ("CHUA XAC NHAN: vung 1-6 ung voi vi tri nao (RU/RM/RL/LU/LM/LL) khong co tai lieu trong repo. "
                          "Can hoi thay hoac khoa.")

    write_json(OUT / "p06_exam_survey.json", out)
    print("\nghi", OUT / "p06_exam_survey.json")


if __name__ == "__main__":
    main()
