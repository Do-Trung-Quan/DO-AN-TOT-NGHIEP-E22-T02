"""P0.4 — Y nghia nhan ketqua (PAG_V2 muc 2).

Voi cac ca bnn=Co va ketqua=2 (57 ca co anh): cac truong ILO co gia tri khong; diem L1 cua ho nam o
dau so voi ca duong va ca am con lai; ketqua cua ho o cac luot kham khac.

Chay LOCAL (doc info.csv + xlsx). Chi in/ghi so lieu tong hop; danh sach id noi bo luu o json de truy vet.
Phan tich khong dung mo hinh (dem, ty le) thuc hien tren toan bo 1835 ca co anh; phan tich diem L1
chi dung dev vi OOF cua test khong ton tai.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fusion.baselines import boot_ci
from phase0.common import OUT, ROOT, Nodes, load_config, read_merged_xlsx, write_json

ILO = ["chatluongphim", "matdotonthuong", "kichthuoctt", "tonthuongkhac"]


def group_of(bnn_co: pd.Series, ketqua: pd.Series) -> pd.Series:
    g = np.where(bnn_co & (ketqua == 2), "bnnCo_ketqua2 (nhom 57)",
        np.where(bnn_co & (ketqua == 1), "bnnCo_ketqua1",
        np.where(~bnn_co & (ketqua == 1), "bnnKhong_ketqua1", "bnnKhong_ketqua2")))
    return pd.Series(g, index=ketqua.index)


def main():
    cfg = load_config()
    seed = cfg["seed"]
    n_boot = cfg["n_boot"]
    m = read_merged_xlsx()
    info = pd.read_csv(ROOT / "data/info.csv", low_memory=False)
    ident = pd.read_parquet(ROOT / "output/ketqua/fusion_node_meta.parquet")
    m = m.merge(ident[["id", "patient_uid", "split", "fold_id"]], on="id", validate="1:1")
    m["bnn_co"] = m["bnn"].astype(str).str.strip().str.lower().eq("co")
    m["grp"] = group_of(m["bnn_co"], m["ketqua"])
    img = m["has_image"].astype(bool)
    out = {"config_hash": cfg["_hash"]}

    print("=" * 78)
    print("P0.4 — nhan ketqua va nhom bnn=Co & ketqua=2")
    print("=" * 78)

    # --- bang cheo
    cross = pd.crosstab([m["has_image"], m["bnn_co"]], m["ketqua"])
    print(cross.to_string())
    out["bang_cheo_bnn_ketqua"] = {f"co_anh={a}|bnn_co={b}": {int(k): int(v) for k, v in r.items()}
                                   for (a, b), r in cross.iterrows()}

    # --- 1. truong ILO co gia tri khong (toan bo 8030 va rieng 1835 co anh)
    print("\n[1] Truong ILO co gia tri (so dong khong thieu / tong) theo nhom")
    rows = []
    for scope, d in (("co_anh", m[img]), ("toan_8030", m)):
        for g, sub in d.groupby("grp"):
            rec = {"pham_vi": scope, "nhom": g, "n": int(len(sub))}
            for c in ILO:
                rec[c] = int(sub[c].notna().sum())
            rows.append(rec)
    ilo = pd.DataFrame(rows)
    print(ilo[ilo.pham_vi == "co_anh"].to_string(index=False))
    out["truong_ilo_theo_nhom"] = rows

    inf = info.merge(m[["id", "grp", "bnn_co"]], on="id", validate="1:1")
    vz = inf.groupby("grp")["vungtonthuong"].apply(lambda s: int(s.notna().sum())).to_dict()
    print("  vungtonthuong (chi co trong info.csv, 1835 dong):", vz)
    out["vungtonthuong_co_gia_tri"] = vz
    g57 = inf["grp"].eq("bnnCo_ketqua2 (nhom 57)")
    out["ilo_nhom57"] = {
        "n": int(g57.sum()),
        "matdotonthuong_co_gia_tri": int(inf.loc[g57, "matdotonthuong"].notna().sum()),
        "kichthuoctt_co_gia_tri": int(inf.loc[g57, "kichthuoctt"].notna().sum()),
        "vungtonthuong_co_gia_tri": int(inf.loc[g57, "vungtonthuong"].notna().sum()),
        "tonthuongkhac_co_gia_tri": int(inf.loc[g57, "tonthuongkhac"].notna().sum()),
        "ghi_chu": "mat do, kich thuoc, vung ton thuong chi co khi ketqua=1 tren toan bo du lieu"}
    print("  nhom 57:", out["ilo_nhom57"])

    # --- nam cong nhan benh vs nam kham, loai benh nghe khai bao
    ye = inf[inf["bnn_co"]].copy()
    ye["cach"] = pd.to_numeric(ye["namdt"], errors="coerce") - pd.to_numeric(ye["nam"], errors="coerce")
    out["nam_cong_nhan_so_voi_nam_kham"] = {
        g: {"n": int(len(s)), "trung_vi_cach_nam": float(s["cach"].median()),
            "q1": float(s["cach"].quantile(0.25)), "q3": float(s["cach"].quantile(0.75)),
            "truoc_dot_kham": int((s["cach"] > 0).sum()), "cung_nam": int((s["cach"] == 0).sum())}
        for g, s in ye.groupby("grp")}
    print("\n[nam cong nhan so voi nam kham]", out["nam_cong_nhan_so_voi_nam_kham"])
    cs = {g: s["bnncuthe"].astype(str).str.strip().str.lower().value_counts().head(4).to_dict()
          for g, s in ye.groupby("grp")}
    out["bnncuthe_phan_bo"] = cs

    # --- 2. diem L1 (chi dev, co anh)
    print("\n[2] Diem L1 (OOF) cua nhom 57 so voi cac nhom con lai — chi dev")
    n_all57 = int(((m.grp == "bnnCo_ketqua2 (nhom 57)") & img).sum())
    l1_out = {}
    for source in ("control", "frozen"):
        nd = Nodes(source)
        oof = np.load(ROOT / f"fusion/output/phase3_oof_{source}.npz")["L1"]
        df = pd.DataFrame({"id": nd.id, "l1": oof, "dev": nd.dev, "has": nd.has, "y": nd.y, "uid": nd.uid})
        df = df.merge(m[["id", "grp", "bnn_co"]], on="id", validate="1:1")
        df = df[df.dev & df.has]
        s57 = df.l1[df.grp == "bnnCo_ketqua2 (nhom 57)"]
        neg = df[df.grp == "bnnKhong_ketqua2"]
        pos = df[df.y == 1]
        ref = {}
        for nm, other in (("am_con_lai", neg), ("duong", pos)):
            sub = pd.concat([df[df.grp == "bnnCo_ketqua2 (nhom 57)"], other])
            yy = (sub.grp == "bnnCo_ketqua2 (nhom 57)").to_numpy().astype(int)
            auc = roc_auc_score(yy, sub.l1.to_numpy())
            ci = boot_ci(yy, sub.l1.to_numpy(), sub.uid.to_numpy(), roc_auc_score, n=n_boot)
            ref[f"AUC_nhom57_vs_{nm}"] = {"auc": float(auc), "ci": ci,
                                         "ghi_chu": "AUC>0.5 nghia la nhom 57 co diem cao hon nhom so sanh"}
        # phan vi cua tung ca trong nhom am con lai va trong nhom duong
        pct_neg = np.array([(neg.l1 < v).mean() for v in s57])
        pct_pos = np.array([(pos.l1 < v).mean() for v in s57])
        l1_out[source] = {
            "n_nhom57_dev": int(len(s57)),
            "diem_L1": {"trung_vi_nhom57": float(s57.median()), "tb_nhom57": float(s57.mean()),
                        "trung_vi_am_con_lai": float(neg.l1.median()),
                        "trung_vi_duong": float(pos.l1.median())},
            "phan_vi_trong_am_con_lai": {"trung_vi": float(np.median(pct_neg)),
                                         "ty_le_tren_P75_am": float((pct_neg > 0.75).mean())},
            "phan_vi_trong_duong": {"trung_vi": float(np.median(pct_pos)),
                                    "ty_le_tren_trung_vi_duong": float((s57 > pos.l1.median()).mean())},
            **ref}
        print(f"  [{source}] n={len(s57)} | trung vi L1: nhom57 {s57.median():.3f} | am con lai {neg.l1.median():.3f} | duong {pos.l1.median():.3f}")
        print(f"     AUC nhom57 vs am con lai {ref['AUC_nhom57_vs_am_con_lai']['auc']:.3f} "
              f"[{ref['AUC_nhom57_vs_am_con_lai']['ci'][0]:.3f}; {ref['AUC_nhom57_vs_am_con_lai']['ci'][1]:.3f}] | "
              f"vs duong {ref['AUC_nhom57_vs_duong']['auc']:.3f} "
              f"[{ref['AUC_nhom57_vs_duong']['ci'][0]:.3f}; {ref['AUC_nhom57_vs_duong']['ci'][1]:.3f}]")
    out["diem_L1"] = l1_out
    out["luu_y_control"] = ("Embedding 'control' duoc fine-tune tren SetA voi nhan bnn (tap ngoai); "
                            "neu nhom 57 co diem cao o 'control' ma khong o 'frozen' thi tin hieu co the den tu viec "
                            "tinh chinh theo bnn, khong chi tu phim cua ho.")
    out["so_dong_nhom57_tong_co_anh"] = n_all57
    out["so_dong_nhom57_trong_dev"] = {s: l1_out[s]["n_nhom57_dev"] for s in l1_out}

    # --- 3. ketqua o cac luot kham khac cua cung nguoi
    print("\n[3] ketqua o cac luot kham khac cua cung nguoi (qua patient_uid, toan 8030 dong)")
    per = {}
    for g, sub in m.groupby("grp"):
        if g != "bnnCo_ketqua2 (nhom 57)" and g != "bnnCo_ketqua1":
            continue
        has_other, other_pos, other_neg = 0, 0, 0
        for idx, r in sub.iterrows():
            oth = m[(m.patient_uid == r.patient_uid) & (m.id != r.id)]
            if len(oth):
                has_other += 1
                other_pos += int((oth.ketqua == 1).any())
                other_neg += int((oth.ketqua == 2).all())
        per[g] = {"n": int(len(sub)), "co_luot_khac": has_other,
                  "co_it_nhat_mot_luot_khac_ketqua1": other_pos,
                  "tat_ca_luot_khac_ketqua2": other_neg}
    # rieng nhom 57 co anh
    sub = m[(m.grp == "bnnCo_ketqua2 (nhom 57)") & img]
    k = {"n": int(len(sub)), "co_luot_khac": 0, "co_it_nhat_mot_luot_khac_ketqua1": 0}
    for _, r in sub.iterrows():
        oth = m[(m.patient_uid == r.patient_uid) & (m.id != r.id)]
        if len(oth):
            k["co_luot_khac"] += 1
            k["co_it_nhat_mot_luot_khac_ketqua1"] += int((oth.ketqua == 1).any())
    per["nhom57_co_anh"] = k
    out["ketqua_luot_khac"] = per
    for g, v in per.items():
        print(f"  {g}: {v}")

    # toan bo 8030: nhom bnn=Co&ketqua=2 chia theo split
    out["nhom_bnnCo_ketqua2_theo_split"] = {
        sp: int(((m.grp == 'bnnCo_ketqua2 (nhom 57)') & img & (m.split == sp)).sum()) for sp in ("dev", "test")}
    out["ids_nhom57_co_anh"] = [int(x) for x in m.loc[(m.grp == "bnnCo_ketqua2 (nhom 57)") & img, "id"]]

    write_json(OUT / "p04_ketqua_meaning.json", out)
    print("\nghi", OUT / "p04_ketqua_meaning.json")


if __name__ == "__main__":
    main()
