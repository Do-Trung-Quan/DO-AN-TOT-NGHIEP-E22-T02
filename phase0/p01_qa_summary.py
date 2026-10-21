"""P0.1 — tong hop QA (tu dong + truc quan) va cong QA (PG_update ban 3, muc 2 "QA").

Doc p01_mask_qa.csv (QA tu dong) va p01_qa_cham_tay.csv (da dien cot `diem`).
  - Cong: ty le `dat` trong cac anh NGAU NHIEN >= 90% (`mot_phan` khong tinh la dat).
  - Anh cham `hong` (ngau nhien hay bi co): ghi vao p01_zone_missing.csv -> dung vector toan anh (V0)
    va gan co zone_missing o cac buoc sau. Khong loai anh nao.

  python -m phase0.p01_qa_summary --qa /content/p01_processed/p01_mask_qa.csv \
      --cham /content/p01_processed/p01_qa_cham_tay.csv --out-dir phase0/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase0.common import OUT, load_config, write_json

MUC = ("dat", "mot_phan", "hong")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, required=True)
    ap.add_argument("--cham", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    cfg = load_config()
    c = cfg["ban3"]["qa_truc_quan"]

    qa = pd.read_csv(args.qa)
    ch = pd.read_csv(args.cham, dtype={"diem": str})
    ch["diem"] = ch["diem"].fillna("").str.strip().str.lower()
    thieu = ch[~ch.diem.isin(MUC)]
    if len(thieu):
        raise SystemExit(f"{len(thieu)} dong chua cham hoac sai muc (chi nhan {MUC}): stt {thieu.stt.tolist()[:20]}")

    rand = ch[ch.nguon.str.contains("ngau_nhien")]
    if len(rand) != c["n_ngau_nhien"]:
        print(f"[CANH BAO] so anh ngau nhien = {len(rand)}, khai bao {c['n_ngau_nhien']}")
    ty_le = float((rand.diem == "dat").mean())
    dat = ty_le >= c["nguong_dat"]
    flagged = ch[ch.nguon.str.contains("co")]
    hong = ch[ch.diem == "hong"]

    flag_cols = [x for x in qa.columns if x.startswith("co_")]
    out = {
        "config_hash": cfg["_hash"],
        "qa_tu_dong": {"so_anh": int(len(qa)),
                       "so_anh_theo_co": {x: int(qa[x].sum()) for x in flag_cols},
                       "anh_co_hop_chu": int((qa.n_hop_chu > 0).sum()),
                       "dien_tich_che_trung_vi": float(qa.dien_tich_che.median()),
                       "dien_tich_phoi_trung_vi": float(qa.dien_tich_phoi.median()),
                       "ty_le_trai_phai_trung_vi": float(qa.ty_le_trai_phai.median())},
        "qa_truc_quan": {
            "ngau_nhien": {"n": int(len(rand)), **{m: int((rand.diem == m).sum()) for m in MUC},
                           "ty_le_dat": ty_le, "nguong": c["nguong_dat"], "dat_cong": bool(dat)},
            "anh_bi_co": {"n": int(len(flagged)), **{m: int((flagged.diem == m).sum()) for m in MUC}},
        },
        "zone_missing": {"n": int(len(hong)), "img_id": [int(x) for x in hong.img_id]},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hong[["img_id"]].to_csv(args.out_dir / "p01_zone_missing.csv", index=False)
    write_json(args.out_dir / "p01_qa_tom_tat.json", out)

    r = out["qa_truc_quan"]
    print(f"QA tu dong: {len(qa)} anh; bi co it nhat mot loi: {int(qa.co_bat_ky.sum())}")
    for x in flag_cols:
        print(f"  {x:28s} {int(qa[x].sum())}")
    print(f"QA truc quan, 50 anh ngau nhien: dat {r['ngau_nhien']['dat']} | mot phan {r['ngau_nhien']['mot_phan']} | "
          f"hong {r['ngau_nhien']['hong']} -> ty le dat {ty_le:.0%} (nguong {c['nguong_dat']:.0%})")
    print(f"Anh bi co: {r['anh_bi_co']['n']} | dat {r['anh_bi_co']['dat']} | mot phan {r['anh_bi_co']['mot_phan']} | "
          f"hong {r['anh_bi_co']['hong']}")
    print(f"zone_missing (anh hong, dung vector toan anh): {len(hong)} -> {args.out_dir / 'p01_zone_missing.csv'}")
    print("\n=> CONG QA: " + ("DAT, duoc trich embedding va chay P0.1" if dat else
                              "KHONG DAT — sua xu ly anh (p01_preprocess.py) roi chay lai, CHUA chay P0.1"))


if __name__ == "__main__":
    main()
