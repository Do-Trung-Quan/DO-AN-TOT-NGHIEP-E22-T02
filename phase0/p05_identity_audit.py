"""P0.5 (b, c, d) — hoan thien kiem toan dinh danh (PAG_V2 muc 2).

  (b) SHA-256 NOI DUNG anh (khong phai hash ten file): anh co trung giua cac luot kham cua cung mot
      nguoi khong, ketqua cua cac luot do co nhat quan khong; SHA trung giua hai patient_uid khac nhau.
  (c) Quy tac ghep 462 ca duong theo tung muc chac chan, so ca o moi muc.
  (d) Ghep noi long (ho ten + nam sinh + gioi + tinh): uoc luong so nguoi bi tach nham thanh hai
      patient_uid va cac cap vat cheo fold / dev-test (ro ri chu the tiem an).

Chay LOCAL. Ho ten, so dien thoai, ten file CHI duoc dung trong bo nho de so khop; khong in, khong ghi.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image

from imagefeat.build_index import extract_batch_date, extract_id
from phase0.common import OUT, ROOT, load_config, read_merged_xlsx, write_json
from scripts.patient_identity import normalize_field, normalize_name, normalize_namsinh

DATA = ROOT / "data"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pix(p) -> np.ndarray:
    im = Image.open(p)
    im.draft("L", (256, 256))
    a = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def scan(folder_rel: str, label: int | None) -> list[dict]:
    rows = []
    base = DATA / folder_rel
    for p in sorted(base.rglob("*.jpg")):
        rows.append({"path": p, "fname": p.name, "id": extract_id(p.name),
                     "folder_pos": int(p.parent.name == "silicosis")})
    return rows


def norm_vi(text) -> str:
    """Bo dau tieng Viet, KE CA chu D-gach (NFD khong tach duoc d-gach nen can doi tay).
    patient_identity.normalize_name khong xu ly d-gach; phep so ten o day can dung."""
    t = str(text).replace("đ", "d").replace("Đ", "D")
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").upper()
    return re.sub(r"[^A-Z]+", "_", t).strip("_")


def name_part(fname: str) -> str:
    """Phan chu trong ten file (bo so, ngay) -> chuoi A_B_C da bo dau; co the rong."""
    stem = re.sub(r"\.[A-Za-z]+$", "", fname)
    return norm_vi(re.sub(r"\d+", " ", stem))


def name_level(fname_name: str, hoten_norm: str) -> str:
    if not fname_name:
        return "khong_co_ten_trong_file"
    if fname_name == hoten_norm:
        return "trung_khop"
    a, b = fname_name.split("_"), hoten_norm.split("_")
    if sorted(a) == sorted(b):
        return "trung_token"
    if set(a) <= set(b) or set(b) <= set(a):
        return "tap_con"
    r = SequenceMatcher(None, fname_name, hoten_norm).ratio()
    return "gan_giong_>=0.85" if r >= 0.85 else "khac"


def main():
    cfg = load_config()
    print("=" * 78)
    print(f"P0.5 (b,c,d) — kiem toan dinh danh | config {cfg['_hash']}")
    print("=" * 78)

    meta = pd.read_parquet(ROOT / "output/ketqua/fusion_node_meta.parquet")
    raw = read_merged_xlsx()
    info = pd.read_csv(DATA / "info.csv", low_memory=False, usecols=["id", "hoten", "ketqua", "namdt"])
    raw = raw.merge(meta[["id", "patient_uid", "split", "fold_id", "patient_group"]], on="id", validate="1:1")
    out = {"config_hash": cfg["_hash"]}

    # ------------------------------------------------------------ anh
    rows = scan("train", None) + scan("test", None)
    img = pd.DataFrame(rows)
    assert len(img) == 1835, len(img)
    img["id_ok"] = img["id"].notna()
    img = img.dropna(subset=["id"]).assign(id=lambda d: d["id"].astype(int))
    print(f"anh: {len(rows)} | trich duoc id: {len(img)}")
    img["sha"] = [sha256_file(p) for p in img["path"]]
    old = pd.DataFrame(scan("sillicosis", None))
    old["sha"] = [sha256_file(p) for p in old["path"]]
    img = img.merge(info[["id", "hoten", "ketqua", "namdt"]].rename(columns={"ketqua": "ketqua_info"}),
                    on="id", how="left", validate="m:1")
    img = img.merge(raw[["id", "ketqua", "patient_uid", "split", "fold_id"]], on="id", how="left",
                    suffixes=("", "_raw"), validate="m:1")
    assert img["ketqua"].notna().all(), "co anh khong khop id"

    # ------------------------------------------------------------ (b)
    print("\n(b) SHA-256 noi dung anh")
    dup_sha = img.groupby("sha").filter(lambda g: len(g) > 1)
    cross_uid = dup_sha.groupby("sha")["patient_uid"].nunique()
    per_uid = img.groupby("patient_uid").filter(lambda g: len(g) > 1)
    pairs, same_sha, corr_hi, ket_incons, pair_records = 0, 0, 0, 0, []
    for uid, g in per_uid.groupby("patient_uid"):
        for (i, a), (j, b) in combinations(g.iterrows(), 2):
            pairs += 1
            s = a.sha == b.sha
            same_sha += int(s)
            r = float(np.dot(pix(a.path), pix(b.path)) / (64 * 64))
            corr_hi += int(r >= 0.99)
            ket_incons += int(a.ketqua != b.ketqua)
            first, second = (a, b) if a.namdt <= b.namdt else (b, a)
            direction = ("cung_nam" if a.namdt == b.namdt else
                         "duong->am" if (first.ketqua == 1 and second.ketqua == 2) else
                         "am->duong" if (first.ketqua == 2 and second.ketqua == 1) else "giong_nhau")
            pair_records.append({"sha_giong": bool(s), "tuong_quan": r, "huong": direction,
                                 "ketqua_khac_nhau": bool(a.ketqua != b.ketqua),
                                 "lo_khac_nhau": bool(extract_batch_date(a.fname) != extract_batch_date(b.fname))})
    pr = pd.DataFrame(pair_records)
    out["sha"] = {
        "so_anh": int(len(img)), "so_sha_khac_nhau": int(img["sha"].nunique()),
        "nhom_sha_trung": int(dup_sha["sha"].nunique()),
        "nhom_sha_trung_giua_hai_patient_uid_khac_nhau": int((cross_uid > 1).sum()),
        "nguoi_co_nhieu_anh": int(per_uid["patient_uid"].nunique()),
        "cap_anh_cung_nguoi": pairs, "cap_sha_giong": same_sha,
        "cap_tuong_quan_pixel_>=0.99": corr_hi,
        "cap_ketqua_khac_nhau": ket_incons,
        "huong_cac_cap_ketqua_khac_nhau": dict(Counter(pr.loc[pr.ketqua_khac_nhau, "huong"])),
        "ghi_chu_huong": ("duong->am (nam kham truoc la duong, nam sau la am) kho giai thich bang dien tien "
                          "lam sang cua bui phoi; co the la nhieu nhan, doc phim khac nhau hoac gop nham nguoi"),
        "tuong_quan_pixel_trung_vi_cap_sha_khac": float(pr.loc[~pr.sha_giong, "tuong_quan"].median()) if (~pr.sha_giong).any() else None,
        "cap_khac_lo": int(pr.lo_khac_nhau.sum()) if len(pr) else 0}
    print(" ", out["sha"])

    # ------------------------------------------------------------ (c)
    print("\n(c) Quy tac ghep 462 ca duong")
    pos = img[img.ketqua == 1].copy()
    old_sha = set(old["sha"])
    tiers = {}
    tiers["tong_anh_ketqua1"] = int(len(pos))
    tiers["M1_id_tu_ten_file_khop_info_va_thu_muc_silicosis"] = int(
        ((pos.folder_pos == 1) & pos.ketqua_info.eq(1)).sum())
    pos["hoten_n"] = pos["hoten"].map(norm_vi)
    pos["ten_file"] = pos["fname"].map(name_part)
    lv = [name_level(a, b) for a, b in zip(pos.ten_file, pos.hoten_n)]
    pos["ten_muc"] = lv
    tiers["ten_trong_file_so_voi_hoten"] = dict(Counter(lv))
    tiers["M2_sha_trung_thu_muc_cu"] = int(pos.sha.isin(old_sha).sum())
    p_old = pos[~pos.sha.isin(old_sha)]
    tiers["M2_khong_trung_sha"] = int(len(p_old))
    strong = pos.ten_muc.isin(["trung_khop", "trung_token"])
    tiers["M3_id_va_ten_trung_khop_hoac_trung_token"] = int(strong.sum())
    tiers["M3b_id_va_ten_it_nhat_tap_con"] = int(pos.ten_muc.isin(["trung_khop", "trung_token", "tap_con"]).sum())
    tiers["M4_id_va_ten_it_nhat_gan_giong"] = int(pos.ten_muc.isin(["trung_khop", "trung_token", "tap_con", "gan_giong_>=0.85"]).sum())
    tiers["M5_khong_ten_trong_file_nhung_sha_trung_thu_muc_cu"] = int(
        (pos.ten_muc.eq("khong_co_ten_trong_file") & pos.sha.isin(old_sha)).sum())
    tiers["so_anh_cu_khong_co_trong_1835"] = int(len(old) - old["sha"].isin(set(img["sha"])).sum())
    # ca khong dat muc chat nhat: liet ke theo id
    weak = pos[~pos.sha.isin(old_sha) | pos.ten_muc.isin(["khac"])]
    tiers["id_khong_dat_muc_chac_chan_nhat"] = [int(x) for x in weak["id"]]
    out["ghep_462"] = tiers
    for k, v in tiers.items():
        if k != "id_khong_dat_muc_chac_chan_nhat":
            print(f"  {k}: {v}")
    print(f"  so id khong dat muc chac chan nhat: {len(weak)}")
    out["ghi_chu_459"] = ("Con so 459 cua thay khong co trong repo. Neu mot muc o tren cho dung 459 thi "
                          "doi chieu voi thay; neu khong, hoi thay manifest nao.")

    # ------------------------------------------------------------ (d)
    print("\n(d) Ghep noi long: ho ten + nam sinh + gioi + tinh")
    loose = pd.DataFrame({
        "uid": raw["patient_uid"], "id": raw["id"], "split": raw["split"], "fold": raw["fold_id"],
        "hoten": raw["hoten"].map(norm_vi),
        "namsinh": raw["namsinh"].map(normalize_namsinh),
        "gioi": raw["gioitinh"].map(normalize_field), "tinh": raw["tinh"].map(normalize_field),
        "has_phone": raw["sdt"].notna(), "ketqua": raw["ketqua"],
        "has_img": raw["has_image"].astype(bool),
        "cao": pd.to_numeric(raw["cao"], errors="coerce"), "can": pd.to_numeric(raw["can"], errors="coerce")})
    loose["key"] = loose[["hoten", "namsinh", "gioi", "tinh"]].agg("|".join, axis=1)
    grp = loose.groupby("key")
    n_uid = grp["uid"].nunique()
    multi = n_uid[n_uid > 1].index
    sub = loose[loose.key.isin(multi)]
    print(f"  khoa noi long: {loose.key.nunique()} khoa | {int(len(multi))} khoa gom >=2 patient_uid "
          f"({int(sub.uid.nunique())} patient_uid, {len(sub)} luot kham)")
    # --- bang chung "cung mot nguoi": chieu cao (do chinh xac ~ +-2 cm giua cac lan do)
    # Moc so sanh: cap nguoi NGAU NHIEN cung gioi, khac patient_uid -> ty le |chenh cao| <= 2 cm
    h = loose.groupby("uid").agg(cao=("cao", "median"), gioi=("gioi", "first"))
    rng = np.random.default_rng(cfg["seed"])
    null_hits = []
    for _gioi, gg in h.groupby("gioi"):
        v = gg["cao"].to_numpy()
        if len(v) > 50:
            a, b = rng.integers(0, len(v), 20000), rng.integers(0, len(v), 20000)
            keep = a != b
            null_hits.append(np.abs(v[a[keep]] - v[b[keep]]) <= 2)
    null_rate = float(np.concatenate(null_hits).mean())

    fold_span = split_span = 0
    sizes, risk_rows, risk_uid = [], set(), set()
    cat = Counter()
    pair_rows = []
    uid_fold = loose.groupby("uid")["fold"].apply(lambda x: set(x))
    for k, g in sub.groupby("key"):
        uids = list(g.uid.unique())
        sizes.append(len(uids))
        ph = g.has_phone.to_numpy()
        cat["tat_ca_co_sdt_khac_nhau" if ph.all() else
            "tat_ca_thieu_sdt" if (~ph).all() else "mot_phan_thieu_sdt"] += 1
        for u1, u2 in combinations(uids, 2):
            folds = uid_fold[u1] | uid_fold[u2]
            dh = abs(h.loc[u1, "cao"] - h.loc[u2, "cao"])
            pair_rows.append({"cao_gan_nhau": bool(dh <= 2),
                              "vat_cheo_fold_dev": len(folds - {-1}) > 1,
                              "vat_cheo_dev_test": (-1 in folds) and len(folds - {-1}) > 0,
                              "cung_fold": len(folds) == 1,
                              "co_anh": bool(g[g.uid.isin([u1, u2])].has_img.any())})
            if (len(folds - {-1}) > 1) or ((-1 in folds) and len(folds - {-1}) > 0):
                risk_uid |= {u1, u2}
                risk_rows |= set(g[g.uid.isin([u1, u2])].id)
    pp = pd.DataFrame(pair_rows)
    obs = float(pp.cao_gan_nhau.mean())
    # hon hop: obs = p*1 + (1-p)*null (gia su cung nguoi thi chieu cao gan nhau, khong chac) -> can duoi cua ty le cap that
    p_same = max(0.0, (obs - null_rate) / (1 - null_rate))
    fold_span = int(pp.vat_cheo_fold_dev.sum())
    split_span = int(pp.vat_cheo_dev_test.sum())
    likely = pp[pp.cao_gan_nhau]
    out["noi_long"] = {
        "so_patient_uid_hien_tai": int(loose.uid.nunique()),
        "so_khoa_noi_long": int(loose.key.nunique()),
        "khoa_gom_nhieu_patient_uid": int(len(multi)),
        "patient_uid_trong_cac_khoa_do": int(sub.uid.nunique()),
        "so_nguoi_co_the_bi_tach_nham_toi_da": int(sub.uid.nunique() - len(multi)),
        "phan_loai_theo_sdt": dict(cat),
        "kich_thuoc_khoa_(so_patient_uid)": dict(Counter(sizes)),
        "so_cap_patient_uid_cung_khoa": int(len(pp)),
        "chieu_cao_chenh_<=2cm": {"quan_sat": obs, "moc_ngau_nhien_cung_gioi": null_rate,
                                  "uoc_luong_ty_le_cap_cung_nguoi_that": p_same},
        "cap_vat_cheo_nhieu_fold_dev": fold_span,
        "cap_vat_cheo_dev_va_test": split_span,
        "cap_cung_fold": int(pp.cung_fold.sum()),
        "trong_do_cap_chieu_cao_gan_nhau": {
            "tong": int(len(likely)), "vat_cheo_fold_dev": int(likely.vat_cheo_fold_dev.sum()),
            "vat_cheo_dev_test": int(likely.vat_cheo_dev_test.sum()),
            "cung_fold": int(likely.cung_fold.sum())},
        "luot_kham_o_cap_vat_cheo": int(len(risk_rows)),
        "patient_uid_o_cap_vat_cheo": int(len(risk_uid)),
    }
    # tach rieng truong hop an toan hon: cung ten+namsinh+gioi+tinh va co anh
    out["noi_long"]["luot_kham_co_anh_o_cap_vat_cheo"] = int(sub[sub.has_img].id.isin(risk_rows).sum())
    for k, v in out["noi_long"].items():
        print(f"  {k}: {v}")
    out["noi_long"]["ghi_chu"] = (
        "Cung ten + nam sinh + gioi + tinh nhung khac patient_uid co the la nguoi khac (trung ten pho bien) "
        "hoac cung nguoi doi so dien thoai. Day la can tren cua so nguoi bi tach nham, khong phai so chac chan.")

    write_json(OUT / "p05_identity_audit.json", out)
    print("\nghi", OUT / "p05_identity_audit.json")


if __name__ == "__main__":
    main()
