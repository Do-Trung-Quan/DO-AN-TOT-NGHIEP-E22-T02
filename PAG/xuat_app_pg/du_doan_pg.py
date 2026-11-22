"""Mo-dun du doan PG cho app (nhan = ketqua: phim doc la nghi bui phoi silic). Mo hinh PG da khoa 07/11/2026, da cham test 1 lan.

Moi luot kham = 1 do thi 14 node (6 vung anh, 6 nhom lam sang, tinh, CLS). Du doan = trung binh 5 mo hinh fold;
moi fold: trung binh sigmoid 3 seed -> hieu chinh Platt rieng nhom co / khong phim (tham so fit trong fold, tren dev).
Moi so lieu lay tu thu muc mo hinh (cau_hinh_chung.json + fold*/), do PAG/scripts_v3/v3_xuat_pg.py xuat.
CHI ap dung cho 5 tinh trong du lieu hoc (tinh khac -> loi). Khong co nguong da khoa: tra xac suat.

    from du_doan_pg import MoHinhPG
    mh = MoHinhPG("duong/den/mo_hinh_PG_web")             # nap 1 lan; tu kiem sha256
    kq = mh.du_doan({"tinh": "Thai Nguyen", "gioitinh": "Nam", "namsinh": 1975, "cviec": "nghien da", "bnn": 0})
    kq = mh.du_doan(ban_ghi, anh="phim.png", duong_dan_biovil=".../biovil_t_image_model_proj_size_128.pt")
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
from mo_hinh_pg import PGNet, base_types  # noqa: E402
from nghe import them_cap_nghe  # noqa: E402

REG = ["RU", "RM", "RL", "LU", "LM", "LL"]
TEN_VUNG = {"RU": "phổi phải trên", "RM": "phổi phải giữa", "RL": "phổi phải dưới",
            "LU": "phổi trái trên", "LM": "phổi trái giữa", "LL": "phổi trái dưới"}
TINH_CO_DAU = {"Binh Dinh": "Bình Định", "Dong Nai": "Đồng Nai", "Hai Duong": "Hải Dương", "Phu Yen": "Phú Yên",
               "Thai Nguyen": "Thái Nguyên"}
TINH_IT_PHIM = {"Dong Nai", "Phu Yen"}
DISCLAIMER = "Kết quả thử nghiệm, không thay thế chẩn đoán của bác sĩ."


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _trong(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _muc(v) -> list[str]:
    """Gia tri -> cac chuoi ung vien, chuoi dau = pandas astype(str) nhu luc hoc."""
    if _trong(v):
        return ["__NA__"]
    out = [str(v)]
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        f = float(v); out += [str(f)] + ([str(int(f))] if f.is_integer() else [])
    return out


def _so(v):
    if _trong(v):
        return np.nan
    if isinstance(v, str):
        v = v.strip().replace(",", ".")
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) else np.nan


class MoHinhPG:
    def __init__(self, thu_muc_mo_hinh=None, thiet_bi=None, kiem_tra_sha=True):
        d = thu_muc_mo_hinh or os.environ.get("PG_MODEL_DIR") or DIR / "mo_hinh_PG_web"
        self.dir = Path(d)
        self.cfg = json.loads((self.dir / "cau_hinh_chung.json").read_text(encoding="utf-8"))
        if kiem_tra_sha:
            sai = [f for f, h in self.cfg["sha256"].items() if sha256(self.dir / f) != h]
            if sai:
                raise RuntimeError(f"file mo hinh bi thay doi (sha256 lech): {sai}")
        self.dev = torch.device(thiet_bi or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.nhom = self.cfg["nhom_bien"]; self.gnames = list(self.nhom)
        self.boi_canh = self.cfg["boi_canh"]
        self.tinh_hop_le = list(self.cfg["tinh_hop_le"])
        self.et = torch.as_tensor(base_types(6), device=self.dev)
        self.folds = []
        for fd in sorted(self.dir.glob("fold*")):
            ch = json.loads((fd / "cau_hinh.json").read_text(encoding="utf-8")); kt = ch["kien_truc"]
            nets = []
            for sp in sorted(fd.glob("seed*.pt")):
                m = PGNet(kt["d_img"], kt["n_img"], kt["clin_dims"], kt["dim"], kt["so_lop"], kt["dropout"], arch=kt["arch"])
                m.load_state_dict(torch.load(sp, map_location="cpu", weights_only=True), strict=True)
                nets.append(m.to(self.dev).eval())
            Z = np.load(fd / "ma_hoa_anh.npz")
            self.folds.append(dict(
                fold=ch["fold"], nets=nets, kien_truc=kt,
                spec=json.loads((fd / "ma_hoa_lam_sang.json").read_text(encoding="utf-8")),
                platt=json.loads((fd / "platt.json").read_text()),
                anh={k: Z[k] for k in Z.files}))
        assert len(self.folds) == 5 and all(len(f["nets"]) == 3 for f in self.folds), "can 5 fold x 3 seed"
        self._xu_ly_anh = None

    # ------------------------------------------------------------------ lam sang
    def chuan_bi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bang tho (ten cot nhu Excel goc) -> bang dung cho ma hoa: o trong -> NA, them bmi, fev1_fvc, cap nghe
        (giong pag_lib.load_base, v3_make_clin_raw, v3_base.add_job_levels). bmi va fev1_fvc LUON tinh lai."""
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: np.nan if _trong(v) else v)
        for c in ("cao", "can", "fvctt", "fev1tt"):
            if c not in df:
                df[c] = np.nan
        h, w = pd.to_numeric(df["cao"], errors="coerce") / 100, pd.to_numeric(df["can"], errors="coerce")
        df["bmi"] = (w / h ** 2).where((h > 1.2) & (h < 2.1) & (w > 25) & (w < 150))
        f1, fv = pd.to_numeric(df["fev1tt"], errors="coerce"), pd.to_numeric(df["fvctt"], errors="coerce")
        df["fev1_fvc"] = (f1 / fv).where(fv > 0)
        df = them_cap_nghe(df)
        return df

    def _ma_hoa(self, df, spec):
        """pag_lib.ClinEncoder.transform voi spec da fit trong fold -> 7 mang + co mat cua 6 nhom."""
        N = len(df); out, present = [], np.zeros((N, 6), bool)
        for gi, g in enumerate(self.gnames + ["Bối cảnh"]):
            cols = self.nhom[g] if g in self.nhom else self.boi_canh
            parts = []
            for c in cols:
                sp = spec[c]; s = df[c] if c in df else pd.Series([np.nan] * N, index=df.index)
                if sp[0] == "num":
                    v = np.array([_so(x) for x in s], np.float64); m = np.isnan(v)
                    z = ((np.where(m, sp[1], v) - sp[2]) / sp[3]).astype(np.float32)
                    parts.append(np.column_stack([z, m]).astype(np.float32) if sp[4] else z[:, None])
                else:
                    lv = sp[1]
                    if not lv:
                        parts.append(np.zeros((N, 0), np.float32)); continue
                    idx = {k: i for i, k in enumerate(lv)}; a = np.zeros((N, len(lv)), np.float32)
                    for r, x in enumerate(s):
                        j = next((idx[u] for u in _muc(x) if u in idx), None)
                        if j is not None:
                            a[r, j] = 1
                    parts.append(a)
            out.append(np.concatenate(parts, 1) if parts else np.zeros((N, 1), np.float32))
            if gi < 6:
                present[:, gi] = df.reindex(columns=cols).notna().any(axis=1).to_numpy() if cols else False
        return out, present

    def _anh_fold(self, regf, co_anh, A):
        """ImgEncoder.transform (chuan hoa + PCA whiten + clip 10) * co_anh."""
        sh = regf.shape; X = regf.reshape(-1, sh[-1]).astype(np.float64)
        Z = ((X - A["mu"]) / A["sd"] - A["pca_mean"]) @ A["pca_comp"].T / np.sqrt(A["pca_var"])
        Z = np.clip(Z, -10, 10).reshape(sh[0], sh[1], -1).astype(np.float32)
        return Z * co_anh[:, None, None]

    # ------------------------------------------------------------------ cham diem theo lo
    @torch.no_grad()
    def diem(self, df: pd.DataFrame, regf=None, area=None, co_anh=None, chi_fold=None, da_chuan_bi=False, bs=1024):
        """df: moi dong 1 luot kham. regf (N,6,512) va area (N,6) cho dong co phim (dong khong phim: bo qua gia tri).
        Tra dict(p: xac suat da hieu chinh, trung binh fold; p_tho: chua hieu chinh; p_fold: (5,N); chu_y: (N,6))."""
        if not da_chuan_bi:
            df = self.chuan_bi(df)
        N = len(df)
        tinh = df["tinh"].astype(str) if "tinh" in df else pd.Series(["nan"] * N)
        sai = sorted(set(tinh[~tinh.isin(self.tinh_hop_le)]))
        if sai:
            raise ValueError(f"tinh khong hop le {sai}; chi nhan {self.tinh_hop_le}")
        co_anh = np.zeros(N, bool) if co_anh is None else np.asarray(co_anh, bool)
        regf = np.zeros((N, 6, 512), np.float32) if regf is None else np.asarray(regf, np.float32)
        area = np.zeros((N, 6), np.float32) if area is None else np.asarray(area, np.float32) * co_anh[:, None]
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.dev)  # noqa: E731
        folds = self.folds if chi_fold is None else [f for f in self.folds if f["fold"] == chi_fold]
        P, PT, ATT = [], [], []
        for F_ in folds:
            clin, pres_g = self._ma_hoa(df, F_["spec"])
            img = self._anh_fold(regf, co_anh, F_["anh"])
            pres = np.zeros((N, 14), bool); pres[:, :6] = co_anh[:, None]; pres[:, 6:12] = pres_g; pres[:, 12] = True; pres[:, 13] = True
            T = dict(img=t(img), area=t(area * 10), clin=[t(c) for c in clin], pres=t(pres, torch.bool))
            ps, ats = [], []
            for m in F_["nets"]:
                pp, aa = [], []
                for i in range(0, N, bs):
                    sl = slice(i, i + bs); n = len(range(N)[sl])
                    lo, _, a = m(T["img"][sl], T["area"][sl], [c[sl] for c in T["clin"]], T["pres"][sl],
                                 self.et[None].expand(n, 14, 14))
                    pp.append(torch.sigmoid(lo)); aa.append(a.mean(1)[:, 13, :6])
                ps.append(torch.cat(pp).cpu().numpy()); ats.append(torch.cat(aa).cpu().numpy())
            p = np.mean(ps, 0); PT.append(p); ATT.append(np.mean(ats, 0))
            pc = p.copy()
            for g, key in ((True, "co_phim"), (False, "khong_phim")):
                mk, pl = co_anh == g, F_["platt"][key]
                if pl is not None and mk.any():
                    pc[mk] = 1 / (1 + np.exp(-(pl["a"] * logit(p[mk]) + pl["b"])))
            P.append(pc)
        att = np.mean(ATT, 0); att[~co_anh] = np.nan
        return dict(p=np.mean(P, 0), p_tho=np.mean(PT, 0), p_fold=np.array(P), chu_y=att)

    # ------------------------------------------------------------------ mot luot (cho API)
    def xu_ly_anh(self, anh, duong_dan_biovil=None):
        if self._xu_ly_anh is None:
            from xu_ly_anh import XuLyAnh
            b = duong_dan_biovil or os.environ.get("BIOVIL_T_PATH")
            if not b:
                raise ValueError("can duong dan file BioViL-T (duong_dan_biovil hoac bien moi truong BIOVIL_T_PATH)")
            self._xu_ly_anh = XuLyAnh(b, thiet_bi=str(self.dev))
        return self._xu_ly_anh(anh)

    def du_doan(self, lam_sang: dict, anh=None, duong_dan_biovil=None) -> dict:
        tinh = lam_sang.get("tinh")
        if tinh not in self.tinh_hop_le:
            raise ValueError(f"tinh '{tinh}' khong hop le; chi nhan {self.tinh_hop_le}")
        df = pd.DataFrame([{k: v for k, v in lam_sang.items()}], dtype=object)
        canh_bao, qa = [], None
        if anh is not None:
            r = self.xu_ly_anh(anh, duong_dan_biovil); qa = r["qa"]
            regf, area, co = r["regf"][None], r["area"][None], np.array([True])
            if not qa["du_phoi"]:
                canh_bao.append("Không tìm thấy đủ hai phổi trên ảnh: ảnh có thể không phải phim ngực thẳng.")
            if qa["dao_am"]:
                canh_bao.append("Ảnh có vẻ bị đảo âm bản; đã tự đảo lại.")
            if qa["vung_rong"]:
                canh_bao.append("Không tách được vùng: " + ", ".join(TEN_VUNG[v] for v in qa["vung_rong"]) + ".")
            if tinh in TINH_IT_PHIM:
                canh_bao.append(f"Dữ liệu học gần như không có ca có phim ở {TINH_CO_DAU[tinh]}; kết quả kèm ảnh ở tỉnh này kém tin cậy.")
        else:
            regf = area = None; co = np.array([False])
        k = self.diem(df, regf, area, co)
        cy = None
        if co[0]:
            a = k["chu_y"][0]; a = a / a.sum() if a.sum() > 0 else a
            cy = {v: float(x) for v, x in zip(REG, a)}
        return dict(
            nhan="ketqua", xac_suat=float(k["p"][0]), xac_suat_theo_fold=[float(x) for x in k["p_fold"][:, 0]],
            co_anh=bool(co[0]), tinh=tinh, chu_y_vung=cy, qa_anh=qa, canh_bao=canh_bao,
            mo_hinh="PG (Graph Transformer ảnh 6 vùng + lâm sàng), khoá 07/11/2026", experimental=True, disclaimer=DISCLAIMER)
