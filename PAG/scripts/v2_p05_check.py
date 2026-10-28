"""Kiem tra lai ghep khoa long P0.5 — chi in so dem."""
import re, unicodedata
import numpy as np, pandas as pd
from v2_lib import *

core = load_core()
xl = pd.read_excel(XLSX, usecols=["id", "hoten", "namsinh", "gioitinh", "tinh"]).set_index("id").loc[core.id].reset_index()
def norm(s):
    s = unicodedata.normalize("NFD", str(s).lower()); s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z ]", "", s.replace("đ", "d")).strip()
nm = xl.hoten.map(norm)
print("ten rong/nan:", int((nm.isin(["", "nan"])).sum()), "| namsinh thieu:", int(xl.namsinh.isna().sum()))
k = pd.DataFrame(dict(key=nm + "|" + xl.namsinh.astype(str) + "|" + xl.gioitinh.astype(str) + "|" + xl.tinh.astype(str),
                      uid=core.uid, split=core.split, y=core.y, ok=~nm.isin(["", "nan"]) & xl.namsinh.notna()))
k = k[k.ok]
g = k.groupby("key").agg(n_uid=("uid", "nunique"), n_luot=("uid", "size"), n_split=("split", "nunique"), ymax=("y", "max"), ymin=("y", "min"))
m = g[g.n_uid > 1]
print("khoa >1 uid:", len(m), "| phan bo n_uid:", m.n_uid.value_counts().head(5).to_dict())
print("cheo dev/test:", int((m.n_split > 1).sum()), "| co duong:", int((m.ymax == 1).sum()), "| nhan khong dong nhat:", int((m.ymax != m.ymin).sum()))
print("so luot test thuoc khoa cheo:", int(k[k.key.isin(m[m.n_split > 1].index) & (k.split == "test")].shape[0]))
# uid trong cung khoa co trung tuoi nghe / nghe? kiem tra do tin: so khoa ma ten co >= 3 tu
print("khoa ten >=3 tu:", int((m.index.str.split("|").str[0].str.split().str.len() >= 3).sum()))
kd = k[k.key.isin(m.index) & (k.split == "dev")].merge(core[["uid", "fold"]].drop_duplicates("uid"), on="uid")
print("khoa trong dev nam o >1 fold:", int((kd.groupby("key").fold.nunique() > 1).sum()))
pos = k[k.key.isin(m.index)].groupby("key").apply(lambda t: (t.y == 1).sum())
print("so luot duong trong cac khoa tach:", int(pos.sum()))
