"""Kien truc PG (Graph Transformer 14 node) — CHEP NGUYEN VAN tu PAG/scripts/pag_lib.py (GTLayer, PGNet, base_types 'anat').
Khong sua: trong so luu o mo_hinh_PG_web/fold*/seed*.pt duoc nap strict=True vao dung cac lop nay.

Node: 0-5 anh 6 vung (RU RM RL LU LM LL), 6-11 sau nhom lam sang, 12 boi canh (tinh), 13 CLS.
Canh: E1 doc trong mot phoi, E2 ngang cung tang, E3 giua nhom lam sang, E4 lam sang <-> vung, E5 CLS <-> moi node.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NONE, SELF, E1, E2, E3, E4, E5 = range(7)
N_ET = 7
E1_PAIRS = [(0, 1), (1, 2), (3, 4), (4, 5)]
E2_PAIRS = [(0, 3), (1, 4), (2, 5)]
CLIN_SLOTS = range(6, 12)


def base_types(n_img=6):
    """pag_lib.base_types(edges='anat', n_img=6)."""
    T = np.zeros((14, 14), np.int64)

    def s(i, j, t):
        T[i, j] = T[j, i] = t
    for i in range(14):
        T[i, i] = SELF
    for i in range(13):
        s(13, i, E5)
    if n_img == 6:
        for a, b in E1_PAIRS:
            s(a, b, E1)
        for a, b in E2_PAIRS:
            s(a, b, E2)
    for a in CLIN_SLOTS:
        for b in CLIN_SLOTS:
            if a != b:
                s(a, b, E3)
        for b in range(n_img):
            s(a, b, E4)
    return T


class GTLayer(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__(); s.h = h
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.eb = nn.Embedding(N_ET, h); nn.init.zeros_(s.eb.weight)
        s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        s.drop = nn.Dropout(drop)

    def forward(s, x, et, allow):
        B, n, d = x.shape; dh = d // s.h
        q, k, v = s.qkv(s.ln1(x)).view(B, n, 3, s.h, dh).permute(2, 0, 3, 1, 4)
        a = q @ k.transpose(-1, -2) / math.sqrt(dh) + s.eb(et).permute(0, 3, 1, 2)
        a = a.masked_fill(~allow[:, None], float("-inf")).softmax(-1)
        x = x + s.drop(s.o((s.drop(a) @ v).transpose(1, 2).reshape(B, n, d)))
        return x + s.drop(s.ff(s.ln2(x))), a


class PGNet(nn.Module):
    def __init__(s, d_img, n_img, clin_dims, d=64, layers=2, drop=0.1, heads=4, arch="gt"):
        super().__init__(); s.n_img = n_img
        assert arch == "gt"
        s.img = nn.Sequential(nn.Dropout(drop), nn.Linear(d_img + 1, d)); s.pos = nn.Embedding(7, d)
        s.clin = nn.ModuleList([nn.Sequential(nn.Linear(k, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, d)) for k in clin_dims[:6]])
        s.ctx = nn.Linear(clin_dims[6], d)
        s.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(s.cls, std=0.02)
        s.layers = nn.ModuleList([GTLayer(d, heads, drop) for _ in range(layers)])
        s.ln = nn.LayerNorm(d); s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(s, img, area, clin, present, et):
        B = img.shape[0]; d = s.cls.shape[-1]
        x = img.new_zeros(B, 14, d)
        pos_ids = torch.arange(s.n_img, device=img.device) if s.n_img == 6 else torch.tensor([6], device=img.device)
        x[:, :s.n_img] = s.img(torch.cat([img, area[..., None]], -1)) + s.pos(pos_ids)
        for g in range(6):
            x[:, 6 + g] = s.clin[g](clin[g])
        x[:, 12] = s.ctx(clin[6]); x[:, 13] = s.cls[:, 0]
        allow = (present[:, None, :] & (et != NONE)) | torch.eye(14, dtype=torch.bool, device=img.device)[None]
        att = None
        for L in s.layers:
            x, att = L(x, et, allow)
        h = s.ln(x[:, 13])
        return s.head(h)[:, 0], h, att
