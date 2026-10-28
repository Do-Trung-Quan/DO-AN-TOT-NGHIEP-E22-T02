"""CheXFound (DIAL-RPI, HuggingFace DIAL-RPI/CheXFound, Apache-2.0) dung lai bang PyTorch thuan tu model.safetensors.
Kien truc doc tu trong so: ViT-L/16 kieu DINOv2 (iBOT, highres512), 24 khoi, 16 head, 4 register token, LayerScale, MLP SwiGLU (w12/w3),
pos_embed 1+32x32 -> anh 512x512 cho luoi patch 32x32. Chuan hoa dau vao theo mac dinh DINOv2 (ImageNet mean/std, xam lap 3 kenh).
"""
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file

WEIGHTS = Path(__file__).resolve().parents[1] / "chexfound" / "model.safetensors"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Attn(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
    def forward(s, x):
        B, N, C = x.shape
        q, k, v = s.qkv(x).reshape(B, N, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        return s.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class SwiGLU(nn.Module):
    def __init__(s, d, hid):
        super().__init__(); s.w12 = nn.Linear(d, 2 * hid); s.w3 = nn.Linear(hid, d)
    def forward(s, x):
        a, b = s.w12(x).chunk(2, -1); return s.w3(F.silu(a) * b)


class LS(nn.Module):
    def __init__(s, d):
        super().__init__(); s.weight = nn.Parameter(torch.ones(d))
    def forward(s, x): return x * s.weight


class Block(nn.Module):
    def __init__(s, d, h, hid):
        super().__init__()
        s.norm1, s.attn, s.ls1 = nn.LayerNorm(d, eps=1e-6), Attn(d, h), LS(d)
        s.norm2, s.mlp, s.ls2 = nn.LayerNorm(d, eps=1e-6), SwiGLU(d, hid), LS(d)
    def forward(s, x):
        x = x + s.ls1(s.attn(s.norm1(x))); return x + s.ls2(s.mlp(s.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(s, d, p):
        super().__init__(); s.proj = nn.Conv2d(3, d, p, p)
    def forward(s, x): return s.proj(x)


class CheXFound(nn.Module):
    def __init__(s, d=1024, depth=24, h=16, p=16, n_reg=4, hid=2736, grid=32):
        super().__init__()
        s.p, s.n_reg, s.grid = p, n_reg, grid
        s.patch_embed = PatchEmbed(d, p)
        s.cls_token = nn.Parameter(torch.zeros(1, 1, d)); s.mask_token = nn.Parameter(torch.zeros(1, d))
        s.register_tokens = nn.Parameter(torch.zeros(1, n_reg, d))
        s.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, d))
        s.blocks = nn.ModuleList([Block(d, h, hid) for _ in range(depth)]); s.norm = nn.LayerNorm(d, eps=1e-6)

    def forward(s, x):
        """x: (B,3,H,W) da chuan hoa. Tra ve (cls (B,d), ban do patch (B,d,H/p,W/p)) sau LayerNorm cuoi."""
        B, _, H, W = x.shape; gh, gw = H // s.p, W // s.p
        t = s.patch_embed(x).flatten(2).transpose(1, 2)
        pos = s.pos_embed
        if (gh, gw) != (s.grid, s.grid):
            pp = pos[:, 1:].reshape(1, s.grid, s.grid, -1).permute(0, 3, 1, 2)
            pp = F.interpolate(pp.float(), size=(gh, gw), mode="bicubic", align_corners=False).to(pos.dtype)
            pos = torch.cat([pos[:, :1], pp.flatten(2).transpose(1, 2)], 1)
        t = torch.cat([s.cls_token.expand(B, -1, -1), t], 1) + pos
        t = torch.cat([t[:, :1], s.register_tokens.expand(B, -1, -1), t[:, 1:]], 1)
        for b in s.blocks: t = b(t)
        t = s.norm(t)
        return t[:, 0], t[:, 1 + s.n_reg:].transpose(1, 2).reshape(B, -1, gh, gw)


def load(device="cuda"):
    m = CheXFound()
    sd = {k[len("model."):]: v for k, v in load_file(str(WEIGHTS)).items()}
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def prep(img_uint8, device="cuda"):
    """anh xam uint8 (H,W) -> tensor (1,3,H,W) chuan hoa kieu ImageNet."""
    x = torch.as_tensor(img_uint8, dtype=torch.float32, device=device)[None, None].expand(1, 3, -1, -1) / 255.0
    return (x - torch.tensor(MEAN, device=device)[None, :, None, None]) / torch.tensor(STD, device=device)[None, :, None, None]
