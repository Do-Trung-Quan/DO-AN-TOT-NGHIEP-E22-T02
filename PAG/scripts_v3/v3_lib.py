"""Thu vien PG ban 3 (PG_update 17/10): dung lai v2_lib, ghi ket qua vao PAG/results_v3, cache vao PAG/cache_v3.
Cache chua id + anh thu nho -> PII, KHONG dua len GitHub. Chi in/ghi so lieu gop."""
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_v2"))
import v2_lib
from v2_lib import *                                   # noqa: F401,F403
CACHE3 = PAG / "cache_v3"; CACHE3.mkdir(exist_ok=True)
RES3 = PAG / "results_v3"; RES3.mkdir(exist_ok=True)
v2_lib.RES = RES3                                      # Report() ghi vao results_v3
QA_DIR = PAG / "qa_v3_KHONG_CHIA_SE"
RES2 = PAG / "results_v2"


def _sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""): h.update(b)
    return h.hexdigest()[:16]


def run_config(folder, script, data=(), **params):
    """Ghi run_config.json canh BAO_CAO.md (PGS 27/10: ghi seed/config moi lan chay).
    Gom: ngay chay, hash script + thu vien, hash file du lieu dau vao, seed, phien ban thu vien, tham so rieng."""
    import json, platform, datetime, sklearn
    d = RES3 / folder; d.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    cfg = dict(ngay_chay=datetime.datetime.now().isoformat(timespec="seconds"), script=Path(script).name,
               hash_script={p.name: _sha(p) for p in (Path(script).resolve(), here / "v3_lib.py", here.parent / "scripts_v2" / "v2_lib.py")},
               hash_du_lieu={Path(p).name: _sha(p) for p in data}, SEED=SEED, NBOOT=NBOOT,
               phien_ban=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
               tham_so=params)
    (d / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return cfg


def anon(ids):
    """Ma an danh cho so kiem toan: hash co muoi (muoi luu cuc bo trong cache_v3, khong chia se)."""
    f = CACHE3 / "salt.txt"
    if not f.exists(): f.write_text(hashlib.sha256(str(np.random.default_rng().random()).encode()).hexdigest())
    s = f.read_text()
    return ["A" + hashlib.sha256((s + str(i)).encode()).hexdigest()[:8] for i in ids]
