"""玻璃库（refractiveindex.info 数据：成都光明 / 肖特 / 小原 / 晶体）与 P–ν 计算、复消色差组合搜索。"""
import json, os, sys, itertools
import numpy as np
from . import optsim as O

def _data_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, name)
    return p if os.path.exists(p) else os.path.join(base, "optsim_gui", name)

GLASSES = json.load(open(_data_path("glasses.json"), encoding="utf-8"))
NAMES = [g["n"] for g in GLASSES]
CATS = np.array([g["cat"] for g in GLASSES])
S = np.array([g["s"] for g in GLASSES])          # B1,C1,B2,C2,B3,C3
IDX = {n: i for i, n in enumerate(NAMES)}
CAT_NAMES = {"CDGM": "成都光明", "SCHOTT": "肖特", "OHARA": "小原", "CRYSTAL": "晶体"}

# 注册进 optsim 的色散表（optsim 的顺序是 B1,B2,B3,C1,C2,C3），镜头自动设计就能用这些玻璃
for g in GLASSES:
    s = g["s"]
    O.SELLMEIER.setdefault(g["n"], (s[0], s[2], s[4], s[1], s[3], s[5]))


def n_of(idx, lam):
    """idx: 索引数组；lam: µm（标量或数组）→ 形状 (len(idx), len(lam))"""
    s = S[np.atleast_1d(idx)][:, :, None]
    l2 = np.atleast_1d(lam)[None, :] ** 2
    return np.sqrt(1 + s[:, 0] * l2 / (l2 - s[:, 1]) + s[:, 2] * l2 / (l2 - s[:, 3]) + s[:, 4] * l2 / (l2 - s[:, 5]))


def coords(lams, idx=None):
    idx = np.arange(len(NAMES)) if idx is None else np.asarray(idx)
    n = n_of(idx, np.asarray(lams))
    a, b, c = n[:, 0], n[:, 1], n[:, 2]
    return (b - 1) / (a - c), (a - b) / (a - c)


def normal_line(nu, P, mask):
    x, y = nu[mask], P[mask]
    b, a = np.polyfit(x, y, 1)
    r = y - (a + b * x); sd = r.std()
    keep = np.abs(r) < 2 * sd
    b, a = np.polyfit(x[keep], y[keep], 1)
    return a, b


def solve(names, lams, f):
    idx = [IDX[n] for n in names]
    phi = 1.0 / f
    if len(idx) == 1:
        return np.array([phi / (n_of(idx, lams[1])[0, 0] - 1)])
    L = list(lams) if len(idx) == 3 else [lams[0], lams[2]]
    A = n_of(idx, np.array(L)).T - 1
    if abs(np.linalg.det(A)) < 1e-14:
        return None
    return np.linalg.solve(A, np.full(len(L), phi))


def shift_curve(names, K, lams, grid):
    idx = [IDX[n] for n in names]
    p = ((n_of(idx, grid) - 1) * np.asarray(K)[:, None]).sum(0)
    p0 = ((n_of(idx, lams[1])[:, 0] - 1) * K).sum()
    return 1 / p - 1 / p0


def ref_curve(lams, grid, f):
    names = ["H-K9L", "H-ZF13"]
    idx = [IDX[n] for n in names]
    A = n_of(idx, np.array([0.48613, 0.65627])).T - 1
    K = np.linalg.solve(A, np.full(2, 1 / f))
    return shift_curve(names, K, lams, grid)


def search(mask, lams, f, amp_lim, top=10, ncand=(45, 18, 18)):
    """在 mask 选中的玻璃里搜索三玻璃组合：放大系数 ≤ amp_lim 时 450–850 nm 焦移峰谷最小。"""
    idx_all = np.nonzero(mask)[0]
    nu, P = coords(lams, idx_all)
    a, b = normal_line(nu, P, CATS[idx_all] != "CRYSTAL")
    dev = P - (a + b * nu)
    pick = set(idx_all[np.argsort(-np.abs(dev))[:ncand[0]]])
    pick |= set(idx_all[np.argsort(nu)[:ncand[1]]]) | set(idx_all[np.argsort(-nu)[:ncand[2]]])
    cand = np.array(sorted(pick))
    grid = np.arange(0.45, 0.8501, 0.01)
    D = n_of(cand, np.asarray(lams)) - 1          # (c,3)
    G = n_of(cand, grid) - 1                       # (c,g)
    Mid = D[:, 1]
    T = np.array(list(itertools.combinations(range(len(cand)), 3)))
    A = np.stack([D[T[:, 0]], D[T[:, 1]], D[T[:, 2]]], 2)   # (t,3λ,3玻璃)
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-14
    T, A = T[ok], A[ok]
    phi = 1 / f
    K = np.linalg.solve(A, np.full((len(T), 3, 1), phi))[:, :, 0]
    amp = np.abs(Mid[T] * K).sum(1) / phi
    keep = amp <= amp_lim
    T, K, amp = T[keep], K[keep], amp[keep]
    if len(T) == 0:
        return []
    pw = (G[T] * K[:, :, None]).sum(1)              # (t,g)
    fl = 1 / pw
    pv = (fl.max(1) - fl.min(1)) * 1000
    order = np.argsort(pv)[:top]
    return [dict(names=[NAMES[cand[j]] for j in T[i]], pv_um=float(pv[i]), amp=float(amp[i])) for i in order]
