"""标定与对焦伺服：物面位置、变倍鼓无焦间隔、传感器（含 NIR 垫片）位置、对焦求解。"""
import numpy as np
from . import optsim as O

def _slope(P, D, axis_x):
    x = P[:, 0] - axis_x
    return np.cov(x, D[:, 0])[0, 1] / np.var(x)

def find_object_z(cfg, lam=0.55):
    """找使 CMO 出射为平行光的物点 z（参考波长）。"""
    ch = O.Channel(+1, 1.0, cfg)
    def k(z0):
        h = np.linspace(-0.9, 0.9, 9) * cfg["stop_diam"] / 2
        tgt = np.stack([ch.cx + h, np.zeros_like(h), np.full_like(h, ch.z_cmo_front)], 1)
        P = np.tile([0, 0, z0], (len(h), 1)).astype(float)
        D = tgt - P; D /= np.linalg.norm(D, axis=1, keepdims=True)
        Pf, Df, *_ = O.trace(ch.cmo, P, D, lam)
        return _slope(Pf, Df, ch.cx)   # 通道子孔径内准直（保证变倍齐焦）
    a, b = 0.0, 5.0
    ka, kb = k(a), k(b)
    for _ in range(30):
        c = b - kb * (b - a) / (kb - ka)
        a, ka, b, kb = b, kb, c, k(c)
        if abs(kb) < 1e-12:
            break
    return b

def tune_zoom(cfg, lam=0.55):
    """调伽利略望远镜两组间隔使其在参考波长严格无焦。"""
    gaps = {}
    for g, pair in O.ZOOM.items():
        if pair is None:
            continue
        def k(gap):
            c = dict(cfg); c["zoom_gap"] = {g: gap}
            ch = O.Channel(+1, g, c)
            h = np.linspace(-2, 2, 9)
            P = np.stack([ch.cx + h, np.zeros_like(h), np.full_like(h, ch.stop[2])], 1)
            D = np.tile([0, 0, 1.0], (len(h), 1))
            Pf, Df, *_ = O.trace(ch.zoom, P, D, lam)
            return _slope(Pf, Df, ch.cx)
        from scipy.optimize import brentq
        grid = np.arange(1.0, 70.0, 0.5)
        vals = np.array([k(x) for x in grid])
        b = None
        for i in range(len(grid) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
                b = brentq(k, grid[i], grid[i + 1], xtol=1e-9)
                break
        if b is None:
            raise RuntimeError(f"zoom {g}: 找不到无焦间隔")
        gaps[g] = b
    return gaps

def setup(cfg_in=None, lam_vis=0.55, lam_nir=0.83):
    cfg = dict(O.DEFAULT if cfg_in is None else cfg_in)
    cfg["z_obj0"] = find_object_z(cfg, lam_vis)
    cfg["zoom_gap"] = tune_zoom(cfg, lam_vis)
    cfg["sensors"] = {}
    for side in (-1, +1):
        ch = O.Channel(side, 1.0, cfg)
        obj = [0, 0, cfg["z_obj0"]]
        for path, lam in (("vis", lam_vis), ("nir", lam_nir)):
            r = ch.run(obj, lam, path)
            ic = (r["grid"] ** 2) // 2
            a = r["D"][ic] / np.linalg.norm(r["D"][ic])
            s, fp, _ = O.best_focus(r, a, r["P"][ic])
            u = np.array([1.0, 0, 0]); u -= (u @ a) * a; u /= np.linalg.norm(u)
            v = np.cross(a, u)
            cfg["sensors"][(side, path)] = (fp, a, u, v)   # NIR 传感器位置 = 一次性垫片
    return cfg

def defocus(cfg, side, gamma, obj, lam, path, tube_shift=0.0, cmo_shift=0.0, n=31):
    ch = O.Channel(side, gamma, cfg)
    p0, a, u, v = cfg["sensors"][(side, path)]
    r = ch.run(obj, lam, path, tube_shift, cmo_shift, n)
    s, fp, rms = O.best_focus(r, a, p0)
    return s, r, ch

def na_image(r, axis):
    D = r["D"][r["mask"]]
    return np.max(np.sqrt(np.maximum(1 - (D @ axis) ** 2, 0)))

def servo(cfg, side, gamma, obj, actuator="tube", lam=0.55):
    """调执行器，使可见光最佳焦面落在可见光传感器上。返回执行器位移 (mm)。"""
    def f(x):
        kw = {"tube_shift": x} if actuator == "tube" else {"cmo_shift": x}
        return defocus(cfg, side, gamma, obj, lam, "vis", **kw)[0]
    a, b = 0.0, 0.05
    fa, fb = f(a), f(b)
    for _ in range(30):
        c = b - fb * (b - a) / (fb - fa)
        a, fa, b, fb = b, fb, c, f(c)
        if abs(fb) < 1e-7:
            break
    return b
