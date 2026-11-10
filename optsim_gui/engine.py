"""optsim 桌面版的计算层：在 optsim / calib 之上提供配置、标定、单点评估、光路记录、批量分析。"""
import json, copy, os
import numpy as np
from . import optsim as O
from . import calib as C

LV, LN = 0.55, 0.83
CACHE = os.path.join(os.path.expanduser("~"), "optsim", "design_cache.json")


def load_cache():
    """镜头自动设计很慢（复消色差要几分钟），结果存盘，下次启动直接用。"""
    try:
        for k, v in json.load(open(CACHE, encoding="utf-8")).items():
            f, d, lt = k.split("|", 2)
            O._DESIGN_CACHE[(float(f), float(d), lt)] = (v[0], np.array(v[1]), np.array(v[2]))
    except Exception:
        pass


def save_cache():
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        d = {f"{k[0]}|{k[1]}|{k[2]}": [list(v[0]), list(map(float, v[1])), list(map(float, v[2]))]
             for k, v in O._DESIGN_CACHE.items()}
        json.dump(d, open(CACHE, "w", encoding="utf-8"))
    except Exception:
        pass


load_cache()

DEFAULT_CFG = dict(
    WD=200.0, f_obj=200.0, cmo_diam=50.0, stereo_base=22.0, stop_diam=8.0,
    f_tube=100.0, dich_dist=40.0, plate_t=1.0, plate_glass="F_SILICA",
    cube_size=10.0, cube_glass="N-BK7", pixel=1.55e-3, sensor_w=6.287, sensor_h=4.712,
    ltype="achro_vis", splitter="cube",
    zoom=[[0.4, -40.0, 100.0], [0.625, -50.0, 80.0], [1.0, None, None],
          [1.6, 80.0, -50.0], [2.5, 100.0, -40.0]],
    custom=None,  # {"glasses":[...], "lams":[µm,...]}
)

LTYPE_NAMES = {"achro_vis": "可见光消色差 (N-BK7 + N-SF11, F/C)",
               "achro_visnir": "可见–NIR 消色差 (N-BK7 + N-SF11, 550/830)",
               "apo_visnir": "复消色差到 NIR (CaF2 + N-BK7 + N-SF11)",
               "custom": "自定义玻璃组合（来自玻璃图）"}


def custom_key(custom):
    return "custom:" + "+".join(custom["glasses"]) + "@" + ",".join(f"{l*1000:.1f}" for l in custom["lams"])


def build_cfg(user):
    """把用户配置转成 optsim 用的 cfg（并设置全局 ZOOM / LENS_TYPES）。"""
    u = copy.deepcopy(DEFAULT_CFG); u.update(user or {})
    O.ZOOM.clear()
    for g, f1, f2 in u["zoom"]:
        O.ZOOM[float(g)] = None if f1 in (None, "", 0) else (float(f1), float(f2))
    lt = u["ltype"]
    if lt == "custom":
        if not u.get("custom"):
            raise ValueError("自定义方案还没有选玻璃：先在玻璃图里选好，再点“应用到镜头设计”。")
        lt = custom_key(u["custom"])
        O.LENS_TYPES[lt] = (list(u["custom"]["glasses"]), list(u["custom"]["lams"]))
    cfg = dict(O.DEFAULT)
    for k in ("WD", "f_obj", "cmo_diam", "stereo_base", "stop_diam", "f_tube", "dich_dist",
              "plate_t", "plate_glass", "cube_size", "cube_glass", "pixel"):
        cfg[k] = u[k]
    cfg.update(cmo_type=lt, zoom_type=lt, tube_type=lt, splitter=u["splitter"])
    return u, cfg


def calibrate(user):
    u, cfg = build_cfg(user)
    cfg = C.setup(cfg, LV, LN)
    save_cache()
    return u, cfg


def zooms(cfg):
    return sorted(O.ZOOM.keys())


def sensor(cfg, side, path, trim=0.0):
    p, a, u, v = cfg["sensors"][(side, path)]
    if path == "nir":
        p = p + a * trim
    return p, a, u, v


def eval_path(cfg, side, gamma, path, obj, tube, trim, n=15):
    lam = LV if path == "vis" else LN
    ch = O.Channel(side, gamma, cfg)
    r = ch.run(obj, lam, path, tube, 0.0, n)
    p, a, u, v = sensor(cfg, side, path, trim)
    if r["mask"].sum() < 3:   # 光线全被挡掉（渐晕/全反射）
        return dict(s=float("nan"), tol=float("nan"), na=float("nan"), xy=np.zeros((0, 2)), rms=float("nan"), lam=lam)
    s, _, rms = O.best_focus(r, a, p)
    na = C.na_image(r, a)
    tol = lam * 1e-3 / (2 * na ** 2)
    xy = O.hit_plane(r, p, a, u, v)
    return dict(s=s, tol=tol, na=na, xy=xy, rms=rms, lam=lam)


def autofocus(cfg, gamma, obj):
    return C.servo(cfg, +1, gamma, obj, "tube", LV)


def trace_record(surfs, P, D, lam):
    """与 optsim.trace 相同，但记录每个面的交点（用于画光路）。"""
    P, D = P.copy(), D.copy()
    pts = [P.copy()]
    ok = np.ones(len(P), bool)
    ncur = 1.0
    for s in surfs:
        t, X, N, ok_s = s.intersect(P, D)
        ok &= ok_s & (t > -1e-9)
        if isinstance(s, O.Plane) and s.mirror:
            D = O.reflect(D, N)
        else:
            nn = O.n_glass(s.glass, lam)
            D, ok_r = O.refract(D, N, ncur, nn)
            ok &= ok_r
            ncur = nn
        P = X
        pts.append(P.copy())
    return np.stack(pts, 1), D, ok


def ray_fans(cfg, gamma, obj, tube, trim, side=+1, nfan=7):
    ch = O.Channel(side, gamma, cfg)
    cx, cy, zs, rs = ch.stop
    out = []
    for path, lam in (("vis", LV), ("nir", LN)):
        V = np.linspace(-0.95, 0.95, nfan)
        tgt = np.stack([np.full(nfan, cx), cy + rs * V, np.full(nfan, ch.z_cmo_front)], 1)
        P = np.tile(np.asarray(obj, float), (nfan, 1))
        D = tgt - P; D /= np.linalg.norm(D, axis=1, keepdims=True)
        pts, Df, ok = trace_record(ch.surfaces(path, tube), P, D, lam)
        p, a, u, v = sensor(cfg, side, path, trim)
        last = pts[:, -1, :]
        t = ((p - last) @ a) / (Df @ a)
        end = last + t[:, None] * Df
        pts = np.concatenate([pts, end[:, None, :]], 1)
        out.append((path, pts[ok]))
    return ch, out


def chroma_curve(cfg, gamma, obj, tube, trim, side=+1):
    ch = O.Channel(side, gamma, cfg)
    pv, av, *_ = sensor(cfg, side, "vis")
    lams = np.arange(450, 851, 20)
    s = []
    for l in lams:
        r = ch.run(obj, l / 1000, "vis", tube, 0.0, 11)
        s.append(O.best_focus(r, av, pv)[0] * 1000)
    r830 = ch.run(obj, LN, "vis", tube, 0.0, 11)
    sv830 = O.best_focus(r830, av, pv)[0]
    sn = eval_path(cfg, side, gamma, "nir", obj, tube, trim, 11)["s"]
    return lams, np.array(s), (sv830 - sn) * 1000


def prescription(cfg):
    """当前配置下各透镜组件的处方（右通道）。"""
    rows = []
    def lens_rows(name, surfs, f):
        for i, s in enumerate(surfs):
            if isinstance(s, O.Sphere):
                R = 1 / s.c if abs(s.c) > 1e-12 else float("inf")
                nxt = surfs[i + 1].zv if i + 1 < len(surfs) else None
                rows.append(dict(组件=name, 面=i + 1, R_mm=R, 顶点z_mm=s.zv,
                                 至下一面_mm=(nxt - s.zv) if nxt is not None else None,
                                 之后介质=s.glass, 通光半径_mm=s.r_clear, 焦距_mm=f))
    ch = O.Channel(+1, 1.0, cfg)
    lens_rows("主物镜 CMO", ch.cmo, cfg["f_obj"])
    for g, pair in O.ZOOM.items():
        if pair is None:
            continue
        chz = O.Channel(+1, g, cfg)
        n1 = len(O.design_lens(pair[0], 24 if pair[0] > 0 else 12, cfg["zoom_type"])[0]) * 2
        lens_rows(f"变倍 {g}× 第一组", chz.zoom[:n1], pair[0])
        lens_rows(f"变倍 {g}× 第二组", chz.zoom[n1:], pair[1])
    lens_rows("筒镜", ch.tube, cfg["f_tube"])
    return rows


def poly_terms(X, order):
    x, y = X[:, 0], X[:, 1]
    cols = [np.ones_like(x)]
    for o in range(1, order + 1):
        for i in range(o + 1):
            cols.append(x ** (o - i) * y ** i)
    return np.stack(cols, 1)


def poly_resid(src, dst, order, pix):
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[ok], dst[ok]
    if len(src) < poly_terms(src[:1], order).shape[1] + 2:
        return float("nan")
    A = poly_terms(src, order)
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return float(np.abs(dst - A @ M).max() / pix)


def centroid(cfg, side, g, obj, lam, path, tube, n=9):
    ch = O.Channel(side, g, cfg)
    p0, a, u, v = cfg["sensors"][(side, path)]
    r = ch.run(obj, lam, path, tube, 0.0, n)
    xy = O.hit_plane(r, p0, a, u, v)
    xy = xy[np.isfinite(xy).all(1)]
    return xy.mean(0) if len(xy) else np.array([np.nan, np.nan])


def run_analysis(user, cfg, depths=(-2.0, 0.0, 2.0), progress=None):
    """每档：对焦（筒镜）后 NIR 残余离焦、查表微调值、最优固定垫片、配准多项式残差、左右垂直视差。"""
    z0 = cfg["z_obj0"]; pix = cfg["pixel"]
    rows = []
    zs = zooms(cfg)
    total = len(zs) * (len(depths) + 1); k = 0
    for g in zs:
        rec = dict(gamma=g, focus=[])
        for dz in depths:
            obj = [0, 0, z0 + dz]
            x = autofocus(cfg, g, obj)
            e = eval_path(cfg, +1, g, "nir", obj, x, 0.0, 21)
            rec["focus"].append(dict(dz=dz, tube_um=x * 1e3, nir_um=e["s"] * 1e3, tol_um=e["tol"] * 1e3))
            k += 1
            if progress: progress(k / total, f"{g}× Δz={dz:+.1f} mm 对焦完成")
        # 配准（Δz=0）
        obj0 = [0, 0, z0]
        x0 = autofocus(cfg, g, obj0)
        xl = C.servo(cfg, -1, g, obj0, "tube", LV)
        c0 = centroid(cfg, +1, g, obj0, LV, "vis", x0)
        c1 = centroid(cfg, +1, g, [1.0, 0, z0], LV, "vis", x0)
        M = float(np.linalg.norm(c1 - c0))
        hx, hy = user["sensor_w"] / 2 / M * 0.9, user["sensor_h"] / 2 / M * 0.9
        pts = [(x, y) for x in np.linspace(-hx, hx, 5) for y in np.linspace(-hy, hy, 5)]
        V = np.array([centroid(cfg, +1, g, [p[0], p[1], z0], LV, "vis", x0) for p in pts])
        N = np.array([centroid(cfg, +1, g, [p[0], p[1], z0], LN, "nir", x0) for p in pts])
        VL = np.array([centroid(cfg, -1, g, [p[0], p[1], z0], LV, "vis", xl) for p in pts])
        rec["M"] = M
        rec["reg_nir"] = [poly_resid(N, V, o, pix) for o in (1, 2, 3)]
        ok = np.isfinite(V).all(1) & np.isfinite(VL).all(1)
        rec["valid_points"] = int(ok.sum())
        if ok.sum() >= 12:
            vd = V[ok, 1] - VL[ok, 1]
            rec["lr_raw"] = float(np.abs(vd - vd.mean()).max() / pix)
            A = poly_terms(VL[ok], 3); c, *_ = np.linalg.lstsq(A, V[ok, 1], rcond=None)
            rec["lr_o3"] = float(np.abs(V[ok, 1] - A @ c).max() / pix)
        else:
            rec["lr_raw"] = rec["lr_o3"] = float("nan")
        rec["lut_trim_um"] = [f for f in rec["focus"] if f["dz"] == 0.0][0]["nir_um"] if 0.0 in depths else None
        rows.append(rec)
        k += 1
        if progress: progress(k / total, f"{g}× 配准完成")
    # 最优固定垫片：使所有档、所有深度 |离焦-垫片|/焦深 的最大值最小
    shims = np.linspace(-3000, 3000, 12001)
    worst = np.zeros_like(shims)
    for rec in rows:
        for f in rec["focus"]:
            if np.isfinite(f["nir_um"]):
                worst = np.maximum(worst, np.abs(f["nir_um"] - shims) / f["tol_um"])
    i = int(np.argmin(worst))
    lut_worst = np.nanmax([abs(f["nir_um"] - rec["lut_trim_um"]) / f["tol_um"] for rec in rows for f in rec["focus"]])
    ref_worst = np.nanmax([abs(f["nir_um"]) / f["tol_um"] for rec in rows for f in rec["focus"]])
    nbad = sum(1 for rec in rows for f in rec["focus"] if not np.isfinite(f["nir_um"]))
    return dict(rows=rows, shim_um=float(shims[i]), shim_worst=float(worst[i]),
                ref_worst=float(ref_worst), lut_worst=float(lut_worst), n_blocked=nbad)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)


def lens_amplification(cfg):
    """当前方案下每个透镜组件的光焦度放大 Σ|φᵢ|/|φ|（越大越难做，>6 基本不可制造）。"""
    lt = cfg["cmo_type"]; out = []
    for (f, d, t), (gl, K, X) in O._DESIGN_CACHE.items():
        if t != lt:
            continue
        ph = [abs((O.n_glass(g, LV) - 1) * k) for g, k in zip(gl, K)]
        out.append((f, sum(ph) * abs(f)))
    return sorted(out)
