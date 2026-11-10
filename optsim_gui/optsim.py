"""
optsim —— 手术显微镜 可见光/荧光 四路成像的光学仿真器（最小可用版）

功能
  * Sellmeier 玻璃色散库
  * 3D 序列光线追迹：球面、任意朝向平面、反射镜（二向色分光面），累计光程
  * 多片薄透镜消色差/复消色差设计（任意校正波长），再生成厚透镜真实曲面
  * 共同主物镜 (CMO) + 离轴立体通道 + 伽利略变倍鼓 + 管镜 + 45° 二向色平板
  * 分析：最佳焦面（几何最小弥散圆，闭式解）、质心像点、波前 OPD (Hopkins 近似)、
          衍射 MTF（光瞳自相关）

单位：长度 mm，波长 µm。坐标：z 为显微镜光轴（物 → 仪器），物面在 z≈0 附近。
"""
import numpy as np

# ------------------------------------------------------------------ 玻璃
SELLMEIER = {   # B1,B2,B3,C1,C2,C3  (λ in µm, C in µm²)
    "N-BK7":  (1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653),
    "N-SF11": (1.73759695, 0.313747346, 1.89878101, 0.013188707, 0.0623068142, 155.23629),
    "F_SILICA": (0.6961663, 0.4079426, 0.8974794, 0.0046791483, 0.0135120631, 97.9340025),
    "CAF2":   (0.5675888, 0.4710914, 3.8484723, 0.050263605**2, 0.1003909**2, 34.649040**2),
}

def n_glass(name, lam_um):
    if name == "AIR":
        return 1.0
    B1, B2, B3, C1, C2, C3 = SELLMEIER[name]
    l2 = lam_um ** 2
    return np.sqrt(1 + B1 * l2 / (l2 - C1) + B2 * l2 / (l2 - C2) + B3 * l2 / (l2 - C3))

# ------------------------------------------------------------------ 面
class Sphere:
    """轴平行于 z 的球面。vertex=(ax,ay,zv)，曲率 c=1/R，之后介质 glass。"""
    def __init__(self, ax, ay, zv, c, glass, r_clear=1e9, name=""):
        self.ax, self.ay, self.zv, self.c, self.glass, self.r_clear, self.name = ax, ay, zv, c, glass, r_clear, name
    def shifted(self, dz):
        return Sphere(self.ax, self.ay, self.zv + dz, self.c, self.glass, self.r_clear, self.name)
    def intersect(self, P, D):
        p = P - np.array([self.ax, self.ay, self.zv])
        c = self.c
        A = c
        B = 2 * (c * np.sum(p * D, 1) - D[:, 2])
        C = c * np.sum(p * p, 1) - 2 * p[:, 2]
        if abs(c) < 1e-12:
            t = -C / B
        else:
            disc = np.sqrt(np.maximum(B * B - 4 * A * C, 0))
            t1 = (-B - disc) / (2 * A)
            t2 = (-B + disc) / (2 * A)
            # 取靠近顶点的那半个球帽 (1 - c*qz > 0)
            v1 = (1 - c * (p[:, 2] + t1 * D[:, 2])) > 0
            t = np.where(v1, t1, t2)
        X = P + t[:, None] * D
        q = X - np.array([self.ax, self.ay, self.zv])
        N = np.stack([c * q[:, 0], c * q[:, 1], c * q[:, 2] - 1], 1)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        ok = (q[:, 0] ** 2 + q[:, 1] ** 2) <= self.r_clear ** 2
        return t, X, N, ok

class Plane:
    """任意朝向平面：过点 p0、法向 nrm；mirror=True 时为反射面。"""
    def __init__(self, p0, nrm, glass="AIR", mirror=False, name=""):
        self.p0 = np.asarray(p0, float)
        n = np.asarray(nrm, float)
        self.nrm = n / np.linalg.norm(n)
        self.glass, self.mirror, self.name = glass, mirror, name
    def shifted(self, dz):
        return Plane(self.p0 + np.array([0, 0, dz]), self.nrm, self.glass, self.mirror, self.name)
    def intersect(self, P, D):
        t = np.sum((self.p0 - P) * self.nrm, 1) / np.sum(D * self.nrm, 1)
        X = P + t[:, None] * D
        N = np.tile(self.nrm, (len(P), 1))
        return t, X, N, np.ones(len(P), bool)

def refract(D, N, n1, n2):
    cosi = -np.sum(D * N, 1)
    flip = cosi < 0
    N = np.where(flip[:, None], -N, N)
    cosi = np.abs(cosi)
    mu = n1 / n2
    k = 1 - mu ** 2 * (1 - cosi ** 2)
    Dn = mu * D + (mu * cosi - np.sqrt(np.maximum(k, 0)))[:, None] * N
    return Dn / np.linalg.norm(Dn, axis=1, keepdims=True), k >= 0

def reflect(D, N):
    Dn = D - 2 * np.sum(D * N, 1)[:, None] * N
    return Dn / np.linalg.norm(Dn, axis=1, keepdims=True)

def trace(surfs, P, D, lam, n0="AIR"):
    """序列追迹。返回最终位置、方向、累计光程、有效掩码、最终介质折射率。"""
    P, D = P.copy(), D.copy()
    opl = np.zeros(len(P))
    ok = np.ones(len(P), bool)
    ncur = n_glass(n0, lam)
    for s in surfs:
        t, X, N, ok_s = s.intersect(P, D)
        opl += ncur * t
        ok &= ok_s & (t > -1e-9)
        if isinstance(s, Plane) and s.mirror:
            D = reflect(D, N)
        else:
            nn = n_glass(s.glass, lam)
            D, ok_r = refract(D, N, ncur, nn)
            ok &= ok_r
            ncur = nn
        P = X
    return P, D, opl, ok, ncur

# ------------------------------------------------------------------ 设计
def solve_multiplet(glasses, phi, lams):
    """薄透镜多片组合：使总光焦度在给定若干波长上都等于 phi。返回每片的 K=1/R1-1/R2。"""
    A = np.array([[n_glass(g, l) - 1 for g in glasses] for l in lams])
    return np.linalg.solve(A, np.full(len(lams), phi))

def build_multiplet(ax, ay, z0, glasses, K, diam, gap=0.3):
    """每片等曲率（双凸/双凹），空气间隔 gap。返回面列表、总长。"""
    surfs, z = [], z0
    r = diam / 2
    for g, k in zip(glasses, K):
        R = 2.0 / k
        sag = R - np.sign(R) * np.sqrt(max(R * R - r * r, 0)) if abs(R) > r else r
        tc = (2 * abs(sag) + 1.0) if k > 0 else 1.5
        surfs.append(Sphere(ax, ay, z, 1 / R, g, r))
        surfs.append(Sphere(ax, ay, z + tc, -1 / R, "AIR", r))
        z += tc + gap
    return surfs, z - gap

LENS_TYPES = {   # 不同校正方案：玻璃组合 + 校正波长
    "achro_vis": (["N-BK7", "N-SF11"], [0.4861, 0.6563]),           # 普通可见光消色差 (F,C)
    "achro_visnir": (["N-BK7", "N-SF11"], [0.5500, 0.8300]),        # 可见–近红外消色差
    "apo_visnir": (["CAF2", "N-BK7", "N-SF11"], [0.4600, 0.6000, 0.8300]),  # 复消色差到 NIR
}

def _sag(c, r):
    return c * r * r / (1 + np.sqrt(max(1 - c * c * r * r, 1e-9)))

def build_bent(ax, ay, z0, glasses, K, X, diam, reverse=False):
    """每片用光焦度参数 K=c1-c2 和弯曲因子 X=(c1+c2)/K 描述；reverse=True 把整组倒装。"""
    els = list(zip(glasses, K, X))
    if reverse:
        els = [(g, k, -x) for g, k, x in els[::-1]]
    r = diam / 2
    surfs, z, prev_c2 = [], z0, None
    for g, k, x in els:
        c1, c2 = k * (1 + x) / 2, k * (x - 1) / 2
        if prev_c2 is not None:
            z += max(0.3, _sag(prev_c2, r) - _sag(c1, r) + 0.2)
        tc = max(1.0 + _sag(c1, r) - _sag(c2, r), 1.2)
        surfs.append(Sphere(ax, ay, z, c1, g, r))
        surfs.append(Sphere(ax, ay, z + tc, c2, "AIR", r))
        z += tc
        prev_c2 = c2
    return surfs, z

_DESIGN_CACHE = {}

def design_lens(f, diam, ltype, aperture_frac=0.9, lam0=0.55):
    """数值优化：EFL=f、各校正波长焦点重合、全口径球差最小（平行光从前方入射）。"""
    key = (f, diam, ltype)
    if key in _DESIGN_CACHE:
        return _DESIGN_CACHE[key]
    from scipy.optimize import minimize
    glasses, lams = LENS_TYPES[ltype]
    K0 = solve_multiplet(glasses, 1.0 / f, lams)
    nel = len(glasses)
    r = diam / 2 * aperture_frac
    h = np.linspace(-r, r, 21)
    P = np.stack([np.zeros_like(h), h, np.full_like(h, -5.0)], 1)
    D = np.tile([0, 0, 1.0], (len(h), 1))
    Pp = np.array([[0, 1e-3 * f, -5.0]])

    def focus(surfs, lam):
        Pf, Df, *_ = trace(surfs, P, D, lam)
        # 最小 RMS 弥散面（只有 y 方向）
        A = Pf[:, 1] - Pf[:, 2] * Df[:, 1] / Df[:, 2]
        B = Df[:, 1] / Df[:, 2]
        dA, dB = A - A.mean(), B - B.mean()
        s = -np.sum(dA * dB) / np.sum(dB * dB)
        return s, np.sqrt(np.mean((dA + s * dB) ** 2))

    def efl(surfs, lam):
        Pf, Df, *_ = trace(surfs, Pp, np.array([[0, 0, 1.0]]), lam)
        return -Pp[0, 1] / (Df[0, 1] / Df[0, 2])

    def cost(v):
        K, X = v[:nel] * K0, v[nel:]
        try:
            s, _ = build_bent(0, 0, 0, glasses, K, X, diam)
            e = efl(s, lam0) / f - 1
            fz = [focus(s, l)[0] for l in lams]
            _, rms = focus(s, lam0)
        except Exception:
            return 1e9
        val = (e * 1e3) ** 2 + sum(((z - fz[0]) / (f * 1e-5)) ** 2 for z in fz[1:]) + (rms / (f * 2e-6)) ** 2
        return val if np.isfinite(val) else 1e9

    best = None
    for X0 in ([0.0] * nel, [1.0] + [-1.0] * (nel - 1), [-0.5] * nel):
        v0 = np.r_[np.ones(nel), X0]
        res = minimize(cost, v0, method="Nelder-Mead",
                       options=dict(maxiter=6000, maxfev=6000, xatol=1e-7, fatol=1e-9))
        if best is None or res.fun < best.fun:
            best = res
    K, X = best.x[:nel] * K0, best.x[nel:]
    _DESIGN_CACHE[key] = (glasses, K, X)
    return glasses, K, X

def lens(ax, ay, z0, f, diam, ltype, reverse=False):
    glasses, K, X = design_lens(f, diam, ltype)
    return build_bent(ax, ay, z0, glasses, K, X, diam, reverse)

# ------------------------------------------------------------------ 系统
ZOOM = {0.4: (-40.0, 100.0), 0.625: (-50.0, 80.0), 1.0: None, 1.6: (80.0, -50.0), 2.5: (100.0, -40.0)}

class Channel:
    """一个立体通道（side=-1 左 / +1 右）的完整光路。"""
    def __init__(self, side, gamma, cfg):
        self.side, self.gamma, self.cfg = side, gamma, cfg
        cx = side * cfg["stereo_base"] / 2
        self.cx = cx
        z = cfg["WD"]
        # 共同主物镜 (CMO)，轴在 x=0
        self.cmo, zc = lens(0, 0, z, cfg["f_obj"], cfg["cmo_diam"], cfg["cmo_type"], reverse=True)
        self.z_cmo_front = z
        z = zc + 5.0
        self.stop = (cx, 0.0, z, cfg["stop_diam"] / 2)
        z += 10.0
        # 变倍鼓（伽利略望远镜），占用固定槽位 [z, z+60]，之后的光路与倍率无关
        self.zoom = []
        if ZOOM[gamma] is not None:
            f1, f2 = ZOOM[gamma]
            s1, ze = lens(cx, 0, z + 5.0, f1, 24 if f1 > 0 else 12, cfg["zoom_type"])
            gap = cfg.get("zoom_gap", {}).get(gamma, f1 + f2 - 1.0)
            s2, ze2 = lens(cx, 0, ze + gap, f2, 24 if f2 > 0 else 12, cfg["zoom_type"], reverse=True)
            self.zoom = s1 + s2
        z += 80.0
        # 管镜（对焦执行器推它）
        self.tube, zt = lens(cx, 0, z, cfg["f_tube"], 20, cfg["tube_type"])
        self.z_tube_back = zt
        # 二向色平板：在管镜后 dich_dist 处，45°，可见光反射 (+y)，NIR 透射
        zd = zt + cfg["dich_dist"]
        a = 1 / np.sqrt(2)
        nrm = np.array([0, a, -a])
        self.dich_front = Plane([cx, 0, zd], nrm, "AIR", mirror=True, name="dichroic")
        tpl = cfg["plate_t"]
        self.plate = [Plane([cx, 0, zd], nrm, cfg["plate_glass"]),
                      Plane(np.array([cx, 0, zd]) + tpl * (-nrm), nrm, "AIR")]
        if cfg.get("splitter", "plate") == "cube":
            L, gl = cfg.get("cube_size", 10.0), cfg.get("cube_glass", "N-BK7")
            entry = Plane([cx, 0, zd - L / 2], [0, 0, 1], gl)
            hyp = Plane([cx, 0, zd], nrm, gl, mirror=True)
            self.cube_vis = [entry, hyp, Plane([cx, L / 2, zd], [0, 1, 0], "AIR")]
            self.cube_nir = [entry, Plane([cx, 0, zd + L / 2], [0, 0, 1], "AIR")]
        self.sensor = {}   # 'vis' / 'nir' -> (point, axis, u, v)

    def surfaces(self, path, tube_shift=0.0, cmo_shift=0.0):
        cmo = [s.shifted(cmo_shift) for s in self.cmo]
        tube = [s.shifted(tube_shift) for s in self.tube]
        base = cmo + self.zoom + tube
        if self.cfg.get("splitter", "plate") == "cube":
            return base + (self.cube_vis if path == "vis" else self.cube_nir)
        return base + ([self.dich_front] if path == "vis" else self.plate)

    def rays(self, obj, lam, n=31, cmo_shift=0.0):
        """从物点发出、瞄准通道光阑的规则网格光线。"""
        cx, cy, zs, rs = self.stop
        g = np.linspace(-1, 1, n)
        U, V = np.meshgrid(g, g)
        tgt = np.stack([cx + rs * 1.05 * U.ravel(), cy + rs * 1.05 * V.ravel(),
                        np.full(U.size, self.z_cmo_front + cmo_shift)], 1)
        P = np.tile(np.asarray(obj, float), (len(tgt), 1))
        D = tgt - P
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        return P, D, U.ravel(), V.ravel()

    def run(self, obj, lam, path, tube_shift=0.0, cmo_shift=0.0, n=31):
        P, D, U, V = self.rays(obj, lam, n, cmo_shift)
        surfs = self.surfaces(path, tube_shift, cmo_shift)
        # 先追到光阑做渐晕裁剪
        n_cmo = len(self.cmo)
        Ps, Ds, _, ok1, _ = trace(surfs[:n_cmo], P, D, lam)
        cx, cy, zs, rs = self.stop
        t = (zs - Ps[:, 2]) / Ds[:, 2]
        Xs = Ps + t[:, None] * Ds
        in_stop = (Xs[:, 0] - cx) ** 2 + (Xs[:, 1] - cy) ** 2 <= rs ** 2
        Pf, Df, opl, ok, nfin = trace(surfs, P, D, lam)
        m = ok & ok1 & in_stop
        return dict(P=Pf, D=Df, opl=opl, mask=m, U=U, V=V, n=nfin, grid=n)

# ------------------------------------------------------------------ 分析
def best_focus(r, axis, ref_point):
    """沿 axis 方向的几何最佳焦面（最小 RMS 弥散），返回相对 ref_point 的有符号距离 s*、焦点处质心。"""
    P, D, m = r["P"][r["mask"]], r["D"][r["mask"]], None
    a = axis / np.linalg.norm(axis)
    Pr = P - ref_point
    pa = Pr @ a
    da = D @ a
    Pp = Pr - pa[:, None] * a
    Dp = D - da[:, None] * a
    A = Pp - (pa / da)[:, None] * Dp
    B = Dp / da[:, None]
    dA, dB = A - A.mean(0), B - B.mean(0)
    s = -np.sum(dA * dB) / np.sum(dB * dB)
    spot = A + s * B
    rms = np.sqrt(np.mean(np.sum((spot - spot.mean(0)) ** 2, 1)))
    return s, ref_point + s * a + spot.mean(0), rms

def hit_plane(r, p0, axis, u, v):
    """光线落在传感器平面上的 2D 坐标 (mm)。"""
    P, D = r["P"][r["mask"]], r["D"][r["mask"]]
    t = ((p0 - P) @ axis) / (D @ axis)
    X = P + t[:, None] * D
    return np.stack([(X - p0) @ u, (X - p0) @ v], 1)

def mtf_on_sensor(r, p0, axis, lam, n_pad=4):
    """波前 OPD（参考球心=传感器上的质心点）→ 光瞳函数 → MTF（自相关）。
    返回频率 (lp/mm) 与径向 MTF（沿 x 方向切片）。"""
    m = r["mask"]
    n = r["grid"]
    P, D, opl = r["P"], r["D"], r["opl"]
    t = ((p0 - P) @ axis) / (D @ axis)
    X = P + t[:, None] * D
    cen = X[m].mean(0)
    opd = opl + r["n"] * np.linalg.norm(cen - P, axis=1)
    opd = opd - np.median(opd[m])
    W = np.where(m, opd, 0.0).reshape(n, n)
    M = m.reshape(n, n)
    pupil = M * np.exp(2j * np.pi * W / (lam * 1e-3))
    N = n * n_pad
    F = np.fft.fft2(pupil, (N, N))
    otf = np.fft.ifft2(np.abs(F) ** 2)
    otf = np.abs(otf) / np.abs(otf[0, 0])
    # 光瞳采样间隔（方向余弦）：由相邻网格光线方向差估计
    u = np.cross(axis, [0, 1, 0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0, 0])
    u /= np.linalg.norm(u)
    Dg = (D @ u).reshape(n, n)
    dalpha = np.nanmedian(np.abs(np.diff(np.where(M, Dg, np.nan), axis=1)))
    k = np.arange(N // 2)
    freq = k * dalpha / (lam * 1e-3)
    return freq, otf[0, :N // 2], (W[M].std(), W[M].max() - W[M].min())

DEFAULT = dict(WD=200.0, f_obj=200.0, cmo_diam=50.0, cmo_type="achro_vis",
               stereo_base=22.0, stop_diam=8.0, zoom_type="achro_vis",
               f_tube=100.0, tube_type="achro_vis", dich_dist=40.0,
               plate_t=1.0, plate_glass="F_SILICA", pixel=1.55e-3)
