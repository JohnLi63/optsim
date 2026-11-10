"""optsim 桌面版：手术显微镜 可见光/荧光 四路成像光学仿真（PySide6 + matplotlib）。"""
import sys, os, json, csv, copy, traceback
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT
from matplotlib.figure import Figure

from . import optsim as O
from . import engine as E
from . import glassdb as G

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei",
                                          "SimHei", "PingFang SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["savefig.dpi"] = 200

VIS, NIR, ACC, MUTED, GOOD, WARN, BAD = "#2a8a4a", "#b02a2a", "#3b4cc0", "#6b7280", "#2a8a4a", "#b7791f", "#c0392b"
CATCOL = {"CDGM": ACC, "SCHOTT": MUTED, "OHARA": WARN, "CRYSTAL": NIR}
APP_DIR = os.path.join(os.path.expanduser("~"), "optsim")


# ------------------------------------------------------------------ 线程
class Worker(QtCore.QThread):
    done = QtCore.Signal(object)
    fail = QtCore.Signal(str)
    prog = QtCore.Signal(float, str)

    def __init__(self, fn, *a, with_progress=False):
        super().__init__()
        self.fn, self.a, self.wp = fn, a, with_progress

    def run(self):
        try:
            if self.wp:
                r = self.fn(*self.a, progress=lambda f, m: self.prog.emit(f, m))
            else:
                r = self.fn(*self.a)
            self.done.emit(r)
        except Exception as e:
            self.fail.emit(f"{e}\n\n{traceback.format_exc()}")


def hline():
    f = QtWidgets.QFrame(); f.setFrameShape(QtWidgets.QFrame.HLine); f.setStyleSheet("color:#d9dce3"); return f


def section(text):
    l = QtWidgets.QLabel(text); l.setStyleSheet("font-weight:600;color:#5d6475;margin-top:4px"); return l


def fill_table(tbl, headers, rows, fmt=None):
    tbl.clear(); tbl.setColumnCount(len(headers)); tbl.setRowCount(len(rows))
    tbl.setHorizontalHeaderLabels(headers)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            if isinstance(v, float):
                s = "—" if np.isnan(v) else "∞" if not np.isfinite(v) else (fmt[j] if fmt and fmt[j] else "{:.3f}").format(v)
            else:
                s = "" if v is None else str(v)
            it = QtWidgets.QTableWidgetItem(s)
            it.setTextAlignment((Qt.AlignLeft if j == 0 else Qt.AlignRight) | Qt.AlignVCenter)
            tbl.setItem(i, j, it)
    tbl.resizeColumnsToContents()


class Slider(QtWidgets.QWidget):
    """带数值显示的滑块（内部用整数 µm）。"""
    changed = QtCore.Signal()

    def __init__(self, label, lo, hi, step, unit="µm", scale=1000.0):
        super().__init__()
        self.scale, self.unit = scale, unit
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        top = QtWidgets.QHBoxLayout(); self.lab = QtWidgets.QLabel(label); top.addWidget(self.lab); top.addStretch()
        self.spin = QtWidgets.QDoubleSpinBox(); self.spin.setRange(lo, hi); self.spin.setSingleStep(step)
        self.spin.setDecimals(0 if step >= 1 else 2); self.spin.setSuffix(" " + unit); self.spin.setFixedWidth(110)
        top.addWidget(self.spin); lay.addLayout(top)
        self.sl = QtWidgets.QSlider(Qt.Horizontal); self.sl.setRange(int(lo / step), int(hi / step)); self.step = step
        lay.addWidget(self.sl)
        self.sl.valueChanged.connect(lambda v: self._set(v * step, from_slider=True))
        self.spin.valueChanged.connect(lambda v: self._set(v, from_slider=False))
        self._busy = False

    def _set(self, v, from_slider):
        if self._busy: return
        self._busy = True
        if from_slider: self.spin.setValue(v)
        else: self.sl.setValue(int(round(v / self.step)))
        self._busy = False
        self.changed.emit()

    def value(self):  # → mm
        return self.spin.value() / self.scale

    def setValue(self, mm):
        self._busy = True
        self.spin.setValue(mm * self.scale); self.sl.setValue(int(round(mm * self.scale / self.step)))
        self._busy = False


# ------------------------------------------------------------------ 系统参数页
PARAMS = [("WD", "工作距离 WD", "mm", 50, 600, 1), ("f_obj", "主物镜焦距", "mm", 50, 600, 1),
          ("cmo_diam", "主物镜口径", "mm", 20, 120, 1), ("stereo_base", "立体基线", "mm", 5, 60, 0.5),
          ("stop_diam", "每路光阑直径", "mm", 2, 30, 0.5), ("f_tube", "筒镜焦距", "mm", 30, 400, 1),
          ("dich_dist", "筒镜到分光器距离", "mm", 10, 200, 1), ("cube_size", "分光棱镜边长", "mm", 5, 40, 0.5),
          ("plate_t", "分光平板厚度", "mm", 0.3, 5, 0.1), ("pixel", "像元尺寸", "µm", 0.5, 10, 0.05),
          ("sensor_w", "传感器宽", "mm", 1, 40, 0.1), ("sensor_h", "传感器高", "mm", 1, 40, 0.1)]


class ParamsTab(QtWidgets.QWidget):
    apply = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QWidget(); left.setFixedWidth(380); fl = QtWidgets.QVBoxLayout(left)
        form = QtWidgets.QFormLayout(); self.w = {}
        for k, lab, unit, lo, hi, st in PARAMS:
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(lo, hi); sp.setSingleStep(st); sp.setDecimals(3 if k == "pixel" else 2)
            sp.setSuffix(" " + unit); self.w[k] = sp; form.addRow(lab, sp)
        self.ltype = QtWidgets.QComboBox()
        for k, v in E.LTYPE_NAMES.items(): self.ltype.addItem(v, k)
        self.split = QtWidgets.QComboBox(); self.split.addItem("二向色分光棱镜", "cube"); self.split.addItem("45° 二向色平板", "plate")
        form.addRow("镜头色差校正", self.ltype); form.addRow("分光方式", self.split)
        self.custom_lab = QtWidgets.QLabel("自定义玻璃：未设置"); self.custom_lab.setWordWrap(True)
        self.custom_lab.setStyleSheet("color:#5d6475")
        fl.addWidget(section("系统参数")); fl.addLayout(form); fl.addWidget(self.custom_lab)
        fl.addWidget(section("变倍档（伽利略望远镜：第一组 / 第二组焦距，空 = 直通）"))
        self.zt = QtWidgets.QTableWidget(0, 3); self.zt.setHorizontalHeaderLabels(["倍率 ×", "第一组 f (mm)", "第二组 f (mm)"])
        self.zt.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); self.zt.setFixedHeight(190)
        fl.addWidget(self.zt)
        zb = QtWidgets.QHBoxLayout(); b1 = QtWidgets.QPushButton("加一档"); b2 = QtWidgets.QPushButton("删选中档")
        b1.clicked.connect(lambda: self.zt.insertRow(self.zt.rowCount())); b2.clicked.connect(lambda: self.zt.removeRow(self.zt.currentRow()))
        zb.addWidget(b1); zb.addWidget(b2); zb.addStretch(); fl.addLayout(zb)
        self.btn = QtWidgets.QPushButton("应用并标定"); self.btn.setStyleSheet("font-weight:600;padding:6px")
        self.btn.clicked.connect(lambda: self.apply.emit(self.get()))
        row = QtWidgets.QHBoxLayout(); self.bload = QtWidgets.QPushButton("载入配置…"); self.bsave = QtWidgets.QPushButton("保存配置…")
        row.addWidget(self.bload); row.addWidget(self.bsave)
        fl.addWidget(self.btn); fl.addLayout(row); fl.addStretch()
        self.bload.clicked.connect(self.load); self.bsave.clicked.connect(self.save)
        lay.addWidget(left)
        right = QtWidgets.QVBoxLayout()
        self.status = QtWidgets.QLabel(""); self.status.setWordWrap(True); self.status.setTextFormat(Qt.RichText)
        right.addWidget(section("镜片处方（右通道；自动设计，供光学设计起点参考，未做公差）"))
        self.tbl = QtWidgets.QTableWidget(); right.addWidget(self.tbl)
        rb = QtWidgets.QHBoxLayout(); self.bcsv = QtWidgets.QPushButton("导出处方 CSV…"); rb.addWidget(self.bcsv); rb.addStretch()
        right.addLayout(rb); right.addWidget(self.status)
        self.bcsv.clicked.connect(self.export_csv)
        lay.addLayout(right, 1)
        self.rows = []
        self.custom = None
        self.set(E.DEFAULT_CFG)

    def set(self, u):
        u = {**E.DEFAULT_CFG, **u}
        for k, *_ in PARAMS:
            self.w[k].setValue(u[k] * (1000 if k == "pixel" else 1))
        self.ltype.setCurrentIndex(max(0, self.ltype.findData(u["ltype"])))
        self.split.setCurrentIndex(max(0, self.split.findData(u["splitter"])))
        self.zt.setRowCount(0)
        for g, f1, f2 in u["zoom"]:
            r = self.zt.rowCount(); self.zt.insertRow(r)
            for j, v in enumerate((g, f1, f2)):
                self.zt.setItem(r, j, QtWidgets.QTableWidgetItem("" if v is None else f"{v:g}"))
        self.set_custom(u.get("custom"))

    def set_custom(self, c):
        self.custom = c
        self.custom_lab.setText("自定义玻璃：未设置（在“P–ν 玻璃图”里选好后点“应用到镜头设计”）" if not c else
                                f"自定义玻璃：{' + '.join(c['glasses'])}，校正波长 {', '.join(f'{l*1000:.0f}' for l in c['lams'])} nm")

    def get(self):
        u = copy.deepcopy(E.DEFAULT_CFG)
        for k, *_ in PARAMS:
            u[k] = self.w[k].value() / (1000 if k == "pixel" else 1)
        u["ltype"] = self.ltype.currentData(); u["splitter"] = self.split.currentData()
        z = []
        for r in range(self.zt.rowCount()):
            vals = []
            for j in range(3):
                it = self.zt.item(r, j); t = it.text().strip() if it else ""
                vals.append(float(t) if t else None)
            if vals[0] is not None:
                z.append(vals)
        u["zoom"] = sorted(z, key=lambda x: x[0]); u["custom"] = self.custom
        return u

    def load(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "载入配置", APP_DIR, "配置 (*.json)")
        if p:
            self.set(json.load(open(p, encoding="utf-8")))

    def save(self):
        os.makedirs(APP_DIR, exist_ok=True)
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存配置", os.path.join(APP_DIR, "config.json"), "配置 (*.json)")
        if p:
            E.save_json(p, self.get())

    def show_prescription(self, rows):
        self.rows = rows
        hd = ["组件", "面", "R (mm)", "顶点 z (mm)", "至下一面 (mm)", "之后介质", "通光半径 (mm)", "组件焦距 (mm)"]
        fill_table(self.tbl, hd, [[r["组件"], r["面"], r["R_mm"], r["顶点z_mm"], r["至下一面_mm"], r["之后介质"],
                                   r["通光半径_mm"], r["焦距_mm"]] for r in rows],
                   [None, None, "{:.2f}", "{:.2f}", "{:.2f}", None, "{:.1f}", "{:.1f}"])

    def export_csv(self):
        if not self.rows: return
        os.makedirs(APP_DIR, exist_ok=True)
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出处方", os.path.join(APP_DIR, "prescription.csv"), "CSV (*.csv)")
        if p:
            with open(p, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys())); w.writeheader(); w.writerows(self.rows)


# ------------------------------------------------------------------ 光路与对焦页
class FocusTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QWidget(); left.setFixedWidth(330); fl = QtWidgets.QVBoxLayout(left)
        fl.addWidget(section("变倍档"))
        self.zoom = QtWidgets.QComboBox(); fl.addWidget(self.zoom)
        self.dz = Slider("物面深度 Δz", -3000, 3000, 50); self.fx = Slider("视场点（物方横向）", 0, 8000, 100)
        self.tube = Slider("音圈：筒镜位移", -4000, 4000, 1); self.trim = Slider("NIR 微调轴", -2500, 2500, 1)
        for s in (self.dz, self.fx): fl.addWidget(s)
        fl.addWidget(hline()); fl.addWidget(self.tube)
        b = QtWidgets.QPushButton("可见光自动对焦"); b.clicked.connect(self.autofocus); fl.addWidget(b)
        fl.addWidget(self.trim)
        r = QtWidgets.QHBoxLayout(); b2 = QtWidgets.QPushButton("NIR 自动微调"); b3 = QtWidgets.QPushButton("全部归零")
        b2.clicked.connect(self.nirauto); b3.clicked.connect(self.zero); r.addWidget(b2); r.addWidget(b3); fl.addLayout(r)
        fl.addWidget(hline())
        self.view = QtWidgets.QComboBox(); self.view.addItems(["光路：全景", "光路：分光区放大"]); fl.addWidget(self.view)
        note = QtWidgets.QLabel("NIR 传感器初始位置是 1× 档、Δz=0 时标定的（相当于固定垫片），微调轴在此基础上移动。左右两路由同一个音圈推动。")
        note.setWordWrap(True); note.setStyleSheet("color:#5d6475"); fl.addWidget(note); fl.addStretch()
        lay.addWidget(left)
        right = QtWidgets.QVBoxLayout()
        tl = QtWidgets.QHBoxLayout(); self.tiles = {}
        for key, nm in (((-1, "vis"), "左 可见"), ((-1, "nir"), "左 NIR"), ((1, "vis"), "右 可见"), ((1, "nir"), "右 NIR")):
            l = QtWidgets.QLabel(); l.setTextFormat(Qt.RichText); l.setMinimumHeight(64)
            self.tiles[key] = (l, nm); tl.addWidget(l)
        right.addLayout(tl)
        self.fig1 = Figure(figsize=(9, 3.0), layout="constrained"); self.c1 = Canvas(self.fig1)
        self.fig2 = Figure(figsize=(9, 3.0), layout="constrained"); self.c2 = Canvas(self.fig2)
        right.addWidget(NavigationToolbar2QT(self.c1, self)); right.addWidget(self.c1, 3); right.addWidget(self.c2, 3)
        lay.addLayout(right, 1)
        self.timer = QtCore.QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(40); self.timer.timeout.connect(self.refresh)
        for s in (self.dz, self.fx, self.tube, self.trim): s.changed.connect(self.timer.start)
        self.zoom.currentIndexChanged.connect(self.timer.start); self.view.currentIndexChanged.connect(self.timer.start)
        self.dz.setValue(2.0)

    def set_cfg(self, cfg):
        self.zoom.blockSignals(True); self.zoom.clear()
        for g in E.zooms(cfg): self.zoom.addItem(f"{g:g}×", g)
        i = self.zoom.findData(1.0); self.zoom.setCurrentIndex(i if i >= 0 else 0); self.zoom.blockSignals(False)
        self.autofocus()

    def obj(self):
        cfg = self.main.cfg
        return [self.fx.value(), 0.0, cfg["z_obj0"] + self.dz.value()]

    def autofocus(self):
        if not self.main.cfg: return
        self.tube.setValue(E.autofocus(self.main.cfg, self.zoom.currentData(), self.obj())); self.refresh()

    def nirauto(self):
        cfg = self.main.cfg
        if not cfg: return
        e = E.eval_path(cfg, 1, self.zoom.currentData(), "nir", self.obj(), self.tube.value(), self.trim.value())
        self.trim.setValue(float(np.clip(self.trim.value() + e["s"], -2.5, 2.5))); self.refresh()

    def zero(self):
        self.tube.setValue(0); self.trim.setValue(0); self.refresh()

    def refresh(self):
        cfg = self.main.cfg
        if not cfg: return
        g, obj, tube, trim = self.zoom.currentData(), self.obj(), self.tube.value(), self.trim.value()
        ev = {}
        for side in (-1, 1):
            for path in ("vis", "nir"):
                e = E.eval_path(cfg, side, g, path, obj, tube, trim); ev[(side, path)] = e
                ratio = e["s"] / e["tol"]; a = abs(ratio)
                c, t = (GOOD, "清晰") if a < 0.5 else (WARN, "临界") if a < 1 else (BAD, "超出焦深")
                lab, nm = self.tiles[(side, path)]
                lab.setStyleSheet(f"border:1px solid #d9dce3;border-left:4px solid {c};border-radius:6px;padding:6px")
                if not np.isfinite(e["s"]):
                    lab.setStyleSheet(f"border:1px solid #d9dce3;border-left:4px solid {BAD};border-radius:6px;padding:6px")
                    lab.setText(f"<b>{nm}</b><br><span style='color:{BAD}'>没有光线到达传感器（渐晕或全反射）</span>"); continue
                lab.setText(f"<b style='color:{VIS if path=='vis' else NIR}'>{nm}</b> &nbsp;<span style='color:{c}'>{t}</span><br>"
                            f"<span style='font-size:15pt;font-family:monospace'>{e['s']*1000:+.1f} µm</span><br>"
                            f"<span style='color:#5d6475'>焦深 ±{e['tol']*1000:.0f} µm · 比值 {ratio:+.2f} · NA {e['na']:.4f}</span>")
        self.draw_layout(cfg, g, obj, tube, trim)
        self.draw_bottom(cfg, g, obj, tube, trim, ev)

    def draw_layout(self, cfg, g, obj, tube, trim):
        self.fig1.clear(); ax = self.fig1.add_subplot(111)
        ch, fans = E.ray_fans(cfg, g, obj, tube, trim)
        cx = ch.cx
        def prof(s, dz):
            dx = cx - s.ax; rr = np.sqrt(max(s.r_clear ** 2 - dx ** 2, 0)); y = np.linspace(-rr, rr, 40)
            h2 = dx * dx + y * y; c = s.c; sag = c * h2 / (1 + np.sqrt(np.maximum(1 - c * c * h2, 1e-9)))
            ax.plot(s.zv + dz + sag, y, color=MUTED, lw=1)
        for s in ch.cmo + ch.zoom: prof(s, 0)
        for s in ch.tube: prof(s, tube)
        _, sy, zs, rs = ch.stop
        ax.plot([zs, zs], [rs, rs + 5], "k", lw=2); ax.plot([zs, zs], [-rs, -rs - 5], "k", lw=2)
        if cfg["splitter"] == "cube":
            zd = ch.cube_vis[1].p0[2]; L = cfg.get("cube_size", 10.0)
            ax.add_patch(matplotlib.patches.Rectangle((zd - L / 2, -L / 2), L, L, fill=False, ec=ACC))
            ax.plot([zd - L / 2, zd + L / 2], [-L / 2, L / 2], color=ACC, lw=1)
        else:
            zd = ch.dich_front.p0[2]; ax.plot([zd - 6, zd + 6], [-6, 6], color=ACC, lw=1.5)
        for path in ("vis", "nir"):
            p, a, *_ = E.sensor(cfg, 1, path, trim); perp = np.array([0, -a[2], a[1]]) * 3.2
            A, B = p + perp, p - perp; ax.plot([A[2], B[2]], [A[1], B[1]], color=VIS if path == "vis" else NIR, lw=3)
        for path, pts in fans:
            for r in pts: ax.plot(r[:, 2], r[:, 1], color=VIS if path == "vis" else NIR, lw=0.6, alpha=0.8)
        ax.axhline(0, color="#d9dce3", ls="--", lw=0.8)
        if self.view.currentIndex() == 1:
            zt = ch.tube[0].zv + tube; allz = np.concatenate([p[:, :, 2].ravel() for _, p in fans])
            ally = np.concatenate([p[:, :, 1].ravel() for _, p in fans])
            ax.set_xlim(zt - 10, allz.max() + 6); m = allz > zt - 10; ax.set_ylim(-14, ally[m].max() + 4)
        ax.set_xlabel("z (mm)"); ax.set_ylabel("y (mm)"); ax.set_title(f"右通道侧视（y–z），{g:g}×", fontsize=9)
        ax.grid(alpha=0.25)
        self.c1.draw_idle()

    def draw_bottom(self, cfg, g, obj, tube, trim, ev):
        self.fig2.clear()
        gs = self.fig2.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.15, 3.2])
        names = {(-1, "vis"): "左可见", (-1, "nir"): "左NIR", (1, "vis"): "右可见", (1, "nir"): "右NIR"}
        pix = cfg["pixel"] * 1000
        for i, k in enumerate(((-1, "vis"), (-1, "nir"), (1, "vis"), (1, "nir"))):
            ax = self.fig2.add_subplot(gs[0, i]); xy = ev[k]["xy"] * 1000
            if len(xy) == 0:
                ax.set_axis_off(); ax.set_title(f"{names[k]}\n无光线", fontsize=8); continue
            xy = xy - xy.mean(0)
            span = max(6.0, np.abs(xy).max() * 1.1)
            ax.scatter(xy[:, 0], xy[:, 1], s=2, color=VIS if k[1] == "vis" else NIR)
            ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect("equal")
            step = pix * max(1, int(np.ceil(span / pix / 10)))
            ax.set_xticks(np.arange(-np.floor(span / step) * step, span, step)); ax.set_yticks(ax.get_xticks())
            ax.tick_params(labelbottom=False, labelleft=False, length=0); ax.grid(alpha=0.3)
            ax.set_title(f"{names[k]}\n框宽 {2*span:.0f} µm", fontsize=8)
        ax = self.fig2.add_subplot(gs[0, 5])
        lams, s, nir_line = E.chroma_curve(cfg, g, obj, tube, trim)
        tv, tn = ev[(1, "vis")]["tol"] * 1000, ev[(1, "nir")]["tol"] * 1000
        ax.axhspan(-tv, tv, color=VIS, alpha=0.1); ax.axhspan(nir_line - tn, nir_line + tn, color=NIR, alpha=0.1)
        ax.axhline(0, color=VIS, ls="--", lw=1); ax.axhline(nir_line, color=NIR, ls="--", lw=1)
        ax.plot(lams, s, color=ACC, lw=1, ls=":")
        m1, m2 = lams <= 690, lams >= 810
        ax.plot(lams[m1], s[m1], color=VIS, lw=2.5, label="可见光波段焦面")
        ax.plot(lams[m2], s[m2], color=NIR, lw=2.5, label="ICG 发射波段焦面")
        ax.text(452, 0, " 可见光传感器", color=VIS, va="bottom", fontsize=8)
        ax.text(452, nir_line, " NIR 传感器（等效位置）", color=NIR, va="bottom", fontsize=8)
        ax.set_xlabel("波长 (nm)"); ax.set_ylabel("最佳焦面 (µm)"); ax.set_title("焦移 vs 波长（右通道）", fontsize=9)
        ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="lower right")
        self.c2.draw_idle()


# ------------------------------------------------------------------ 批量分析页
class AnalysisTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main; self.res = None
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton("运行全档分析"); self.btn.setStyleSheet("font-weight:600;padding:6px")
        self.depth = QtWidgets.QDoubleSpinBox(); self.depth.setRange(0.1, 10); self.depth.setValue(2.0); self.depth.setSuffix(" mm")
        top.addWidget(self.btn); top.addWidget(QtWidgets.QLabel("物深范围 ±")); top.addWidget(self.depth)
        self.pb = QtWidgets.QProgressBar(); self.pb.setRange(0, 1000); top.addWidget(self.pb, 1)
        self.bexp = QtWidgets.QPushButton("导出 CSV…"); self.bjson = QtWidgets.QPushButton("导出 JSON…")
        top.addWidget(self.bexp); top.addWidget(self.bjson)
        lay.addLayout(top)
        self.msg = QtWidgets.QLabel("每一档：分别在 Δz = −d、0、+d 用筒镜自动对焦，记下 NIR 残余离焦；再在 Δz=0 做 5×5 视场点的配准分析。")
        self.msg.setWordWrap(True); lay.addWidget(self.msg)
        self.summary = QtWidgets.QLabel(""); self.summary.setTextFormat(Qt.RichText); self.summary.setWordWrap(True); lay.addWidget(self.summary)
        self.tbl = QtWidgets.QTableWidget(); lay.addWidget(self.tbl, 1)
        self.btn.clicked.connect(self.run); self.bexp.clicked.connect(self.export_csv); self.bjson.clicked.connect(self.export_json)

    def run(self):
        if not self.main.cfg: return
        d = self.depth.value()
        self.btn.setEnabled(False); self.pb.setValue(0)
        self.w = Worker(E.run_analysis, self.main.user, self.main.cfg, (-d, 0.0, d), with_progress=True)
        self.w.prog.connect(lambda f, m: (self.pb.setValue(int(f * 1000)), self.msg.setText(m)))
        self.w.done.connect(self.show); self.w.fail.connect(self.fail); self.w.start()

    def fail(self, e):
        self.btn.setEnabled(True); QtWidgets.QMessageBox.warning(self, "分析失败", e)

    def table_rows(self):
        rows = []
        for r in self.res["rows"]:
            f = r["focus"]
            rows.append([f"{r['gamma']:g}×", r["M"], *[x["tube_um"] for x in f], *[x["nir_um"] for x in f], f[1]["tol_um"],
                         r["lut_trim_um"], *r["reg_nir"], r["lr_raw"], r["lr_o3"]])
        return rows

    def headers(self):
        d = self.res["rows"][0]["focus"]
        dz = [f"{x['dz']:+g}" for x in d]
        return (["变倍档", "物方倍率 M"] + [f"筒镜位移 Δz{z} (µm)" for z in dz] + [f"NIR 离焦 Δz{z} (µm)" for z in dz] +
                ["NIR 焦深 ± (µm)", "查表微调值 (µm)", "NIR→可见 1阶 (px)", "2阶 (px)", "3阶 (px)", "左右垂直视差 原始 (px)", "3阶校正后 (px)"])

    def show(self, res):
        self.res = res; self.btn.setEnabled(True); self.pb.setValue(1000); self.msg.setText("完成。")
        n = len(self.headers())
        fill_table(self.tbl, self.headers(), self.table_rows(), [None, "{:.3f}"] + ["{:.1f}"] * (n - 2))
        c = lambda r: GOOD if r < 0.5 else WARN if r < 1 else BAD
        self.summary.setText(
            f"不加垫片（传感器停在 1× 标定位置）：最差 <b style='color:{c(res['ref_worst'])}'>{res['ref_worst']:.2f}</b> 倍焦深　·　"
            f"最优固定垫片 <b>{res['shim_um']:+.0f} µm</b>：最差 <b style='color:{c(res['shim_worst'])}'>{res['shim_worst']:.2f}</b> 倍　·　"
            f"每档查表微调：最差 <b style='color:{c(res['lut_worst'])}'>{res['lut_worst']:.2f}</b> 倍")

    def export_csv(self):
        if not self.res: return
        os.makedirs(APP_DIR, exist_ok=True)
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出", os.path.join(APP_DIR, "analysis.csv"), "CSV (*.csv)")
        if p:
            with open(p, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f); w.writerow(self.headers()); w.writerows(self.table_rows())
                w.writerow([]); w.writerow(["最优固定垫片 µm", self.res["shim_um"], "最差倍数", self.res["shim_worst"]])

    def export_json(self):
        if not self.res: return
        os.makedirs(APP_DIR, exist_ok=True)
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出", os.path.join(APP_DIR, "analysis.json"), "JSON (*.json)")
        if p:
            E.save_json(p, dict(config=self.main.user, result=self.res))


# ------------------------------------------------------------------ 玻璃图页
class GlassTab(QtWidgets.QWidget):
    to_lens = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.sel = ["CaF2", "H-K9L", "H-ZF13"]
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QWidget(); left.setFixedWidth(330); fl = QtWidgets.QVBoxLayout(left)
        fl.addWidget(section("设计波长（同时定义坐标轴）"))
        self.preset = QtWidgets.QComboBox()
        for t, v in (("可见–NIR：450 / 550 / 830 nm", (450, 550, 830)), ("经典可见：F / d / C", (486.13, 587.56, 656.27)),
                     ("宽带：460 / 600 / 850 nm", (460, 600, 850))):
            self.preset.addItem(t, v)
        fl.addWidget(self.preset)
        r = QtWidgets.QHBoxLayout(); self.l = []
        for v in (450, 550, 830):
            s = QtWidgets.QDoubleSpinBox(); s.setRange(380, 1100); s.setDecimals(1); s.setValue(v); s.setSuffix(" nm"); self.l.append(s); r.addWidget(s)
        fl.addLayout(r)
        fl.addWidget(section("玻璃库"))
        cr = QtWidgets.QGridLayout(); self.cats = {}
        for i, (k, v) in enumerate(G.CAT_NAMES.items()):
            cb = QtWidgets.QCheckBox(v); cb.setChecked(True); self.cats[k] = cb; cr.addWidget(cb, i // 2, i % 2)
        fl.addLayout(cr)
        form = QtWidgets.QFormLayout()
        self.f = QtWidgets.QDoubleSpinBox(); self.f.setRange(10, 2000); self.f.setValue(100); self.f.setSuffix(" mm")
        self.fn = QtWidgets.QDoubleSpinBox(); self.fn.setRange(1, 100); self.fn.setValue(12.5)
        self.amp = QtWidgets.QDoubleSpinBox(); self.amp.setRange(1, 20); self.amp.setValue(4); self.amp.setSingleStep(0.5)
        form.addRow("组合总焦距", self.f); form.addRow("像方 F 数", self.fn); form.addRow("放大上限 Σ|φᵢ|/φ", self.amp)
        fl.addLayout(form)
        fl.addWidget(section("已选玻璃（在图上点选，最多 3 种）"))
        self.sel_lab = QtWidgets.QLabel(); self.sel_lab.setWordWrap(True); fl.addWidget(self.sel_lab)
        rr = QtWidgets.QHBoxLayout(); bc = QtWidgets.QPushButton("清除选择"); bc.clicked.connect(self.clear); rr.addWidget(bc)
        self.badd = QtWidgets.QPushButton("按名字添加…"); self.badd.clicked.connect(self.add_by_name); rr.addWidget(self.badd); fl.addLayout(rr)
        self.bapply = QtWidgets.QPushButton("应用到镜头设计（自定义方案）"); self.bapply.setStyleSheet("font-weight:600")
        self.bapply.clicked.connect(self.apply_lens); fl.addWidget(self.bapply)
        fl.addWidget(hline())
        self.bsearch = QtWidgets.QPushButton("在当前玻璃库里找最好的三种"); self.bsearch.clicked.connect(self.search); fl.addWidget(self.bsearch)
        self.found = QtWidgets.QListWidget(); fl.addWidget(self.found, 1)
        self.found.itemClicked.connect(lambda it: self.set_sel(it.data(Qt.UserRole)))
        lay.addWidget(left)
        right = QtWidgets.QVBoxLayout()
        self.fig = Figure(figsize=(8, 4.6), layout="constrained"); self.cv = Canvas(self.fig)
        right.addWidget(NavigationToolbar2QT(self.cv, self))
        split = QtWidgets.QSplitter(Qt.Vertical); split.addWidget(self.cv)
        botw = QtWidgets.QWidget(); bot = QtWidgets.QHBoxLayout(botw); bot.setContentsMargins(0, 0, 0, 0)
        bl = QtWidgets.QVBoxLayout(); self.tbl = QtWidgets.QTableWidget(); bl.addWidget(self.tbl)
        self.info = QtWidgets.QLabel(); self.info.setWordWrap(True); self.info.setTextFormat(Qt.RichText); bl.addWidget(self.info)
        bw = QtWidgets.QWidget(); bw.setLayout(bl); bot.addWidget(bw, 1)
        self.fig2 = Figure(figsize=(5, 3), layout="constrained"); self.cv2 = Canvas(self.fig2); bot.addWidget(self.cv2, 1)
        split.addWidget(botw); split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        right.addWidget(split, 1)
        lay.addLayout(right, 1)
        self.preset.currentIndexChanged.connect(self.on_preset)
        for s in self.l: s.editingFinished.connect(self.update)
        for cb in self.cats.values(): cb.stateChanged.connect(self.update)
        for s in (self.f, self.fn): s.editingFinished.connect(self.update)
        self.cv.mpl_connect("pick_event", self.on_pick); self.cv.mpl_connect("motion_notify_event", self.on_hover)
        self.update()

    def lams(self):
        return np.sort(np.array([s.value() for s in self.l])) / 1000

    def mask(self):
        return np.array([self.cats[c].isChecked() for c in G.CATS])

    def on_preset(self):
        for s, v in zip(self.l, self.preset.currentData()): s.setValue(v)
        self.found.clear(); self.update()

    def clear(self):
        self.sel = []; self.update()

    def set_sel(self, names):
        self.sel = list(names); self.update()

    def add_by_name(self):
        n, ok = QtWidgets.QInputDialog.getItem(self, "添加玻璃", "牌号：", G.NAMES, 0, True)
        if ok and n in G.IDX:
            self.sel.append(n); self.sel = self.sel[-3:]; self.update()

    def apply_lens(self):
        if len(self.sel) < 2:
            QtWidgets.QMessageBox.information(self, "提示", "至少选两种玻璃。"); return
        l = self.lams()
        lams = [l[0], l[1], l[2]] if len(self.sel) == 3 else [l[0], l[2]]
        self.to_lens.emit(dict(glasses=list(self.sel), lams=[float(x) for x in lams]))

    def on_pick(self, ev):
        if not len(ev.ind): return
        n = G.NAMES[self.vis_idx[ev.ind[0]]]
        if n in self.sel: self.sel.remove(n)
        else:
            self.sel.append(n); self.sel = self.sel[-3:]
        self.update()

    def on_hover(self, ev):
        if ev.inaxes is not self.ax:
            if self.annot.get_visible(): self.annot.set_visible(False); self.cv.draw_idle()
            return
        ok, d = self.sc.contains(ev)
        if ok:
            i = d["ind"][0]; gi = self.vis_idx[i]
            nd = G.n_of([gi], 0.5876)[0, 0]
            self.annot.xy = (self.nu[i], self.P[i])
            self.annot.set_text(f"{G.NAMES[gi]}（{G.CAT_NAMES[G.CATS[gi]]}）\nn_d {nd:.4f}  ν {self.nu[i]:.2f}  P {self.P[i]:.4f}\n"
                                f"偏离正常线 ΔP {self.P[i]-(self.na+self.nb*self.nu[i]):+.4f}")
            self.annot.set_visible(True); self.cv.draw_idle()
        elif self.annot.get_visible():
            self.annot.set_visible(False); self.cv.draw_idle()

    def update(self):
        lams = self.lams(); m = self.mask()
        self.vis_idx = np.nonzero(m)[0]
        self.nu, self.P = G.coords(lams, self.vis_idx)
        self.na, self.nb = G.normal_line(self.nu, self.P, G.CATS[self.vis_idx] != "CRYSTAL")
        self.fig.clear(); ax = self.ax = self.fig.add_subplot(111)
        cols = [CATCOL[c] for c in G.CATS[self.vis_idx]]
        self.sc = ax.scatter(self.nu, self.P, s=14, c=cols, alpha=0.6, picker=4, lw=0)
        x = np.array([self.nu.min(), self.nu.max()]); ax.plot(x, self.na + self.nb * x, "--", color=MUTED, lw=1, label="正常线")
        if self.sel:
            si = [G.IDX[n] for n in self.sel]; snu, sP = G.coords(lams, si)
            if len(si) >= 2: ax.fill(snu, sP, color=ACC, alpha=0.15, ec=ACC, lw=2)
            ax.scatter(snu, sP, s=60, c=[CATCOL[G.CATS[i]] for i in si], edgecolors="k", zorder=5)
            for n, a, b in zip(self.sel, snu, sP): ax.annotate(n, (a, b), xytext=(6, 6), textcoords="offset points", fontweight="bold")
        ax.invert_xaxis()
        L = (lams * 1000).astype(int)
        ax.set_xlabel(f"ν = (n{L[1]}−1)/(n{L[0]}−n{L[2]})（越往右色散越大）"); ax.set_ylabel(f"P = (n{L[0]}−n{L[1]})/(n{L[0]}−n{L[2]})")
        for k, c in CATCOL.items():
            if self.cats[k].isChecked(): ax.scatter([], [], c=c, label=G.CAT_NAMES[k])
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)
        ax.set_title(f"P–ν 玻璃图（{len(self.vis_idx)} 种）", fontsize=10)
        self.annot = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="white", ec="#d9dce3"), fontsize=8)
        self.annot.set_visible(False)
        self.cv.draw_idle()
        self.solve_and_plot(lams)

    def solve_and_plot(self, lams):
        f = self.f.value(); F = self.fn.value(); na = 1 / (2 * F); tol = 0.55 / (2 * na * na)
        grid = np.arange(0.45, 0.8501, 0.005)
        self.sel_lab.setText(" + ".join(self.sel) if self.sel else "还没有选择")
        self.fig2.clear(); ax = self.fig2.add_subplot(111)
        ref = G.ref_curve(lams, grid, f) * 1000
        ax.axhspan(-tol, tol, color=VIS, alpha=0.1, label=f"瑞利焦深 ±{tol:.0f} µm (F/{F:g})")
        ax.plot(grid * 1000, ref, "--", color=MUTED, label="参考 H-K9L + H-ZF13 消色差")
        self.tbl.clear(); self.tbl.setRowCount(0); self.info.setText("")
        if self.sel:
            K = G.solve(self.sel, lams, f)
            if K is None:
                self.info.setText("这几种玻璃几乎共线，方程组无解。")
            else:
                nmid = G.n_of([G.IDX[n] for n in self.sel], lams[1])[:, 0]
                phis = (nmid - 1) * K; phi = 1 / f; amp = np.abs(phis).sum() / phi
                nu, P = G.coords(lams, [G.IDX[n] for n in self.sel])
                fill_table(self.tbl, ["玻璃", "n(λ₂)", "ν", "P", "φᵢ/φ", "fᵢ (mm)"],
                           [[n, nmid[i], nu[i], P[i], phis[i] / phi, 1 / phis[i]] for i, n in enumerate(self.sel)],
                           [None, "{:.4f}", "{:.2f}", "{:.4f}", "{:.3f}", "{:.1f}"])
                sh = G.shift_curve(self.sel, K, lams, grid) * 1000
                ax.plot(grid * 1000, sh, color=ACC, lw=2, label="当前选择")
                ax.plot(lams * 1000 if len(self.sel) == 3 else [lams[0] * 1000, lams[2] * 1000], [0] * (3 if len(self.sel) == 3 else 2), "o", color=ACC)
                c = GOOD if amp < 3 else WARN if amp < 6 else BAD
                self.info.setText(f"光焦度放大 Σ|φᵢ|/φ = <b style='color:{c}'>{amp:.2f}</b>　·　450–850 nm 焦移峰谷 "
                                  f"<b>{np.ptp(sh):.1f} µm</b>（参考 {np.ptp(ref):.0f} µm）")
        all_ = ax.get_lines()
        ax.set_xlabel("波长 (nm)"); ax.set_ylabel("Δf (µm)"); ax.set_title("焦移曲线（薄透镜密接）", fontsize=9)
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
        self.cv2.draw_idle()

    def search(self):
        self.bsearch.setEnabled(False); self.found.clear(); self.found.addItem("搜索中…")
        self.w = Worker(G.search, self.mask(), self.lams(), self.f.value(), self.amp.value())
        self.w.done.connect(self.show_found); self.w.fail.connect(lambda e: (self.bsearch.setEnabled(True), self.found.clear(), self.found.addItem(e[:200])))
        self.w.start()

    def show_found(self, res):
        self.bsearch.setEnabled(True); self.found.clear()
        if not res: self.found.addItem("这个放大上限下没有组合，把上限调大一点。")
        for r in res:
            it = QtWidgets.QListWidgetItem(f"{' + '.join(r['names'])}\n峰谷 {r['pv_um']:.1f} µm · 放大 {r['amp']:.2f}")
            it.setData(Qt.UserRole, r["names"]); self.found.addItem(it)


# ------------------------------------------------------------------ 主窗口
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("optsim 光学仿真台 — 手术显微镜 可见/NIR 四路"); self.resize(1500, 950)
        self.user, self.cfg = None, None
        self.tabs = QtWidgets.QTabWidget(); self.setCentralWidget(self.tabs)
        self.params = ParamsTab(); self.focus = FocusTab(self); self.ana = AnalysisTab(self); self.glass = GlassTab()
        self.tabs.addTab(self.params, "系统参数"); self.tabs.addTab(self.focus, "光路与对焦")
        self.tabs.addTab(self.ana, "全档分析"); self.tabs.addTab(self.glass, "P–ν 玻璃图")
        self.params.apply.connect(self.calibrate)
        self.glass.to_lens.connect(self.use_custom)
        self.sb = self.statusBar()
        self.calibrate(self.params.get())

    def use_custom(self, c):
        self.params.set_custom(c); self.params.ltype.setCurrentIndex(self.params.ltype.findData("custom"))
        self.tabs.setCurrentWidget(self.params)
        self.sb.showMessage("已把玻璃组合放进“自定义”方案，点“应用并标定”即可重新设计镜头。", 8000)

    def calibrate(self, user):
        self.params.btn.setEnabled(False)
        self.sb.showMessage("正在自动设计镜头并标定（首次或换玻璃时需要十几秒到几分钟）…")
        QtWidgets.QApplication.setOverrideCursor(Qt.BusyCursor)
        self.w = Worker(self._cal, user); self.w.done.connect(self.calibrated); self.w.fail.connect(self.cal_fail); self.w.start()

    @staticmethod
    def _cal(user):
        u, cfg = E.calibrate(user)
        return u, cfg, E.prescription(cfg)

    def calibrated(self, r):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.user, self.cfg, rows = r
        self.params.btn.setEnabled(True); self.params.show_prescription(rows)
        self.params.status.setText(f"标定完成：物面 z = {self.cfg['z_obj0']:.3f} mm；变倍组无焦间隔 " +
                                   "，".join(f"{g:g}×: {v:.2f} mm" for g, v in self.cfg["zoom_gap"].items()))
        amps = E.lens_amplification(self.cfg)
        worst = max((a for _, a in amps), default=0)
        txt = "；各透镜组件光焦度放大 Σ|φᵢ|/φ：" + "，".join(f"f{f:+g}: {a:.1f}" for f, a in amps)
        if worst > 6:
            txt += "　<b style='color:#c0392b'>有组件放大超过 6，曲率过陡，这组玻璃基本做不出来，换三角形更大的玻璃组合。</b>"
        self.params.status.setText(self.params.status.text() + txt)
        self.sb.showMessage("标定完成", 5000)
        self.focus.set_cfg(self.cfg)

    def cal_fail(self, e):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.params.btn.setEnabled(True); self.sb.showMessage("标定失败")
        QtWidgets.QMessageBox.warning(self, "标定失败", e)


def main():
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    f = app.font(); f.setPointSize(10); app.setFont(f)
    w = Main(); w.show()
    sys.exit(app.exec())
