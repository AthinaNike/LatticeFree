# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-or-later
#
# LatticeFree V2 - TPMS lattice infill generator for FreeCAD
# Copyright (C) 2026 Veronica Massara
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
LatticeFree — topology optimization: FreeCAD side
==================================================
Command "Topology optimization (variable lattice density)":

1. reads loads and constraints from the FEM workbench (active Analysis):
   Fem::ConstraintFixed, Fem::ConstraintForce, Fem::ConstraintPressure,
   on faces, edges or vertices of the component;
2. builds the voxel model of the component (same occupancy/SDF as the
   generator) and runs topopt.optimize();
3. shows a 3D preview of the density map, saves it (.npz + .vtk) and
   offers to open the generator with "density from topology
   optimization" already enabled.

All the numerics are in topopt.py (pure numpy).
"""

import os
import numpy as np
import FreeCAD as App
import FreeCADGui as Gui

from PySide import QtCore, QtGui
try:
    from PySide import QtWidgets
except ImportError:
    QtWidgets = QtGui

from . import engine, topopt

DEMO_NAME = "LatticeFree_DemoCantilever"


# ======================================================================
# FEM CONSTRAINTS
# ======================================================================

def find_analysis(doc):
    try:
        import FemGui
        a = FemGui.getActiveAnalysis()
        if a is not None and a.Document == doc:
            return a
    except Exception:
        pass
    for o in doc.Objects:
        if o.isDerivedFrom("Fem::FemAnalysis"):
            return o
    return None


def collect_constraints(doc):
    """FEM constraints of the active Analysis (or of the whole document
    if there is no Analysis). Suppressed constraints are skipped."""
    analysis = find_analysis(doc)
    objs = analysis.Group if analysis is not None else doc.Objects
    out = dict(analysis=analysis, fixed=[], force=[], pressure=[], unsupported=[])
    for o in objs:
        if getattr(o, "Suppressed", False):
            continue
        t = o.TypeId
        if t == "Fem::ConstraintFixed":
            out["fixed"].append(o)
        elif t == "Fem::ConstraintForce":
            out["force"].append(o)
        elif t == "Fem::ConstraintPressure":
            out["pressure"].append(o)
        elif t.startswith("Fem::Constraint"):
            out["unsupported"].append(o)
    return out


def _sub_shapes(cobj):
    shapes = []
    for ref in cobj.References:
        obj, subs = ref[0], ref[1]
        if isinstance(subs, str):
            subs = (subs,)
        for s in subs:
            try:
                shapes.append(obj.Shape.getElement(s))
            except Exception as e:
                App.Console.PrintWarning(
                    "TopOpt: cannot read {}.{} ({}): {}\n".format(
                        obj.Label, s, cobj.Label, e))
    return shapes


def _sample_shape(sh, step):
    """Samples of a Face/Edge/Vertex: (points, weights, outward normals).
    Weights are areas (faces), lengths (edges) or 1 (vertices)."""
    st = sh.ShapeType
    if st == "Face":
        pts, tris = sh.tessellate(max(0.01, 0.1 * step))
        if not tris:
            return None
        P = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
        T = np.array(tris, dtype=np.int64)
        S, W, tid = topopt.sample_triangles(P, T, step)
        N = np.zeros((len(T), 3))
        for i, c in enumerate(P[T].mean(axis=1)):
            try:
                u, v = sh.Surface.parameter(App.Vector(*c))
                nv = sh.normalAt(u, v)
                N[i] = (nv.x, nv.y, nv.z)
            except Exception:
                a, b, cc = P[T[i]]
                nn = np.cross(b - a, cc - a)
                N[i] = nn / (np.linalg.norm(nn) + 1e-30)
        return S, W, N[tid]
    if st == "Edge":
        n = max(2, int(np.ceil(sh.Length / step)) + 1)
        S = np.array([[p.x, p.y, p.z] for p in sh.discretize(n)])
        return S, np.full(len(S), sh.Length / len(S)), np.zeros_like(S)
    if st == "Vertex":
        S = np.array([[sh.X, sh.Y, sh.Z]])
        return S, np.ones(1), np.zeros_like(S)
    return None


def _samples_of(cobj, step):
    parts = [_sample_shape(s, step) for s in _sub_shapes(cobj)]
    parts = [p for p in parts if p is not None and len(p[0])]
    if not parts:
        return None
    return (np.vstack([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.vstack([p[2] for p in parts]))


def _value(q, unit):
    """Quantity -> float in the given unit (older versions: plain float)."""
    if hasattr(q, "getValueAs"):
        return float(q.getValueAs(unit))
    return float(q)


def apply_constraints(model, cons, log):
    """Maps the FEM constraints onto the voxel model nodes."""
    step = 0.5 * min(model.hx, model.hy, model.hz)
    for o in cons["fixed"]:
        smp = _samples_of(o, step)
        if smp is None:
            App.Console.PrintWarning("TopOpt: '{}' has no geometry, "
                                     "skipped.\n".format(o.Label))
            continue
        n = model.add_fixed(smp[0])
        log("  Fixed     '{}': {} nodes clamped\n".format(o.Label, n))
    for o in cons["force"]:
        smp = _samples_of(o, step)
        if smp is None:
            continue
        S, W, _ = smp
        F = _value(o.Force, "N")
        d = o.DirectionVector
        d = np.array([d.x, d.y, d.z], dtype=np.float64)
        if np.linalg.norm(d) < 1e-12:
            App.Console.PrintWarning("TopOpt: '{}' has no direction, "
                                     "skipped.\n".format(o.Label))
            continue
        d /= np.linalg.norm(d)
        vec = (F * W / W.sum())[:, None] * d[None, :]
        n = model.add_forces(S, vec)
        log("  Force     '{}': {:.1f} N along ({:.2f}, {:.2f}, {:.2f}) "
            "on {} nodes\n".format(o.Label, F, d[0], d[1], d[2], n))
    for o in cons["pressure"]:
        smp = _samples_of(o, step)
        if smp is None:
            continue
        S, W, N = smp
        p = _value(o.Pressure, "MPa")
        sign = -1.0 if getattr(o, "Reversed", False) else 1.0
        # positive pressure pushes INTO the surface (CalculiX convention)
        vec = -sign * p * W[:, None] * N
        n = model.add_forces(S, vec)
        tot = vec.sum(axis=0)
        log("  Pressure  '{}': {:.3f} MPa on {:.1f} mm2 -> resultant "
            "({:.1f}, {:.1f}, {:.1f}) N on {} nodes\n".format(
                o.Label, p, W.sum(), tot[0], tot[1], tot[2], n))
    for o in cons["unsupported"]:
        App.Console.PrintWarning(
            "TopOpt: constraint '{}' ({}) not supported, ignored.\n"
            .format(o.Label, o.TypeId))


def apply_demo_loads(model, bb, force_n, log):
    """Demo cantilever: clamped at X min, downward load on face X max."""
    xmin, xmax = bb.XMin, bb.XMax
    face = np.array([[0, bb.YMin, bb.ZMin], [0, bb.YMax, bb.ZMin],
                     [0, bb.YMax, bb.ZMax], [0, bb.YMin, bb.ZMax]], float)
    tri = [[0, 1, 2], [0, 2, 3]]
    step = 0.5 * min(model.hx, model.hy, model.hz)
    f0 = face.copy(); f0[:, 0] = xmin
    S, _, _ = topopt.sample_triangles(f0, tri, step)
    n_fix = model.add_fixed(S)
    f1 = face.copy(); f1[:, 0] = xmax
    S, W, _ = topopt.sample_triangles(f1, tri, step)
    vec = np.zeros((len(S), 3))
    vec[:, 2] = -force_n * W / W.sum()
    n_f = model.add_forces(S, vec)
    log("  Demo: clamped face X={:.1f} ({} nodes), {:.0f} N downward "
        "on face X={:.1f} ({} nodes)\n".format(xmin, n_fix, force_n, xmax, n_f))


# ======================================================================
# VOXEL MODEL OF THE COMPONENT
# ======================================================================

def build_model(target, h, t_shell, Es, nu, C, n, log):
    bb = target.Shape.BoundBox
    bb_min = (bb.XMin, bb.YMin, bb.ZMin)
    bb_max = (bb.XMax, bb.YMax, bb.ZMax)
    xn, yn, zn = topopt.make_grid(bb_min, bb_max, h)
    xc, yc, zc = topopt.centers(xn), topopt.centers(yn), topopt.centers(zn)
    hm = min(xn[1] - xn[0], yn[1] - yn[0], zn[1] - zn[0])
    log("FE grid: {} x {} x {} elements ({:.2f} mm)\n".format(
        len(xc), len(yc), len(zc), hm))
    pts, fcs = target.Shape.tessellate(max(0.05, 0.25 * hm))
    P = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
    F = np.array(fcs, dtype=np.int64)
    occ = engine.occupancy_from_tris(P, F, xc, yc, zc, hm)
    if not occ.any():
        raise ValueError("No element inside '{}': is it a closed solid? "
                         "Try a smaller FE element.".format(target.Label))
    # pad with one empty layer: the FE grid coincides with the bounding
    # box, so without it the distance transform would see no outside
    occ_p = np.pad(occ, 1, mode="constant", constant_values=False)
    sdf = engine.sdf_coarse(occ_p, hm, t_shell + 3.0 * hm)[1:-1, 1:-1, 1:-1]
    if engine.HAVE_SCIPY:
        # exact EDT measures centre-to-outside-centre: surface is h/2 closer
        sdf = sdf - 0.5 * hm
    phi = topopt.shell_fraction(sdf, t_shell, hm)
    return topopt.Model(xn, yn, zn, occ, phi, Es, nu, C, n), bb


# ======================================================================
# 3D PREVIEW (one mesh per density band, cube size ~ density)
# ======================================================================

_BAND_COLORS = [(0.23, 0.30, 0.75), (0.27, 0.62, 0.85), (0.55, 0.80, 0.45),
                (0.98, 0.75, 0.25), (0.85, 0.22, 0.18)]


def _set_color(vo, rgb):
    for prop in ("ShapeColor",):
        try:
            setattr(vo, prop, rgb)
            return
        except Exception:
            pass
    try:
        mat = App.Material()
        mat.DiffuseColor = rgb
        vo.ShapeAppearance = (mat,)
    except Exception:
        pass


def _cube_bands(doc, res, values, bands, label):
    """values: (nx,ny,nz) grid with NaN outside. bands: list of
    (lo, hi, rgb, name, size) with size in (0,1] of the element."""
    import Mesh
    xc, yc, zc = res["xc"], res["yc"], res["zc"]
    h = np.array([xc[1] - xc[0] if len(xc) > 1 else 1.0,
                  yc[1] - yc[0] if len(yc) > 1 else 1.0,
                  zc[1] - zc[0] if len(zc) > 1 else 1.0])
    old = doc.getObject(label)
    if old is not None:
        for o in list(old.Group):
            doc.removeObject(o.Name)
        doc.removeObject(old.Name)
    grp = doc.addObject("App::DocumentObjectGroup", label)
    I, J, K = np.nonzero(~np.isnan(values))
    vals = values[I, J, K]
    C = np.stack([xc[I], yc[J], zc[K]], axis=1)
    cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) - 0.5
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    tris = np.array([[q[0], q[1], q[2]] for q in quads] +
                    [[q[0], q[2], q[3]] for q in quads])
    for b, (lo, hi, rgb, name, size) in enumerate(bands):
        sel = (vals >= lo) & (vals < hi)
        if not sel.any():
            continue
        V = C[sel][:, None, :] + cube[None, :, :] * (size * h)
        soup = V[:, tris, :].reshape(-1, 3, 3)
        obj = doc.addObject("Mesh::Feature", "{}_{}".format(label, b))
        obj.Mesh = Mesh.Mesh(soup.tolist())
        obj.Label = name
        grp.addObject(obj)
        try:
            _set_color(obj.ViewObject, rgb)
        except Exception:
            pass
    return grp


def make_fs_preview(doc, res, fs_target, label="TopOpt_safety"):
    """Safety-factor map: critical elements big and red, safe ones
    small and blue (only the lattice region is shown)."""
    fs = res["fs_grid"]
    t = float(fs_target)
    bands = [
        (-1.0, 1.0, (0.85, 0.15, 0.12), "FS < 1  (yields)", 0.95),
        (1.0, t, (0.98, 0.55, 0.15), "FS 1-{:.1f}  (below target)".format(t), 0.8),
        (t, 2 * t, (0.95, 0.85, 0.30), "FS {:.1f}-{:.1f}".format(t, 2 * t), 0.5),
        (2 * t, 1e30, (0.30, 0.50, 0.85), "FS > {:.1f}".format(2 * t), 0.25),
    ]
    return _cube_bands(doc, res, fs, bands, label)


def make_preview(doc, res, label="TopOpt_density"):
    import Mesh
    rho = res["rho"]
    xc, yc, zc = res["xc"], res["yc"], res["zc"]
    h = np.array([xc[1] - xc[0] if len(xc) > 1 else 1.0,
                  yc[1] - yc[0] if len(yc) > 1 else 1.0,
                  zc[1] - zc[0] if len(zc) > 1 else 1.0])
    old = doc.getObject(label)
    if old is not None:
        for o in list(old.Group):
            doc.removeObject(o.Name)
        doc.removeObject(old.Name)
    grp = doc.addObject("App::DocumentObjectGroup", label)
    lo, hi = float(res["rho_min"]), float(res["rho_max"])
    edges = np.linspace(lo, hi, len(_BAND_COLORS) + 1)
    I, J, K = np.nonzero(~np.isnan(rho))
    vals = rho[I, J, K]
    C = np.stack([xc[I], yc[J], zc[K]], axis=1)
    cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) - 0.5
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    tris = np.array([[q[0], q[1], q[2]] for q in quads] +
                    [[q[0], q[2], q[3]] for q in quads])
    for b in range(len(_BAND_COLORS)):
        sel = (vals >= edges[b]) & ((vals < edges[b + 1]) |
                                    (b == len(_BAND_COLORS) - 1))
        if not sel.any():
            continue
        s = 0.35 + 0.6 * (vals[sel] - lo) / max(hi - lo, 1e-9)
        V = C[sel][:, None, :] + cube[None, :, :] * (s[:, None, None] * h)
        soup = V[:, tris, :].reshape(-1, 3, 3)
        m = Mesh.Mesh(soup.tolist())
        obj = doc.addObject("Mesh::Feature", "{}_{:02.0f}_{:02.0f}".format(
            label, edges[b] * 100, edges[b + 1] * 100))
        obj.Mesh = m
        obj.Label = "density {:.0f}-{:.0f}%".format(
            edges[b] * 100, edges[b + 1] * 100)
        grp.addObject(obj)
        try:
            _set_color(obj.ViewObject, _BAND_COLORS[b])
        except Exception:
            pass
    return grp


# ======================================================================
# DIALOG
# ======================================================================

class TopOptDialog(QtWidgets.QDialog):

    def __init__(self):
        super(TopOptDialog, self).__init__(Gui.getMainWindow())
        self.setWindowTitle("LatticeFree V2 — Topology optimization")
        self.setMinimumWidth(460)
        form = QtWidgets.QFormLayout(self)

        intro = QtWidgets.QLabel(
            "Optimal lattice density distribution (max stiffness for "
            "the given mass). Loads and constraints come from the FEM "
            "workbench: Fixed, Force, Pressure in the active Analysis.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        form.addRow(intro)

        self.combo_target = QtWidgets.QComboBox()
        self.combo_target.addItem(
            "— Demo: cantilever 120x30x40 mm (built-in load) —", None)
        doc = App.ActiveDocument
        if doc:
            sel = Gui.Selection.getSelection()
            sel_name = sel[0].Name if sel else None
            from .commands import is_valid_solid
            for obj in doc.Objects:
                if is_valid_solid(obj):
                    self.combo_target.addItem(obj.Label, obj.Name)
                    if obj.Name == sel_name:
                        self.combo_target.setCurrentIndex(
                            self.combo_target.count() - 1)
        form.addRow("Component:", self.combo_target)

        self.label_cons = QtWidgets.QLabel("")
        self.label_cons.setWordWrap(True)
        form.addRow("", self.label_cons)

        self.combo_tpms = QtWidgets.QComboBox()
        for k, d in engine.TPMS.items():
            self.combo_tpms.addItem(d["label"], k)
        form.addRow("TPMS:", self.combo_tpms)

        self.spin_cell = self._dspin(0.5, 500.0, 10.0, " mm")
        form.addRow("TPMS cell size:", self.spin_cell)
        cell_row = QtWidgets.QHBoxLayout()
        self.btn_cell = QtWidgets.QPushButton("Fit cell to target")
        self.btn_cell.setToolTip(
            "Smallest cell whose printable minimum density leaves room "
            "for the optimization (min density ~55% of the mean).")
        cell_row.addWidget(self.btn_cell)
        cell_row.addStretch(1)
        form.addRow("", cell_row)

        # --- Printability limits (feasibility) -----------------------
        self.spin_lw = self._dspin(0.1, 2.0, topopt.DEFAULT_LINE_WIDTH,
                                   " mm", 0.05)
        self.spin_lines = QtWidgets.QSpinBox()
        self.spin_lines.setRange(1, 6)
        self.spin_lines.setValue(topopt.DEFAULT_MIN_LINES)
        self.spin_lines.setSuffix(" lines")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.spin_lw)
        row.addWidget(QtWidgets.QLabel("x"))
        row.addWidget(self.spin_lines)
        form.addRow("Min wall (line width x lines):", row)
        self.spin_pore = self._dspin(0.2, 20.0, topopt.DEFAULT_MIN_PORE,
                                     " mm", 0.1)
        form.addRow("Minimum pore diameter:", self.spin_pore)
        self.spin_shell = self._dspin(0.0, 20.0, 1.2, " mm", 0.2)
        form.addRow("Outer shell thickness:", self.spin_shell)

        self.spin_target = self._dspin(5.0, 70.0, 25.0, " %", 1.0)
        form.addRow("Mean lattice density (mass):", self.spin_target)
        self.spin_rmax = self._dspin(10.0, 90.0, engine.RHO_MAX * 100, " %", 1.0)
        form.addRow("Maximum density:", self.spin_rmax)
        self.label_range = QtWidgets.QLabel("")
        self.label_range.setStyleSheet("color: gray;")
        self.label_range.setWordWrap(True)
        form.addRow("", self.label_range)

        # --- Material ----------------------------------------------
        self.combo_mat = QtWidgets.QComboBox()
        for k in topopt.MATERIALS:
            self.combo_mat.addItem(k, k)
        self.combo_mat.addItem("Custom", None)
        form.addRow("Material:", self.combo_mat)
        self.spin_E = self._dspin(1.0, 500000.0, 3500.0, " MPa", 100.0, 0)
        self.spin_nu = self._dspin(0.0, 0.49, 0.36, "", 0.01)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.spin_E)
        row.addWidget(QtWidgets.QLabel("nu"))
        row.addWidget(self.spin_nu)
        form.addRow("Young modulus Es:", row)
        self.spin_sy = self._dspin(0.1, 5000.0, 55.0, " MPa", 1.0, 1)
        self.spin_su = self._dspin(0.1, 5000.0, 60.0, " MPa", 1.0, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.spin_sy)
        row.addWidget(QtWidgets.QLabel("ultimate"))
        row.addWidget(self.spin_su)
        form.addRow("Yield strength:", row)
        self.spin_dens_m = self._dspin(0.1, 25.0, 1.24, " g/cm3", 0.01)
        form.addRow("Material density:", self.spin_dens_m)
        self.spin_zf = self._dspin(0.05, 1.0, 0.5, "", 0.05)
        self.combo_axis = QtWidgets.QComboBox()
        for i, a in enumerate("XYZ"):
            self.combo_axis.addItem("build axis " + a, i)
        self.combo_axis.setCurrentIndex(2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.spin_zf)
        row.addWidget(self.combo_axis)
        form.addRow("Interlayer strength factor:", row)
        self.label_mat = QtWidgets.QLabel(
            "Typical values for printed parts (XY): enter your filament "
            "datasheet. Interlayer factor = strength across layers / XY "
            "strength (FDM: 0.3-0.7).")
        self.label_mat.setWordWrap(True)
        self.label_mat.setStyleSheet("color: gray;")
        form.addRow("", self.label_mat)
        self.combo_mat.currentIndexChanged.connect(self._material_changed)
        for w in (self.spin_E, self.spin_nu, self.spin_sy, self.spin_su,
                  self.spin_dens_m, self.spin_zf):
            w.valueChanged.connect(self._material_edited)
        self._loading_mat = False
        self._material_changed()
        self.spin_C = self._dspin(0.01, 2.0, 0.414, "", 0.01, 3)
        self.spin_n = self._dspin(0.8, 4.0, 1.386, "", 0.01, 3)
        law_row = QtWidgets.QHBoxLayout()
        law_row.addWidget(QtWidgets.QLabel("C"))
        law_row.addWidget(self.spin_C)
        law_row.addWidget(QtWidgets.QLabel("n"))
        law_row.addWidget(self.spin_n)
        form.addRow("Law E*/Es = C rho^n:", law_row)
        self.label_law = QtWidgets.QLabel("")
        self.label_law.setStyleSheet("color: gray;")
        self.label_law.setWordWrap(True)
        form.addRow("", self.label_law)
        self.spin_C2 = self._dspin(0.01, 2.0, 0.30, "", 0.01, 3)
        self.spin_m = self._dspin(0.5, 4.0, 1.5, "", 0.05, 2)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("C2"))
        row.addWidget(self.spin_C2)
        row.addWidget(QtWidgets.QLabel("m"))
        row.addWidget(self.spin_m)
        form.addRow("Strength sigma*/sigma_s = C2 rho^m:", row)
        self.spin_fs = self._dspin(1.0, 10.0, 2.0, "", 0.1)
        form.addRow("Target safety factor:", self.spin_fs)

        elem_row = QtWidgets.QHBoxLayout()
        self.spin_elem = self._dspin(0.2, 100.0, 5.0, " mm", 0.5)
        self.check_auto = QtWidgets.QCheckBox("auto")
        self.check_auto.setChecked(True)
        elem_row.addWidget(self.spin_elem)
        elem_row.addWidget(self.check_auto)
        form.addRow("FE element size:", elem_row)
        filt_row = QtWidgets.QHBoxLayout()
        self.spin_filter = self._dspin(0.5, 500.0, 10.0, " mm", 1.0)
        self.check_filter = QtWidgets.QCheckBox("= cell")
        self.check_filter.setChecked(True)
        filt_row.addWidget(self.spin_filter)
        filt_row.addWidget(self.check_filter)
        form.addRow("Filter radius:", filt_row)
        self.spin_iter = QtWidgets.QSpinBox()
        self.spin_iter.setRange(5, 300)
        self.spin_iter.setValue(60)
        form.addRow("Max iterations:", self.spin_iter)
        self.spin_demo_force = self._dspin(1.0, 100000.0, 500.0, " N", 50.0)
        form.addRow("Demo load:", self.spin_demo_force)

        self.label_info = QtWidgets.QLabel("")
        self.label_info.setWordWrap(True)
        form.addRow(self.label_info)

        for w in (self.spin_cell, self.spin_lw, self.spin_pore,
                  self.spin_target,
                  self.spin_rmax, self.spin_elem, self.spin_C, self.spin_n,
                  self.spin_lines):
            w.valueChanged.connect(self._update)
        self.btn_cell.clicked.connect(self._fit_cell)
        for w in (self.check_auto, self.check_filter):
            w.toggled.connect(self._update)
        self.combo_target.currentIndexChanged.connect(self._update)
        self.combo_tpms.currentIndexChanged.connect(self._tpms_changed)
        self._tpms_changed()

        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Optimize", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _dspin(lo, hi, val, suffix, step=1.0, dec=2):
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(dec)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _target(self):
        name = self.combo_target.currentData()
        if name and App.ActiveDocument:
            return App.ActiveDocument.getObject(name)
        return None

    def _bb(self):
        t = self._target()
        if t is not None:
            b = t.Shape.BoundBox
            return (b.XMin, b.YMin, b.ZMin), (b.XMax, b.YMax, b.ZMax)
        return (0.0, 0.0, 0.0), (120.0, 30.0, 40.0)

    def _tpms_changed(self, *args):
        tp = self.combo_tpms.currentData()
        law = topopt.STIFFNESS_LAW[tp]
        self.spin_C.setValue(law["C"])
        self.spin_n.setValue(law["n"])
        sl = topopt.STRENGTH_LAW[tp]
        self.spin_C2.setValue(sl["C2"])
        self.spin_m.setValue(sl["m"])
        self._update()

    def _material_changed(self, *args):
        key = self.combo_mat.currentData()
        if key is None:
            return
        d = topopt.MATERIALS[key]
        self._loading_mat = True
        self.spin_E.setValue(d["E"])
        self.spin_nu.setValue(d["nu"])
        self.spin_sy.setValue(d["sy"])
        self.spin_su.setValue(d["su"])
        self.spin_dens_m.setValue(d["dens"])
        self.spin_zf.setValue(d["z_factor"])
        self._loading_mat = False

    def _material_edited(self, *args):
        # editing a value turns the selection into "Custom"
        if self._loading_mat or self.combo_mat.currentData() is None:
            return
        key = self.combo_mat.currentData()
        d = topopt.MATERIALS[key]
        cur = dict(E=self.spin_E.value(), nu=self.spin_nu.value(),
                   sy=self.spin_sy.value(), su=self.spin_su.value(),
                   dens=self.spin_dens_m.value(), z_factor=self.spin_zf.value())
        if any(abs(cur[k] - d[k]) > 1e-6 for k in cur):
            self.combo_mat.blockSignals(True)
            self.combo_mat.setCurrentIndex(self.combo_mat.count() - 1)
            self.combo_mat.blockSignals(False)

    def feas(self):
        return topopt.feasibility(
            self.combo_tpms.currentData(), self.spin_cell.value(),
            self.spin_lw.value(), self.spin_lines.value(),
            self.spin_pore.value(), self.spin_rmax.value() / 100.0,
            bb=self._bb())

    def bounds(self):
        f = self.feas()
        return f["rho_min"], f["rho_max"]

    def _fit_cell(self, *args):
        tp = self.combo_tpms.currentData()
        tgt = self.spin_target.value() / 100.0
        w = self.spin_lw.value() * self.spin_lines.value()
        rho_lo = max(engine.RHO_MIN, 0.55 * tgt)
        cell = topopt.cell_for_min_density(rho_lo, w, tp)
        self.spin_cell.setValue(round(cell + 0.05, 1))

    def _update(self, *args):
        tp = self.combo_tpms.currentData()
        law = topopt.STIFFNESS_LAW[tp]
        f = self.feas()
        rmin, rmax = f["rho_min"], f["rho_max"]
        lo_fit, hi_fit = law["rho_fit"]
        txt = ("Printable density {:.0f}-{:.0f}%:  wall >= {:.2f} mm -> "
               "min {:.0f}%  |  pores >= {:.1f} mm -> max {:.0f}% (pore "
               "{:.1f} mm at max)  |  generator: element {:.2f} mm, "
               "~{:.1f} M points".format(
                   rmin * 100, rmax * 100, f["w_min"], rmin * 100,
                   self.spin_pore.value(), rmax * 100, f["pore_at_max"],
                   f["spacing"], f.get("points", 0) / 1e6))
        bad = (not f["ok"]) or f.get("points", 0) > topopt.MAX_GENERATOR_POINTS
        if not f["ok"]:
            txt += "\nNOT FEASIBLE: " + f["msg"]
        elif f.get("points", 0) > topopt.MAX_GENERATOR_POINTS:
            txt += ("\nThe lattice would need too fine a generation grid: "
                    "increase the cell or the minimum wall.")
        tgt_ = self.spin_target.value() / 100.0
        if f["ok"] and tgt_ - rmin < 0.03:
            txt += ("\nMean density too close to the printable minimum: "
                    "little room to optimize (use 'Fit cell to target').")
            bad = True
        self.label_range.setText(txt)
        self.label_range.setStyleSheet(
            "color: #c0392b;" if bad else "color: gray;")
        self.label_law.setText(
            "source: {}. Fitted on {:.0f}-{:.0f}%: above it the law is "
            "extrapolated.".format(law["source"], lo_fit * 100, hi_fit * 100))
        if self.check_filter.isChecked():
            self.spin_filter.setValue(self.spin_cell.value())
        self.spin_filter.setEnabled(not self.check_filter.isChecked())
        bb_min, bb_max = self._bb()
        if self.check_auto.isChecked():
            self.spin_elem.setValue(round(topopt.auto_element_size(
                bb_min, bb_max, self.spin_cell.value()), 2))
        self.spin_elem.setEnabled(not self.check_auto.isChecked())
        h = self.spin_elem.value()
        nel = 1
        for a in range(3):
            nel *= max(2, int(round((bb_max[a] - bb_min[a]) / h)))
        hints = ["Grid: ~{} elements (bounding box).".format(nel)]
        tgt = self.spin_target.value() / 100.0
        if not rmin <= tgt <= rmax:
            hints.append("WARNING: mean density outside the allowed range.")
        if nel > 150000:
            hints.append("Very heavy: increase the element size.")
        elif nel > 60000:
            hints.append("Heavy (several minutes).")
        if self.spin_filter.value() < 1.5 * h:
            hints.append("Filter radius < 1.5 elements: the map may get "
                         "checkerboard patterns.")
        self.label_info.setText(" ".join(hints))
        # FEM constraints summary
        if self._target() is None:
            self.label_cons.setText("Demo: clamped at X min, load on X max.")
            self.label_cons.setStyleSheet("color: gray;")
        elif App.ActiveDocument:
            c = collect_constraints(App.ActiveDocument)
            an = c["analysis"].Label if c["analysis"] else "no Analysis"
            ok = c["fixed"] and (c["force"] or c["pressure"])
            self.label_cons.setText(
                "FEM ({}): {} Fixed, {} Force, {} Pressure{}".format(
                    an, len(c["fixed"]), len(c["force"]),
                    len(c["pressure"]),
                    "" if ok else "  -> at least one Fixed and one load "
                    "are needed"))
            self.label_cons.setStyleSheet("" if ok else "color: #c0392b;")

    def _confirm(self):
        if self.spin_su.value() < self.spin_sy.value():
            QtWidgets.QMessageBox.warning(
                self, "LatticeFree V2",
                "The ultimate strength must be >= the yield strength.")
            return
        f = self.feas()
        if not f["ok"]:
            QtWidgets.QMessageBox.warning(
                self, "LatticeFree V2", "Lattice not printable with these "
                "limits:\n" + f["msg"])
            return
        if f.get("points", 0) > 40e6:
            QtWidgets.QMessageBox.warning(
                self, "LatticeFree V2",
                "The thinnest wall would need a generation grid of "
                "{:.0f} M points: increase the cell size or the minimum "
                "wall.".format(f["points"] / 1e6))
            return
        rmin, rmax = f["rho_min"], f["rho_max"]
        tgt = self.spin_target.value() / 100.0
        if not rmin <= tgt <= rmax:
            QtWidgets.QMessageBox.warning(
                self, "LatticeFree V2",
                "The mean density must lie within {:.0f}-{:.0f}%.".format(
                    rmin * 100, rmax * 100))
            return
        if self._target() is not None:
            c = collect_constraints(App.ActiveDocument)
            if not (c["fixed"] and (c["force"] or c["pressure"])):
                QtWidgets.QMessageBox.warning(
                    self, "LatticeFree V2",
                    "Define the loads in the FEM workbench first:\n"
                    "Analysis + Fixed constraint + Force or Pressure "
                    "constraint on the faces of the component.")
                return
        self.accept()

    def parameters(self):
        f = self.feas()
        rmin, rmax = f["rho_min"], f["rho_max"]
        return dict(
            target=self._target(), tpms=self.combo_tpms.currentData(),
            cell_size=self.spin_cell.value(), nozzle=f["w_min"],
            line_width=self.spin_lw.value(), min_lines=self.spin_lines.value(),
            min_pore=self.spin_pore.value(),
            t_shell=self.spin_shell.value(),
            rho_target=self.spin_target.value() / 100.0,
            rho_min=rmin, rho_max=rmax,
            Es=self.spin_E.value(), nu=self.spin_nu.value(),
            material=self.combo_mat.currentText(),
            sy=self.spin_sy.value(), su=self.spin_su.value(),
            dens=self.spin_dens_m.value(), z_factor=self.spin_zf.value(),
            build_axis=self.combo_axis.currentData(),
            C2=self.spin_C2.value(), m=self.spin_m.value(),
            fs_target=self.spin_fs.value(),
            C=self.spin_C.value(), n=self.spin_n.value(),
            h=self.spin_elem.value(), filter_radius=self.spin_filter.value(),
            maxit=self.spin_iter.value(),
            demo_force=self.spin_demo_force.value(),
        )


# ======================================================================
# RUN
# ======================================================================

def _demo_target(doc):
    obj = doc.getObject(DEMO_NAME)
    if obj is None:
        obj = doc.addObject("Part::Box", DEMO_NAME)
        obj.Length, obj.Width, obj.Height = 120.0, 30.0, 40.0
        obj.Label = "Demo cantilever"
        doc.recompute()
    return obj


def run(p):
    log = App.Console.PrintMessage
    doc = App.ActiveDocument or App.newDocument("LatticeFree")
    demo = p["target"] is None
    target = _demo_target(doc) if demo else p["target"]
    law = topopt.STIFFNESS_LAW[p["tpms"]]

    sep = "=" * 56
    log(sep + "\n  LatticeFree V2  -  topology optimization\n" + sep + "\n")
    log("  Component:      {}\n".format(target.Label))
    log("  TPMS:           {}  (cell {:.1f} mm)\n".format(
        engine.TPMS[p["tpms"]]["label"], p["cell_size"]))
    log("  Law:            E*/Es = {:.3f} rho^{:.3f}  [{}]\n".format(
        p["C"], p["n"], law["source"]))
    log("  Material:       {}  Es {:.0f} MPa, nu {:.2f}, yield {:.1f} MPa, "
        "ultimate {:.1f} MPa, {:.2f} g/cm3\n".format(
            p["material"], p["Es"], p["nu"], p["sy"], p["su"], p["dens"]))
    log("  Interlayer:     factor {:.2f} along {} (build axis)\n".format(
        p["z_factor"], "XYZ"[p["build_axis"]]))
    log("  Strength law:   sigma*/sigma_s = {:.3f} rho^{:.2f}\n".format(
        p["C2"], p["m"]))
    log("  Density:        mean {:.0f}%, range {:.0f}-{:.0f}%\n".format(
        p["rho_target"] * 100, p["rho_min"] * 100, p["rho_max"] * 100))
    log("  Printability:   wall >= {:.2f} mm ({} x {:.2f} mm lines), "
        "pores >= {:.1f} mm\n".format(
            p["nozzle"], p["min_lines"], p["line_width"], p["min_pore"]))
    if p["rho_max"] > law["rho_fit"][1] + 1e-6:
        App.Console.PrintWarning(
            "  Note: above {:.0f}% the stiffness law is extrapolated.\n"
            .format(law["rho_fit"][1] * 100))
    log("  Shell:          {:.2f} mm (solid, outside the design)\n".format(
        p["t_shell"]))
    log("  Filter radius:  {:.1f} mm\n".format(p["filter_radius"]))
    log(sep + "\n")

    model, bb = build_model(target, p["h"], p["t_shell"], p["Es"], p["nu"],
                            p["C"], p["n"], log)
    log("Active elements: {}, DOF: {}\n".format(model.nel, model.ndof))
    log("Loads and constraints:\n")
    if demo:
        apply_demo_loads(model, bb, p["demo_force"], log)
    else:
        apply_constraints(model, collect_constraints(doc), log)
    ftot = model.f.reshape(-1, 3).sum(axis=0)
    log("  Total resultant: ({:.1f}, {:.1f}, {:.1f}) N\n".format(*ftot))

    prog = QtWidgets.QProgressDialog(
        "Topology optimization...", "Stop", 0, p["maxit"], Gui.getMainWindow())
    prog.setWindowTitle("LatticeFree V2")
    prog.setWindowModality(QtCore.Qt.WindowModal)
    prog.setMinimumDuration(0)
    prog.setValue(0)
    QtWidgets.QApplication.processEvents()

    def cb(it, c, mr, change):
        prog.setValue(min(it, p["maxit"]))
        prog.setLabelText("Iteration {}  |  compliance {:.4g} N*mm  |  "
                          "change {:.3f}".format(it, c, change))
        QtWidgets.QApplication.processEvents()
        return not prog.wasCanceled()

    def log_flush(s):
        log(s)
        QtWidgets.QApplication.processEvents()

    try:
        res = topopt.optimize(model, p["rho_target"], p["rho_min"],
                              p["rho_max"], p["filter_radius"],
                              maxit=p["maxit"], callback=cb, log=log_flush)
    finally:
        prog.close()

    res.update(tpms=p["tpms"], target=target.Name,
               cell_size=p["cell_size"], t_shell=p["t_shell"],
               nozzle=p["nozzle"], material=p["material"],
               line_width=p["line_width"], min_lines=p["min_lines"],
               min_pore=p["min_pore"])

    # --- strength verification: optimized vs uniform lattice --------
    vkw = dict(sy=p["sy"], su=p["su"], C2=p["C2"], m=p["m"],
               z_factor=p["z_factor"], build_axis=p["build_axis"],
               dens_g_cm3=p["dens"])
    ver = topopt.verify(model, res["rho_e"], res["u"], **vkw)
    ver_u = topopt.verify(model, res["rho_uniform"], res["u_uniform"], **vkw)
    fs_grid = np.full(model.active.shape, np.nan)
    fs_grid[model.active] = ver["fs_yield"]
    res.update(fs_grid=fs_grid, fs_yield_min=ver["fs_yield_min"],
               fs_ult_min=ver["fs_ult_min"], mass_g=ver["mass_g"])
    topopt.set_result(res)

    rho = res["rho"]
    log(sep + "\n")
    log("Result{}: {} iterations in {:.0f} s\n".format(
        " (STOPPED)" if res["cancelled"] else "", res["iterations"],
        res["elapsed"]))
    log("  Density:        {:.0f}% - {:.0f}%  (mean {:.1f}%)\n".format(
        np.nanmin(rho) * 100, np.nanmax(rho) * 100, res["mean_rho"] * 100))
    log("  Compliance:     {:.4g} N*mm  (uniform lattice {:.4g})\n".format(
        res["compliance"], res["compliance_uniform"]))
    log("  Stiffness:      +{:.0f}% vs uniform lattice at equal mass\n"
        .format(100.0 * (res["compliance_uniform"] / res["compliance"] - 1.0)))
    log("  Max displacement: {:.4g} mm  (linear estimate, homogenized "
        "lattice)\n".format(res["max_disp"]))
    log("  Mass:           {:.1f} g  (shell + lattice, {:.0f} mm3, voxel "
        "estimate)\n".format(ver["mass_g"], ver["volume_mm3"]))
    log("-" * 56 + "\n  STRENGTH CHECK ({})\n".format(p["material"]))
    log("                      optimized   uniform\n")
    log("  FS yield (min)      {:8.2f}   {:8.2f}\n".format(
        ver["fs_yield_min"], ver_u["fs_yield_min"]))
    log("  FS ultimate (min)   {:8.2f}   {:8.2f}\n".format(
        ver["fs_ult_min"], ver_u["fs_ult_min"]))
    log("  FS yield lattice    {:8.2f}   {:8.2f}\n".format(
        ver["fs_yield_lattice_min"], ver_u["fs_yield_lattice_min"]))
    if p["t_shell"] > 0 and ver["fs_yield_shell_min"] < 1e8:
        log("  FS yield shell      {:8.2f}   {:8.2f}\n".format(
            ver["fs_yield_shell_min"], ver_u["fs_yield_shell_min"]))
    fs_min = ver["fs_yield_min"]
    log("  Yield load:     ~{:.3g} x applied load  (linear)\n".format(fs_min))
    log("  Critical point: ({:.1f}, {:.1f}, {:.1f}) mm, {} , {}, local "
        "density {:.0f}%\n".format(
            ver["crit_xyz"][0], ver["crit_xyz"][1], ver["crit_xyz"][2],
            "shell" if ver["crit_in_shell"] else "lattice",
            ver["crit_mode"], ver["crit_rho"] * 100))
    log("  (elements touching load/constraint nodes excluded: there the "
        "point-like load application dominates; overall min incl. "
        "them {:.2f})\n".format(ver["fs_yield_min_all"]))
    warn = App.Console.PrintWarning
    if fs_min < 1.0:
        warn("  RESULT: the part YIELDS under this load ({} elements "
             "with FS < 1).\n".format(ver["n_below_1"]))
    elif fs_min < p["fs_target"]:
        warn("  RESULT: FS {:.2f} below the target {:.1f}: raise the "
             "mean density, the shell or the cell density.\n".format(
                 fs_min, p["fs_target"]))
    else:
        log("  RESULT: verified, FS {:.2f} >= target {:.1f}.\n".format(
            fs_min, p["fs_target"]))
    log("  NB: homogenized linear check; wall buckling and fatigue are "
        "not covered.\n")

    npz, vtk = topopt.default_paths(doc.FileName or None,
                                    "topopt_" + target.Name)
    try:
        topopt.save_npz(res, npz)
        topopt.write_vtk(res, vtk)
        log("  Saved map:      {}\n                  {} (ParaView)\n"
            .format(npz, vtk))
    except Exception as e:
        App.Console.PrintWarning("Save failed: {}\n".format(e))
    log(sep + "\n")

    try:
        make_preview(doc, res)
        grp_fs = make_fs_preview(doc, res, p["fs_target"])
        grp_fs.ViewObject.Visibility = False
    except Exception as e:
        App.Console.PrintWarning("Preview not available: {}\n".format(e))
    # make the part see-through (App::Part containers have no
    # Transparency: apply it to the solids inside instead)
    try:
        objs = [target] + list(getattr(target, "OutListRecursive", []))
        for o in objs:
            vo = getattr(o, "ViewObject", None)
            if vo is not None and hasattr(vo, "Transparency") and \
                    getattr(vo, "Visibility", False):
                vo.Transparency = 80
    except Exception:
        pass
    doc.recompute()
    Gui.SendMsgToActiveView("ViewFit")

    ans = QtWidgets.QMessageBox.question(
        Gui.getMainWindow(), "LatticeFree V2",
        "Optimization done: +{:.0f}% stiffness at equal mass.\n"
        "Safety factor (yield, {}): {:.2f}  [uniform lattice: {:.2f}]\n"
        "Mass: {:.0f} g\n\n"
        "The safety-factor map is in the 'TopOpt_safety' group "
        "(hidden).\n\n"
        "Open the {} generator with this density map?".format(
            100.0 * (res["compliance_uniform"] / res["compliance"] - 1.0),
            p["material"], ver["fs_yield_min"], ver_u["fs_yield_min"],
            ver["mass_g"], engine.TPMS[p["tpms"]]["label"]),
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    if ans == QtWidgets.QMessageBox.Yes:
        from . import commands
        dlg = commands.TPMSDialog(p["tpms"], preset=dict(
            target=target.Name, cell_size=p["cell_size"],
            t_shell=p["t_shell"], nozzle=p["nozzle"], topo=True))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            commands.generate(**dlg.parameters())


# ======================================================================
# COMMAND
# ======================================================================

class CommandTopOpt:
    def GetResources(self):
        icondir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "Resources", "icons")
        icon = os.path.join(icondir, "topopt.svg")
        if not os.path.isfile(icon):
            icon = os.path.join(icondir, "freelattice.svg")
        return {
            "Pixmap": icon,
            "MenuText": "Topology optimization (variable lattice density)",
            "ToolTip": ("Computes the optimal lattice density map from "
                        "the FEM loads and constraints, then generates "
                        "the variable-density infill."),
        }

    def IsActive(self):
        return True

    def Activated(self):
        dlg = TopOptDialog()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            App.Console.PrintMessage("LatticeFree: cancelled.\n")
            return
        try:
            run(dlg.parameters())
        except Exception as e:
            import traceback
            App.Console.PrintError(
                "LatticeFree TopOpt: error:\n{}\n".format(
                    traceback.format_exc()))
            QtWidgets.QMessageBox.critical(
                Gui.getMainWindow(), "LatticeFree V2",
                "Error during optimization:\n{}".format(e))


TOPOPT_COMMANDS = ["FreeLattice_TopOpt"]
Gui.addCommand("FreeLattice_TopOpt", CommandTopOpt())
