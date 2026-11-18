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

def keep_tris_from_constraints(cons, h):
    """Triangulated faces of all FEM constraints (Fixed, Force, Pressure):
    around them the part stays SOLID (bolt holes, seats, load pads)."""
    Ps, Ts, off = [], [], 0
    for o in cons["fixed"] + cons["force"] + cons["pressure"]:
        for sh in _sub_shapes(o):
            if sh.ShapeType != "Face":
                continue
            pts, tris = sh.tessellate(max(0.02, 0.05 * h))
            if not tris:
                continue
            Ps.append(np.array([[q.x, q.y, q.z] for q in pts]))
            Ts.append(np.array(tris, dtype=np.int64) + off)
            off += len(pts)
    if not Ps:
        return None, None
    return np.vstack(Ps), np.vstack(Ts)


def demo_keep_tris(bb):
    P, T = [], []
    for k, x in enumerate((bb.XMin, bb.XMax)):
        P += [[x, bb.YMin, bb.ZMin], [x, bb.YMax, bb.ZMin],
              [x, bb.YMax, bb.ZMax], [x, bb.YMin, bb.ZMax]]
        T += [[4 * k, 4 * k + 1, 4 * k + 2], [4 * k, 4 * k + 2, 4 * k + 3]]
    return np.array(P, float), np.array(T, np.int64)


def build_part(target, h, t_shell, log):
    """Voxel grid of the part: node axes, occupancy and signed distance
    of the element centres."""
    bb = target.Shape.BoundBox
    bb_min = (bb.XMin, bb.YMin, bb.ZMin)
    bb_max = (bb.XMax, bb.YMax, bb.ZMax)
    xn, yn, zn = topopt.make_grid(bb_min, bb_max, h)
    xc, yc, zc = topopt.centers(xn), topopt.centers(yn), topopt.centers(zn)
    hm = min(xn[1] - xn[0], yn[1] - yn[0], zn[1] - zn[0])
    log("FE grid: {} x {} x {} elements ({:.2f} mm)\n".format(
        len(xc), len(yc), len(zc), hm))
    pts, fcs = target.Shape.tessellate(max(0.05, 0.25 * hm))
    P = np.array([[q.x, q.y, q.z] for q in pts], dtype=np.float64)
    F = np.array(fcs, dtype=np.int64)
    occ = engine.occupancy_from_tris(P, F, xc, yc, zc, hm)
    if not occ.any():
        raise ValueError("No element inside '{}': is it a closed solid? "
                         "Try a smaller FE element.".format(target.Label))
    occ_p = np.pad(occ, 1, mode="constant", constant_values=False)
    sdf = engine.sdf_coarse(occ_p, hm, t_shell + 3.0 * hm)[1:-1, 1:-1, 1:-1]
    if engine.HAVE_SCIPY:
        sdf = sdf - 0.5 * hm
    return dict(xn=xn, yn=yn, zn=zn, occ=occ, sdf=sdf, hm=hm, bb=bb)


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

        # --- Shape (stage 1) and solid zones -----------------------
        self.check_shape = QtWidgets.QCheckBox(
            "Stage 1: find the outer shape (remove the volume that "
            "does not carry load)")
        form.addRow(self.check_shape)
        self.spin_vf = self._dspin(10.0, 90.0, 40.0, " %", 5.0, 0)
        self.spin_vf.setToolTip(
            "Stage 1 removes the material that carries little load and "
            "keeps this percentage of the ORIGINAL part volume as the "
            "outer shape. The lattice then fills that shape, so the final "
            "mass is much lower. Lower = lighter but less stiff.")
        form.addRow("  Material kept by stage 1 (% of the part):",
                    self.spin_vf)
        self.check_keep = QtWidgets.QCheckBox(
            "Solid, no lattice, around the FEM load/constraint faces "
            "(bolt holes, seats)")
        self.check_keep.setChecked(True)
        form.addRow(self.check_keep)
        self.spin_ring = self._dspin(0.5, 30.0, 3.0, " mm", 0.5, 1)
        form.addRow("  Solid ring thickness:", self.spin_ring)

        self.combo_obj = QtWidgets.QComboBox()
        self.combo_obj.addItem("Max stiffness for a given mass", "stiffness")
        self.combo_obj.addItem("Min mass for the target safety factor",
                               "strength")
        self.combo_obj.addItem("Min mass for a required stiffness",
                               "stiffness_target")
        form.addRow("Objective:", self.combo_obj)
        self.spin_k = self._dspin(0.1, 1.0e7, 1000.0, " N/mm", 100.0, 1)
        self.label_k = QtWidgets.QLabel("Required stiffness (at the load):")
        form.addRow(self.label_k, self.spin_k)
        self.spin_target = self._dspin(5.0, 70.0, 25.0, " %", 1.0)
        self.label_target = QtWidgets.QLabel("Mean lattice density (mass):")
        form.addRow(self.label_target, self.spin_target)
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
        self.combo_obj.currentIndexChanged.connect(self._update)
        self.spin_fs.valueChanged.connect(self._update)
        self.spin_k.valueChanged.connect(self._update)
        for w in (self.check_shape, self.check_keep):
            w.toggled.connect(self._update)
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

    def strength_mode(self):
        """True when the solver searches the mass (strength or
        stiffness target) instead of using the given mean density."""
        return self.combo_obj.currentData() in ("strength",
                                                "stiffness_target")

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
        smode = self.strength_mode()
        self.spin_target.setEnabled(not smode)
        self.label_target.setEnabled(not smode)
        kmode = self.combo_obj.currentData() == "stiffness_target"
        self.spin_k.setEnabled(kmode)
        self.label_k.setEnabled(kmode)
        self.spin_vf.setEnabled(self.check_shape.isChecked())
        self.spin_ring.setEnabled(self.check_keep.isChecked())
        if kmode:
            txt += ("\nObjective: the solver searches the LIGHTEST "
                    "lattice whose stiffness under the applied load is "
                    ">= {:.0f} N/mm (several optimizations: slower)."
                    .format(self.spin_k.value()))
        elif smode:
            txt += ("\nObjective: the solver searches the LIGHTEST "
                    "lattice whose safety factor is >= {:.1f} (several "
                    "optimizations: slower).".format(self.spin_fs.value()))
        if self.check_shape.isChecked():
            txt += ("\nStage 1 on: the outer shape keeps {:.0f}% of the "
                    "part volume, members >= {:.0f} mm (one cell).".format(
                        self.spin_vf.value(), self.spin_cell.value()))
        tgt_ = self.spin_target.value() / 100.0
        if f["ok"] and not smode and tgt_ - rmin < 0.03:
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
        if not self.strength_mode() and not rmin <= tgt <= rmax:
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
        if not self.strength_mode() and not rmin <= tgt <= rmax:
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
            objective=self.combo_obj.currentData(),
            k_target=self.spin_k.value(),
            shape_on=self.check_shape.isChecked(),
            shape_vf=self.spin_vf.value() / 100.0,
            keep_on=self.check_keep.isChecked(),
            keep_ring=self.spin_ring.value(),
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
    kmode = p.get("objective") == "stiffness_target"
    smode = p.get("objective") == "strength"
    sized = smode or kmode
    if kmode:
        log("  Objective:      MIN MASS with stiffness >= {:.0f} N/mm\n"
            .format(p["k_target"]))
        log("  Density:        range {:.0f}-{:.0f}% (mean found by the "
            "solver)\n".format(p["rho_min"] * 100, p["rho_max"] * 100))
    elif smode:
        log("  Objective:      MIN MASS with lattice FS >= {:.2f}\n".format(
            p["fs_target"]))
        log("  Density:        range {:.0f}-{:.0f}% (mean found by the "
            "solver)\n".format(p["rho_min"] * 100, p["rho_max"] * 100))
    else:
        log("  Objective:      max stiffness for the given mass\n")
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
    if p.get("shape_on"):
        log("  Stage 1:        outer shape, {:.0f}% of the part volume\n"
            .format(p["shape_vf"] * 100))
    if p.get("keep_on"):
        log("  Solid zones:    {:.1f} mm around load/constraint faces\n"
            .format(p["keep_ring"]))
    log(sep + "\n")

    nsteps = 10 if sized else p["maxit"]
    prog = QtWidgets.QProgressDialog(
        "Topology optimization...", "Stop", 0, nsteps, Gui.getMainWindow())
    prog.setWindowTitle("LatticeFree V2")
    prog.setWindowModality(QtCore.Qt.WindowModal)
    prog.setMinimumDuration(0)
    prog.setValue(0)
    QtWidgets.QApplication.processEvents()

    def log_flush(s):
        log(s)
        QtWidgets.QApplication.processEvents()

    def cb1(it, c, vol, change, beta=1.0):
        k = int(round(np.log2(max(beta, 1.0)))) + 1
        prog.setValue(min(it, 200))
        prog.setLabelText(
            "Stage 1 - shape: sharpening step {}/5, iteration {}\n"
            "compliance {:.4g} N*mm  |  change {:.3f}\n"
            "(compliance jumps at each step: it is expected)".format(
                k, it, c, change))
        QtWidgets.QApplication.processEvents()
        return not prog.wasCanceled()

    part = build_part(target, p["h"], p["t_shell"], log)
    xn, yn, zn, hm, bb = part["xn"], part["yn"], part["zn"], part["hm"], \
        part["bb"]
    occ, sdf = part["occ"], part["sdf"]
    cons = None if demo else collect_constraints(doc)

    def apply_loads(mdl, lg):
        if demo:
            apply_demo_loads(mdl, bb, p["demo_force"], lg)
        else:
            apply_constraints(mdl, cons, lg)

    # --- solid zones around the connections -------------------------
    keep_P = keep_T = keep_pts = None
    if p.get("keep_on") and p.get("keep_ring", 0) > 0:
        keep_P, keep_T = (demo_keep_tris(bb) if demo
                          else keep_tris_from_constraints(cons, hm))
        if keep_P is not None:
            keep_pts, _, _ = topopt.sample_triangles(keep_P, keep_T, 0.5 * hm)

    # --- stage 1: outer shape ---------------------------------------
    shape_grid = None
    domain, sdf2 = occ, sdf
    if p.get("shape_on"):
        prog.setMaximum(200)
        ms = topopt.Model(xn, yn, zn, occ, np.zeros(occ.shape), p["Es"],
                          p["nu"], 1.0, 1.0)
        apply_loads(ms, lambda s: None)
        keep_s = (topopt.keep_mask_from_points(ms, keep_pts, p["keep_ring"])
                  if keep_pts is not None else None)
        R1 = max(p["cell_size"], 2.0 * hm)
        ratio = topopt.shape_size_check(
            (bb.XMin, bb.YMin, bb.ZMin), (bb.XMax, bb.YMax, bb.ZMax), R1)
        if ratio < 3.0:
            App.Console.PrintWarning(
                "  STAGE 1: the part is only {:.1f} members thick (member "
                ">= {:.1f} mm = one cell): there is little room for a "
                "truss-like shape. Use a smaller cell or skip stage 1.\n"
                .format(ratio, R1))
        log("Stage 1: outer shape (solid/void SIMP, volume {:.0f}%, "
            "members >= {:.1f} mm)...\n".format(p["shape_vf"] * 100, R1))
        xp, its1 = topopt.simp_shape(ms, p["shape_vf"], R1, keep=keep_s,
                                     callback=cb1, log=log_flush)
        if prog.wasCanceled():
            prog.close()
            App.Console.PrintWarning("LatticeFree: stopped in stage 1.\n")
            return
        shape_grid = topopt.grid_from_elements(ms, xp, fill=0.0)
        domain = occ & (shape_grid >= 0.5)
        if keep_s is not None:
            domain |= topopt.grid_from_elements(
                ms, keep_s.astype(float), fill=0.0) > 0.5
        sdf2 = np.minimum(sdf, topopt.shape_distance(shape_grid, hm))
        # stiffness kept by the (solid) shape vs the full solid part
        try:
            m_full = topopt.Model(xn, yn, zn, occ, np.zeros(occ.shape),
                                  p["Es"], p["nu"], 1.0, 1.0)
            apply_loads(m_full, lambda s: None)
            m_shp = topopt.Model(xn, yn, zn, domain, np.zeros(occ.shape),
                                 p["Es"], p["nu"], 1.0, 1.0)
            apply_loads(m_shp, lambda s: None)
            k_full = topopt.load_stiffness(m_full, m_full.solve(
                np.full(m_full.nel, p["Es"]), tol=1e-5)[0])[0]
            k_shp = topopt.load_stiffness(m_shp, m_shp.solve(
                np.full(m_shp.nel, p["Es"]), tol=1e-5)[0])[0]
            log("  Shape check (both fully solid): {:.0f}% of the volume "
                "keeps {:.0f}% of the stiffness ({:.4g} vs {:.4g} N/mm)\n"
                .format(100.0 * domain.sum() / occ.sum(),
                        100.0 * k_shp / max(k_full, 1e-30), k_shp, k_full))
            if k_shp < 0.15 * k_full:
                App.Console.PrintWarning(
                    "  The shape lost most of the stiffness: raise the "
                    "shape volume or use a smaller cell.\n")
        except Exception as e:
            App.Console.PrintWarning("  Shape check skipped: {}\n".format(e))
        log("Stage 1 done in {} iterations: the shape keeps {:.0f}% of the "
            "part ({} of {} elements).\n".format(
                its1, 100.0 * domain.sum() / occ.sum(), int(domain.sum()),
                int(occ.sum())))
        prog.setValue(0)
        prog.setMaximum(nsteps)

    # --- stage 2 model: lattice inside the shape ---------------------
    phi = topopt.shell_fraction(sdf2, p["t_shell"], hm)
    model = topopt.Model(xn, yn, zn, domain, phi, p["Es"], p["nu"],
                         p["C"], p["n"])
    if keep_pts is not None:
        keep2 = topopt.keep_mask_from_points(model, keep_pts, p["keep_ring"])
        model.phi_e[keep2] = 1.0
        # the FE model treats whole elements as solid: the generated solid
        # zone must match it, or a thin lattice sliver would remain
        k_lay = int(round(p["keep_ring"] / hm))
        keep_eff = max(p["keep_ring"], (k_lay + 1) * hm)
        log("Solid zones: {} elements, {:.1f} mm around the load/"
            "constraint faces (requested {:.1f} mm, rounded to the FE "
            "grid)\n".format(int(keep2.sum()), keep_eff, p["keep_ring"]))
    log("Active elements: {}, DOF: {}\n".format(model.nel, model.ndof))
    thin, t_med = topopt.local_thickness_check(sdf2, domain, p["cell_size"])
    if thin > 0.25:
        App.Console.PrintWarning(
            "  CELL FIT: in {:.0f}% of the part the local thickness is below "
            "one cell ({:.1f} mm; median thickness {:.1f} mm): the lattice "
            "there has few cells and is less stiff than predicted. Use a "
            "smaller cell or a larger part.\n".format(
                thin * 100, p["cell_size"], t_med))
    else:
        log("  Cell fit: OK (median local thickness {:.1f} mm vs cell "
            "{:.1f} mm)\n".format(t_med, p["cell_size"]))
    log("Loads and constraints:\n")
    apply_loads(model, log)
    ftot = model.f.reshape(-1, 3).sum(axis=0)
    log("  Total resultant: ({:.1f}, {:.1f}, {:.1f}) N\n".format(*ftot))
    k0, Fres = topopt.load_stiffness(model, np.zeros(model.ndof))

    def cb(it, c, mr, change):
        prog.setValue(min(it, nsteps))
        if kmode:
            prog.setLabelText("Step {}  |  mean density {:.1f}%  |  "
                              "stiffness {:.4g} N/mm".format(
                                  it, mr * 100, change))
        elif smode:
            prog.setLabelText("Step {}  |  mean density {:.1f}%  |  "
                              "lattice FS {:.2f}".format(it, mr * 100, change))
        else:
            prog.setLabelText("Iteration {}  |  compliance {:.4g} N*mm  |  "
                              "change {:.3f}".format(it, c, change))
        QtWidgets.QApplication.processEvents()
        return not prog.wasCanceled()

    try:
        if kmode:
            log("Searching the lightest lattice with stiffness >= {:.0f} "
                "N/mm:\n".format(p["k_target"]))
            res = topopt.size_for_stiffness(
                model, p["k_target"], p["rho_min"], p["rho_max"],
                p["filter_radius"], maxit=min(p["maxit"], 40), callback=cb,
                log=log_flush)
        elif smode:
            log("Searching the lightest lattice with FS >= {:.2f}:\n".format(
                p["fs_target"]))
            res = topopt.size_for_strength(
                model, p["fs_target"], p["sy"], p["C2"], p["m"],
                p["rho_min"], p["rho_max"], p["filter_radius"],
                z_factor=p["z_factor"], build_axis=p["build_axis"],
                maxit=min(p["maxit"], 40), callback=cb, log=log_flush)
        else:
            res = topopt.optimize(model, p["rho_target"], p["rho_min"],
                                  p["rho_max"], p["filter_radius"],
                                  maxit=p["maxit"], callback=cb,
                                  log=log_flush)
    finally:
        prog.close()

    res.update(tpms=p["tpms"], target=target.Name,
               cell_size=p["cell_size"], t_shell=p["t_shell"],
               nozzle=p["nozzle"], material=p["material"],
               line_width=p["line_width"], min_lines=p["min_lines"],
               min_pore=p["min_pore"])
    if shape_grid is not None:
        res["shape"] = np.where(occ, shape_grid, 0.0)
    if keep_P is not None:
        res.update(keep_P=keep_P, keep_T=keep_T,
                   keep_ring=keep_eff if keep_pts is not None
                   else p["keep_ring"])

    # --- strength verification: optimized vs uniform lattice --------
    vkw = dict(sy=p["sy"], su=p["su"], C2=p["C2"], m=p["m"],
               z_factor=p["z_factor"], build_axis=p["build_axis"],
               dens_g_cm3=p["dens"])
    ver = topopt.verify(model, res["rho_e"], res["u"], **vkw)
    ver_u = topopt.verify(model, res["rho_uniform"], res["u_uniform"], **vkw)
    fs_grid = np.full(model.active.shape, np.nan)
    fs_grid[model.active] = ver["fs_yield"]
    res.update(fs_grid=fs_grid, fs_yield_min=ver["fs_yield_min"],
               fs_ult_min=ver["fs_ult_min"], mass_g=ver["mass_g"],
               mass_pred_g=ver["mass_g"], volume_pred_mm3=ver["volume_mm3"],
               dens=p["dens"])
    topopt.set_result(res)

    rho = res["rho"]
    log(sep + "\n")
    log("Result{}: {} {} in {:.0f} s\n".format(
        " (STOPPED)" if res["cancelled"] else "", res["iterations"],
        "steps" if sized else "iterations", res["elapsed"]))
    log("  Density:        {:.0f}% - {:.0f}%  (mean {:.1f}%)\n".format(
        np.nanmin(rho) * 100, np.nanmax(rho) * 100, res["mean_rho"] * 100))
    log("  Compliance:     {:.4g} N*mm  (uniform lattice {:.4g})\n".format(
        res["compliance"], res["compliance_uniform"]))
    stiff = 100.0 * (res["compliance_uniform"] / res["compliance"] - 1.0)
    k_now, _ = topopt.load_stiffness(model, res["u"])
    log("  Stiffness at the load: {:.4g} N/mm  ({:.4g} mm under {:.0f} N)\n"
        .format(k_now, Fres / k_now if k_now > 0 else 0.0, Fres))
    if kmode:
        log("  Stiffness:      {:+.0f}% vs uniform lattice with the same "
            "stiffness (compliance-based)\n".format(stiff))
    elif smode:
        log("  Stiffness:      {:+.0f}% vs uniform lattice with the same "
            "FS\n".format(stiff))
    else:
        log("  Stiffness:      +{:.0f}% vs uniform lattice at equal mass\n"
            .format(stiff))
    log("  Max displacement: {:.4g} mm  (linear estimate, homogenized "
        "lattice)\n".format(res["max_disp"]))
    log("  Mass:           {:.1f} g  (shell + lattice, {:.0f} mm3, voxel "
        "estimate)\n".format(ver["mass_g"], ver["volume_mm3"]))
    if sized and res.get("feasible") and \
            res["mean_rho"] <= p["rho_min"] + 0.005:
        App.Console.PrintWarning(
            "  The target is already met at the MINIMUM printable density "
            "({:.0f}%): the lattice cannot get lighter. To save more mass "
            "reduce the shape volume (stage 1), use a larger cell (lower "
            "minimum density) or a thinner shell.\n".format(
                p["rho_min"] * 100))
    if kmode:
        if not res["feasible"]:
            App.Console.PrintWarning(
                "  NOT ACHIEVABLE: even at the maximum density {:.0f}% the "
                "stiffness is {:.4g} < {:.4g} N/mm. Increase the shell, the "
                "maximum density or the shape volume.\n".format(
                    p["rho_max"] * 100, res["k_found"], p["k_target"]))
        elif res["uniform_feasible"]:
            log("  Uniform lattice with the same stiffness: density {:.0f}%, "
                "{:.1f} g  ->  mass saved {:.0f}% ({:.1f} g)\n".format(
                    res["rho_uniform_equiv"] * 100, ver_u["mass_g"],
                    100.0 * (1.0 - ver["mass_g"] / ver_u["mass_g"]),
                    ver_u["mass_g"] - ver["mass_g"]))
        else:
            log("  A uniform lattice cannot reach {:.0f} N/mm within {:.0f}% "
                "density: only the graded map does.\n".format(
                    p["k_target"], p["rho_max"] * 100))
    if smode:
        if not res["feasible"]:
            App.Console.PrintWarning(
                "  NOT ACHIEVABLE: even at the maximum density {:.0f}% the "
                "lattice FS is {:.2f} < {:.2f}. Increase the shell, the "
                "maximum density or use a stronger material.\n".format(
                    p["rho_max"] * 100, res["fs_lattice_min"],
                    p["fs_target"]))
        elif res["uniform_feasible"]:
            log("  Uniform lattice with the same FS: density {:.0f}%, "
                "{:.1f} g  ->  mass saved {:.0f}% ({:.1f} g)\n".format(
                    res["rho_uniform_equiv"] * 100, ver_u["mass_g"],
                    100.0 * (1.0 - ver["mass_g"] / ver_u["mass_g"]),
                    ver_u["mass_g"] - ver["mass_g"]))
        else:
            log("  A uniform lattice cannot reach FS {:.2f} within {:.0f}% "
                "density: only the graded map does.\n".format(
                    p["fs_target"], p["rho_max"] * 100))
    log("-" * 56 + "\n  STRENGTH CHECK ({})\n".format(p["material"]))
    log("                      optimized   uniform{}\n".format(
        " (same FS)" if smode else (" (same k)" if kmode else "")))
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
        if shape_grid is not None:
            g = _cube_bands(doc, res, np.where(domain, 1.0, np.nan),
                            [(0.5, 2.0, (0.70, 0.72, 0.78),
                              "stage-1 shape", 0.9)], "TopOpt_shape")
            g.ViewObject.Visibility = False
        if keep_pts is not None and model.phi_e.max() >= 1.0:
            kg = np.full(model.active.shape, np.nan)
            kg[model.active] = np.where(model.phi_e >= 1.0, 1.0, np.nan)
            _cube_bands(doc, res, kg, [(0.5, 2.0, (0.95, 0.55, 0.15),
                                        "solid zones (no lattice)", 0.95)],
                        "TopOpt_solid")
    except Exception as e:
        App.Console.PrintWarning("Shape preview not available: {}\n"
                                 .format(e))
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

    if kmode and res.get("feasible") and res.get("uniform_feasible"):
        head = ("Lightest lattice with stiffness >= {:.0f} N/mm: {:.0f} g, "
                "{:.0f}% lighter than a uniform lattice with the same "
                "stiffness.\n".format(
                    p["k_target"], ver["mass_g"],
                    100.0 * (1.0 - ver["mass_g"] / ver_u["mass_g"])))
    elif kmode:
        head = "Stiffness sizing done (see the report view).\n"
    elif smode and res.get("feasible") and res.get("uniform_feasible"):
        head = ("Lightest lattice with FS >= {:.1f}: {:.0f} g, {:.0f}% "
                "lighter than a uniform lattice with the same FS.\n".format(
                    p["fs_target"], ver["mass_g"],
                    100.0 * (1.0 - ver["mass_g"] / ver_u["mass_g"])))
    elif smode:
        head = "Strength sizing done (see the report view).\n"
    else:
        head = "Optimization done: +{:.0f}% stiffness at equal mass.\n".format(
            stiff)
    ans = QtWidgets.QMessageBox.question(
        Gui.getMainWindow(), "LatticeFree V2",
        head +
        "Safety factor (yield, {}): {:.2f}  [uniform lattice: {:.2f}]\n"
        "Mass: {:.0f} g\n\n"
        "The safety-factor map is in the 'TopOpt_safety' group "
        "(hidden).\n\n"
        "Open the {} generator with this density map?".format(
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
