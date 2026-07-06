# -*- coding: utf-8 -*-
"""
LatticeFree — GUI commands
===========================
Defines the workbench commands: one button per TPMS (Gyroid,
Schwarz P, Diamond), all sharing the same parameter dialog
(TPMSDialog). Generates the infill with shell, fillet, grading, mesh
cleanup and (optional) FEM export. All the math lives in engine.py;
here there is only the interface and the FreeCAD orchestration.
"""

import os
import FreeCAD as App
import FreeCADGui as Gui
import Part
import Mesh
import numpy as np

from PySide import QtCore, QtGui
try:
    from PySide import QtWidgets
except ImportError:
    QtWidgets = QtGui

from . import engine


def generate(cell_size, density, spacing, t_shell, demo_box,
           target, mode, n_smooth, riduzione, r_blend,
           fem_export, fem_elem, fem_formato,
           grad_on, grad_core_density, nozzle_mm, tpms=engine.TPMS_DEFAULT):
    log = App.Console.PrintMessage
    label = engine.TPMS[tpms]["label"]
    thickness = engine.density_to_isovalue(density, tpms)

    # ---------------------------------------------------------------
    #  Generation parameters report
    # ---------------------------------------------------------------
    sep = "=" * 56
    log(sep + "\n")
    log("  LatticeFree  -  generation parameters\n")
    log(sep + "\n")
    log("  TPMS:              {}\n".format(label))
    if target is not None:
        log("  Component:         {}\n".format(target.Label))
    else:
        log("  Component:         demo cube ({:.1f} mm)\n".format(demo_box))
    log("  Output:            {}\n".format(mode))
    log("  TPMS cell:         {:.2f} mm\n".format(cell_size))
    log("  Outer density:     {:.0f}%   (isovalue {:.3f}, "
        "wall {:.2f} mm)\n".format(
            density * 100.0, thickness,
            engine.wall_thickness_mm(thickness, cell_size, tpms)))
    if grad_on:
        log("  Grading:           ON  ({:.0f}% skin -> {:.0f}% core, "
            "nozzle {:.2f} mm)\n".format(
                density * 100.0, grad_core_density * 100.0, nozzle_mm))
    else:
        log("  Grading:           OFF\n")
    log("  Shell:             {:.2f} mm{}\n".format(
        t_shell, "" if t_shell > 0 else "   (exposed lattice)"))
    log("  Fillet:            {:.2f} mm\n".format(r_blend))
    ratio = (cell_size / spacing) if spacing > 0 else 0.0
    # thinnest wall actually present (grading included): it dictates
    # the minimum resolution (>= ~3 elements per wall)
    rho_thinnest = min(density, grad_core_density) if grad_on else density
    w_min = engine.wall_thickness_mm(
        engine.density_to_isovalue(rho_thinnest, tpms), cell_size, tpms)
    el_per_wall = (w_min / spacing) if spacing > 0 else 0.0
    log("  Grid element:      {:.3f} mm   (cell/{:.0f}, "
        "{:.1f} el./min wall)\n".format(spacing, ratio, el_per_wall))
    if el_per_wall < 2.0:
        App.Console.PrintWarning(
            "  WARNING: minimum wall {:.2f} mm < 2 elements "
            "({:.3f} mm): risk of under-sampling holes. "
            "Suggested element: <= {:.3f} mm.\n".format(
                w_min, spacing, w_min / 3.0))
    log("  Taubin smoothing:  {}\n".format(
        "{} iterations".format(n_smooth) if n_smooth > 0 else "no"))
    log("  Decimation:        {}\n".format(
        "{:.0f}%".format(riduzione) if riduzione > 0 else "no"))
    if fem_export:
        log("  FEM export:        yes  (format {}, element {:.2f} mm)\n"
            .format(fem_formato, fem_elem))
    else:
        log("  FEM export:        no\n")
    log(sep + "\n")

    doc = App.ActiveDocument or App.newDocument("LatticeFree")

    if target is not None:
        bb = target.Shape.BoundBox
        bb_min = (bb.XMin, bb.YMin, bb.ZMin)
        bb_max = (bb.XMax, bb.YMax, bb.ZMax)
        log("Shell+infill on '{}'\n".format(target.Label))
    else:
        h = demo_box / 2.0
        bb_min, bb_max = (-h, -h, -h), (h, h, h)
        log("Demo cube.\n")

    pad = 2 * spacing
    x = engine._axis_samples(bb_min[0] - pad, bb_max[0] + pad, spacing)
    y = engine._axis_samples(bb_min[1] - pad, bb_max[1] + pad, spacing)
    z = engine._axis_samples(bb_min[2] - pad, bb_max[2] + pad, spacing)
    log("Grid: {} x {} x {}\n".format(len(x), len(y), len(z)))

    log("Distance field...\n")
    if target is not None:
        defl = max(0.05, spacing * 0.5)
        pts, fcs = target.Shape.tessellate(defl)
        P = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
        F = np.array(fcs, dtype=np.int64)
        log("Tessellation: {} triangles\n".format(len(F)))
        occ = engine.occupancy_from_tris(P, F, x, y, z, spacing, log)
        band = min(max(2.5 * spacing,
                       t_shell + r_blend + 1.5 * spacing),
                   10.0 * spacing)
        sdf = engine.sdf_coarse(occ, spacing, band + 2 * spacing)
        del occ
        sdf = engine.refine_band(P, F, x, y, z, sdf, band, log)
    else:
        Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
        sdf = engine.sdf_box_demo(Xg, Yg, Zg, demo_box)
        del Xg, Yg, Zg

    log("{} field and combination...\n".format(label))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # --- thickness field: constant or graded (dense outside) ---
    if grad_on:
        # core density clamped to the printable minimum for this cell
        rho_min = engine.min_printable_density(cell_size, nozzle_mm, tpms)
        rho_int = max(grad_core_density, rho_min, engine.RHO_MIN)
        if rho_int > grad_core_density + 1e-6:
            App.Console.PrintWarning(
                "Core density {:.0f}% below the printable minimum "
                "({:.0f}% for cell {:.1f}mm, nozzle {:.2f}mm): "
                "clamped to {:.0f}%.\n".format(
                    grad_core_density * 100, rho_min * 100, cell_size,
                    nozzle_mm, rho_int * 100))
        t_ext = engine.density_to_isovalue(density, tpms)
        t_int = engine.density_to_isovalue(rho_int, tpms)
        # sdf>0 inside the part; normalized by the local half-thickness.
        # depth = clamped sdf; the full band is the max scale, but for a
        # "down to the core" gradient we normalize on the real sdf max.
        sdf_pos = np.maximum(sdf, 0.0)
        s_max = float(sdf_pos.max()) if sdf_pos.max() > 0 else 1.0
        u = engine.smoothstep(sdf_pos / s_max)        # 0 pelle, 1 cuore
        t_field = (t_ext + (t_int - t_ext) * u).astype(np.float32)
        log("Grading: density {:.0f}% (skin) -> {:.0f}% (core), "
            "wall thickness {:.2f}-{:.2f} mm\n".format(
                density * 100, rho_int * 100,
                engine.wall_thickness_mm(t_int, cell_size, tpms),
                engine.wall_thickness_mm(t_ext, cell_size, tpms)))
        gy = engine.tpms_field(X, Y, Z, cell_size, t_field, tpms)
        del t_field
    else:
        gy = engine.tpms_field(X, Y, Z, cell_size, thickness, tpms)
    del X, Y, Z
    F_field = engine.final_field(sdf, gy, t_shell, r_blend)
    del sdf, gy
    F_field += np.float32(1.0e-4) * spacing

    log("Marching tetrahedra...\n")
    triangles = engine.marching_tetrahedra(x, y, z, F_field)
    del F_field
    log("Raw triangles: {}\n".format(len(triangles)))

    log("Welding vertices...\n")
    V, T = engine.weld_vertices(triangles, spacing * 1e-3)
    del triangles
    log("Vertices: {}, triangles: {}\n".format(len(V), len(T)))

    if n_smooth > 0:
        log("Taubin smoothing ({} iterations)...\n".format(n_smooth))
        V = engine.taubin(V, T, n_smooth)

    log("Mesh cleanup (normal orientation)...\n")
    V, T = engine.clean_mesh(V, T, log=log)

    log("Building FreeCAD mesh...\n")
    soup = V[T]
    m = Mesh.Mesh([[tuple(p) for p in tri] for tri in soup.tolist()])
    # Weld the coincident points of the triangle soup: essential and
    # NON-destructive (points are bit-identical, so it only merges true
    # duplicates and rebuilds the watertight connectivity).
    m.removeDuplicatedPoints()
    # NB: we do NOT call removeDegenerations()/removeNonManifolds(): they
    # delete faces and, on an already-closed mesh, every removed face opens
    # a hole. Residual micro-holes are closed CONSTRUCTIVELY instead:
    # fillupHoles adds faces, never removes them, so it cannot open new
    # borders (the length limit only closes small artifact holes).
    if hasattr(m, "fillupHoles"):
        try:
            m.fillupHoles(200)
        except Exception:
            App.Console.PrintWarning("fillupHoles failed.\n")
    m.harmonizeNormals()

    if riduzione > 0:
        try:
            prima = m.CountFacets
            m.decimate(spacing * 0.05, riduzione / 100.0)
            log("Decimation: {} -> {} triangles\n"
                .format(prima, m.CountFacets))
        except Exception as e:
            App.Console.PrintWarning(
                "Decimation not available: {}\n".format(e))

    obj_name = "{}_ShellInfill".format(label.replace(" ", ""))
    if mode == "mesh":
        mobj = doc.addObject("Mesh::Feature", obj_name)
        mobj.Mesh = m
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            Gui.getMainWindow(), "Save STL",
            os.path.expanduser("~/{}_shell_infill.stl".format(tpms)),
            "STL (*.stl)")
        if path:
            m.write(path)
            log("Saved: {}\n".format(path))
    else:
        log("Converting to solid (SLOW)...\n")
        shape = Part.Shape()
        shape.makeShapeFromMesh(m.Topology, 0.1)
        solid = Part.makeSolid(shape)
        if solid.Volume < 0:
            solid.reverse()
        obj = doc.addObject("Part::Feature", obj_name)
        obj.Shape = solid
        if target is not None:
            target.ViewObject.Visibility = False

    # -------------------------------------------------------- FEM
    if fem_export:
        log("\n=== FEM export: mid-surface (single wall) ===\n")
        tgt_tris = tgt_P = None
        if target is not None:
            defl = max(0.05, spacing * 0.5)
            pts, fcs = target.Shape.tessellate(defl)
            tgt_P = np.array([[p.x, p.y, p.z] for p in pts],
                             dtype=np.float64)
            tgt_tris = np.array(fcs, dtype=np.int64)
        Vm, Tm = engine.mid_surface(bb_min, bb_max, cell_size, spacing,
                                  tgt_tris, tgt_P, log, tpms_type=tpms)
        log("Mid-surface: {} vertices, {} triangles\n"
            .format(len(Vm), len(Tm)))
        ext = "msh" if fem_formato == "msh" else "inp"
        file_filter = ("Gmsh mesh (*.msh)" if ext == "msh"
                  else "Abaqus/CalculiX (*.inp)")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            Gui.getMainWindow(), "Save FEM mesh",
            os.path.expanduser("~/{}_fem.{}".format(tpms, ext)), file_filter)
        if path:
            qual = engine.export_fem(Vm, Tm, fem_elem, ext, path, log)
            if qual:
                log("FEM mesh quality: {} triangles | min angle "
                    "{:.1f} deg | mean min angle {:.1f} deg | "
                    "max angle {:.1f} deg\n".format(
                        qual.get("n_tri", 0), qual.get("ang_min", 0),
                        qual.get("ang_min_medio", 0),
                        qual.get("ang_max", 0)))
                if qual.get("ang_min", 90) < 15:
                    App.Console.PrintWarning(
                        "Some triangles have angles < 15 deg: "
                        "consider a larger FEM element or another "
                        "Mesh.Algorithm.\n")

    doc.recompute()
    Gui.SendMsgToActiveView("ViewFit")
    log("Done!\n")


# ======================================================================
# DIALOG
# ======================================================================

class TPMSDialog(QtWidgets.QDialog):
    """Parameter dialog, shared by all TPMS: only the tpms_type changes
    (hence the density/wall calibration), the parameters are identical."""

    def __init__(self, tpms_type=engine.TPMS_DEFAULT):
        super(TPMSDialog, self).__init__(Gui.getMainWindow())
        self.tpms_type = tpms_type
        self.label = engine.TPMS[tpms_type]["label"]
        self.setWindowTitle("LatticeFree — {}".format(self.label))
        self.setMinimumWidth(420)
        form = QtWidgets.QFormLayout(self)

        self.combo_target = QtWidgets.QComboBox()
        self.combo_target.addItem("— Demo cube —", None)
        doc = App.ActiveDocument
        if doc:
            sel = Gui.Selection.getSelection()
            sel_name = sel[0].Name if sel else None
            for obj in doc.Objects:
                if hasattr(obj, "Shape") and obj.Shape.Volume > 0:
                    self.combo_target.addItem(obj.Label, obj.Name)
                    if obj.Name == sel_name:
                        self.combo_target.setCurrentIndex(
                            self.combo_target.count() - 1)
        form.addRow("Component:", self.combo_target)

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItem("Mesh only — 3D printing (FAST)", "mesh")
        self.combo_mode.addItem("CAD solid (SLOW)", "solido")
        form.addRow("Output:", self.combo_mode)

        self.spin_shell = QtWidgets.QDoubleSpinBox()
        self.spin_shell.setRange(0.0, 20.0)
        self.spin_shell.setSingleStep(0.2)
        self.spin_shell.setValue(1.2)
        self.spin_shell.setSuffix(" mm")
        form.addRow("Shell thickness (0 = none):", self.spin_shell)

        self.spin_cell = QtWidgets.QDoubleSpinBox()
        self.spin_cell.setRange(0.5, 500.0)
        self.spin_cell.setValue(10.0)
        self.spin_cell.setSuffix(" mm")
        form.addRow("{} cell size:".format(self.label),
                    self.spin_cell)

        self.spin_dens = QtWidgets.QDoubleSpinBox()
        self.spin_dens.setRange(engine.RHO_MIN * 100.0, engine.RHO_MAX * 100.0)
        self.spin_dens.setSingleStep(1.0)
        self.spin_dens.setValue(30.0)
        self.spin_dens.setSuffix(" %")
        form.addRow("Relative density (outer):", self.spin_dens)

        self.label_dens = QtWidgets.QLabel("")
        self.label_dens.setStyleSheet("color: gray;")
        form.addRow("", self.label_dens)

        # --- Density grading (dense outside, light inside) ---
        self.check_grad = QtWidgets.QCheckBox(
            "Grading: dense skin, light core")
        form.addRow(self.check_grad)

        self.spin_dens_int = QtWidgets.QDoubleSpinBox()
        self.spin_dens_int.setRange(engine.RHO_MIN * 100.0, engine.RHO_MAX * 100.0)
        self.spin_dens_int.setSingleStep(1.0)
        self.spin_dens_int.setValue(15.0)
        self.spin_dens_int.setSuffix(" %")
        self.spin_dens_int.setEnabled(False)
        form.addRow("  Core density:", self.spin_dens_int)

        self.spin_nozzle = QtWidgets.QDoubleSpinBox()
        self.spin_nozzle.setRange(0.05, 2.0)
        self.spin_nozzle.setSingleStep(0.05)
        self.spin_nozzle.setValue(0.20)
        self.spin_nozzle.setSuffix(" mm")
        self.spin_nozzle.setEnabled(False)
        form.addRow("  Minimum wall thickness (nozzle):", self.spin_nozzle)

        self.label_grad = QtWidgets.QLabel("")
        self.label_grad.setStyleSheet("color: gray;")
        self.label_grad.setVisible(False)
        form.addRow("", self.label_grad)

        def _toggle_grad(on):
            self.spin_dens_int.setEnabled(on)
            self.spin_nozzle.setEnabled(on)
            self.label_grad.setVisible(on)
            self._update_grading()
            self._auto_element()   # la parete min cambia col grading
        self.check_grad.toggled.connect(_toggle_grad)

        self.spin_blend = QtWidgets.QDoubleSpinBox()
        self.spin_blend.setRange(0.0, 10.0)
        self.spin_blend.setSingleStep(0.1)
        self.spin_blend.setValue(0.8)
        self.spin_blend.setSuffix(" mm")
        form.addRow("Shell-lattice fillet radius:", self.spin_blend)

        elem_row = QtWidgets.QHBoxLayout()
        self.spin_elem = QtWidgets.QDoubleSpinBox()
        self.spin_elem.setRange(0.05, 10.0)
        self.spin_elem.setSingleStep(0.05)
        self.spin_elem.setValue(round(10.0 / 15.0, 2))
        self.spin_elem.setSuffix(" mm")
        # The V0 auto was cell/15: at low densities the wall became
        # sub-grid (the "hexagonal holes"). The auto now also tracks the
        # thinnest wall: min(cell/15, min_wall/3).
        self.check_auto = QtWidgets.QCheckBox("auto (cell/15 and wall/3)")
        self.check_auto.setChecked(True)
        elem_row.addWidget(self.spin_elem)
        elem_row.addWidget(self.check_auto)
        form.addRow("Grid element size:", elem_row)

        self.spin_smooth = QtWidgets.QSpinBox()
        self.spin_smooth.setRange(0, 25)
        self.spin_smooth.setValue(8)
        form.addRow("Taubin smoothing iterations:", self.spin_smooth)

        self.spin_decim = QtWidgets.QSpinBox()
        self.spin_decim.setRange(0, 90)
        self.spin_decim.setValue(40)
        self.spin_decim.setSuffix(" %")
        form.addRow("Triangle reduction:", self.spin_decim)

        self.spin_demo = QtWidgets.QDoubleSpinBox()
        self.spin_demo.setRange(5.0, 500.0)
        self.spin_demo.setValue(30.0)
        self.spin_demo.setSuffix(" mm")
        form.addRow("Demo cube side:", self.spin_demo)

        # --- FEM export (mid-surface) ----------------------------------
        self.check_fem = QtWidgets.QCheckBox(
            "Export mid-surface for FEM (shell)")
        form.addRow(self.check_fem)

        self.spin_fem_elem = QtWidgets.QDoubleSpinBox()
        self.spin_fem_elem.setRange(0.1, 20.0)
        self.spin_fem_elem.setSingleStep(0.1)
        self.spin_fem_elem.setValue(1.0)
        self.spin_fem_elem.setSuffix(" mm")
        self.spin_fem_elem.setEnabled(False)
        form.addRow("  Surface resolution (info):", self.spin_fem_elem)

        self.combo_fem_fmt = QtWidgets.QComboBox()
        self.combo_fem_fmt.addItem("Abaqus/CalculiX (.inp)", "inp")
        self.combo_fem_fmt.addItem("Gmsh (.msh)", "msh")
        self.combo_fem_fmt.setEnabled(False)
        form.addRow("  FEM mesh format:", self.combo_fem_fmt)

        self.label_fem = QtWidgets.QLabel(
            "Exports the TPMS mid-surface (single wall) as .inp/.msh + a "
            "helper STL. Set the thickness in the solver as a shell "
            "property. For FEM-quality triangles, remesh in the solver "
            "(PrePoMax/Gmsh).")
        self.label_fem.setWordWrap(True)
        self.label_fem.setStyleSheet("color: gray;")
        self.label_fem.setVisible(False)
        form.addRow("", self.label_fem)

        def _toggle_fem(on):
            self.spin_fem_elem.setEnabled(on)
            self.combo_fem_fmt.setEnabled(on)
            self.label_fem.setVisible(on)
        self.check_fem.toggled.connect(_toggle_fem)

        self.label_info = QtWidgets.QLabel("")
        self.label_info.setWordWrap(True)
        form.addRow(self.label_info)

        self.spin_cell.valueChanged.connect(self._auto_element)
        self.check_auto.toggled.connect(self._auto_element)
        self.spin_dens.valueChanged.connect(self._update_density)
        # density (and grading) determine the thinnest wall, hence
        # also the auto element
        self.spin_dens.valueChanged.connect(self._auto_element)
        self.spin_dens_int.valueChanged.connect(self._auto_element)
        self.spin_cell.valueChanged.connect(self._update_grading)
        self.spin_dens_int.valueChanged.connect(self._update_grading)
        self.spin_nozzle.valueChanged.connect(self._update_grading)
        for w in (self.spin_elem, self.spin_demo, self.spin_shell,
                  self.spin_blend):
            w.valueChanged.connect(self._update_estimate)
        self.combo_target.currentIndexChanged.connect(self._update_estimate)
        self.combo_mode.currentIndexChanged.connect(self._update_estimate)
        self._update_density()
        self._update_grading()
        self._update_estimate()

        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Generate", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _update_density(self, *args):
        rho = self.spin_dens.value() / 100.0
        t = engine.density_to_isovalue(rho, self.tpms_type)
        rho_eff = engine.isovalue_to_density(t, self.tpms_type)
        self.label_dens.setText(
            "resulting isovalue {:.3f}  |  effective density "
            "{:.1f}%  (useful range {:.0f}-{:.0f}%)".format(
                t, rho_eff * 100.0, engine.RHO_MIN * 100.0, engine.RHO_MAX * 100.0))

    def _update_grading(self, *args):
        if not self.check_grad.isChecked():
            return
        L = self.spin_cell.value()
        noz = self.spin_nozzle.value()
        rho_min = engine.min_printable_density(L, noz, self.tpms_type)
        rho_int = self.spin_dens_int.value() / 100.0
        t_int = engine.density_to_isovalue(max(rho_int, rho_min), self.tpms_type)
        t_ext = engine.density_to_isovalue(self.spin_dens.value() / 100.0,
                                            self.tpms_type)
        w_int = engine.wall_thickness_mm(t_int, L, self.tpms_type)
        w_ext = engine.wall_thickness_mm(t_ext, L, self.tpms_type)
        warn_txt = ""
        if rho_int < rho_min - 1e-6:
            warn_txt = ("  -> below the printable minimum {:.0f}%, "
                      "will be clamped".format(rho_min * 100))
        self.label_grad.setText(
            "min printable density {:.0f}% (cell {:.1f}mm, nozzle "
            "{:.2f}mm) | wall {:.2f}-{:.2f} mm{}".format(
                rho_min * 100, L, noz, w_int, w_ext, warn_txt))

    def _thinnest_rho(self):
        """Density of the thinnest wall actually present."""
        rho = self.spin_dens.value() / 100.0
        if self.check_grad.isChecked():
            rho = min(rho, self.spin_dens_int.value() / 100.0)
        return rho

    def _auto_element(self, *args):
        if self.check_auto.isChecked():
            val = engine.recommended_spacing(
                self.spin_cell.value(), self._thinnest_rho(), self.tpms_type)
            self.spin_elem.setValue(round(val, 2))
        self._update_estimate()

    def _current_bb(self):
        nome = self.combo_target.currentData()
        if nome and App.ActiveDocument:
            obj = App.ActiveDocument.getObject(nome)
            if obj:
                bb = obj.Shape.BoundBox
                return ((bb.XMin, bb.YMin, bb.ZMin),
                        (bb.XMax, bb.YMax, bb.ZMax))
        h = self.spin_demo.value() / 2.0
        return ((-h, -h, -h), (h, h, h))

    def _n_points(self):
        bb_min, bb_max = self._current_bb()
        return engine.estimate_points(bb_min, bb_max, self.spin_elem.value())

    def _update_estimate(self, *args):
        n = self._n_points()
        mode = self.combo_mode.currentData()
        hints = ["Grid: ~{:.1f} M points.".format(n / 1e6)]
        sp = self.spin_elem.value()
        # under-sampling hole tell-tale: min wall < 2 elements
        t_min = engine.density_to_isovalue(self._thinnest_rho(),
                                            self.tpms_type)
        w_min = engine.wall_thickness_mm(t_min, self.spin_cell.value(),
                                          self.tpms_type)
        if sp > 0 and w_min < 2.0 * sp:
            hints.append("WARNING: min wall {:.2f} mm < 2 elements "
                          "-> under-sampling holes; use element "
                          "<= {:.3f} mm (or auto).".format(w_min, w_min / 3.0))
        sh = self.spin_shell.value()
        if 0.0 < sh < 2.5 * sp:
            hints.append("Shell too thin for the grid: use element "
                          "<= {:.2f} mm.".format(sh / 2.5))
        rb = self.spin_blend.value()
        if 0.0 < rb < 2.0 * sp:
            hints.append("Fillet < 2 elements: barely visible, "
                          "use radius >= {:.2f} mm.".format(2.0 * sp))
        if n > 40e6:
            hints.append("TOO MANY points.")
        elif mode == "solido" and n > 3e6:
            hints.append("CAD solid: VERY slow, consider Mesh only.")
        elif n > 15e6:
            hints.append("Heavy but feasible as mesh.")
        else:
            hints.append("OK.")
        if not engine.HAVE_SCIPY:
            hints.append("(scipy missing: far field approximated, "
                          "near-surface band still exact)")
        self.label_info.setText(" ".join(hints))

    def _confirm(self):
        if self._n_points() > 40e6:
            QtWidgets.QMessageBox.warning(
                self, "Grid too fine",
                "Over 40 million points: increase the element "
                "size.")
            return
        self.accept()

    def parameters(self):
        nome = self.combo_target.currentData()
        target = None
        if nome and App.ActiveDocument:
            target = App.ActiveDocument.getObject(nome)
        return dict(
            cell_size=self.spin_cell.value(),
            density=self.spin_dens.value() / 100.0,
            spacing=self.spin_elem.value(),
            t_shell=self.spin_shell.value(),
            demo_box=self.spin_demo.value(),
            target=target,
            mode=self.combo_mode.currentData(),
            n_smooth=self.spin_smooth.value(),
            riduzione=self.spin_decim.value(),
            r_blend=self.spin_blend.value(),
            fem_export=self.check_fem.isChecked(),
            fem_elem=self.spin_fem_elem.value(),
            fem_formato=self.combo_fem_fmt.currentData(),
            grad_on=self.check_grad.isChecked(),
            grad_core_density=self.spin_dens_int.value() / 100.0,
            nozzle_mm=self.spin_nozzle.value(),
            tpms=self.tpms_type,
        )


# V0 compatibility
GyroidDialog = TPMSDialog

# ======================================================================
# FREECAD COMMANDS
# ======================================================================

class CommandGenerateTPMS:
    """'Generate <TPMS> infill' command: one button per tpms_type,
    same dialog and same parameters for all."""

    def __init__(self, tpms_type, icona):
        self.tpms_type = tpms_type
        self.icona = icona
        self.label = engine.TPMS[tpms_type]["label"]

    def GetResources(self):
        icondir = os.path.join(App.getUserAppDataDir(), "Mod",
                               "LatticeFree", "Resources", "icons")
        icon = os.path.join(icondir, self.icona)
        if not os.path.isfile(icon):
            icon = os.path.join(icondir, "freelattice.svg")
        return {
            "Pixmap": icon,
            "MenuText": "Generate {} infill".format(self.label),
            "ToolTip": ("Fills a solid with a {} infill: shell, "
                        "fillet, density, grading, FEM export."
                        .format(self.label)),
        }

    def IsActive(self):
        return App.ActiveDocument is not None or True

    def Activated(self):
        dlg = TPMSDialog(self.tpms_type)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            try:
                generate(**dlg.parameters())
            except Exception as e:
                import traceback
                App.Console.PrintError(
                    "LatticeFree: error during generation:\n{}\n"
                    .format(traceback.format_exc()))
                QtWidgets.QMessageBox.critical(
                    Gui.getMainWindow(), "LatticeFree",
                    "Error during generation:\n{}".format(e))
        else:
            App.Console.PrintMessage("LatticeFree: cancelled.\n")


TPMS_COMMANDS = ["FreeLattice_Gyroid", "FreeLattice_SchwarzP",
                "FreeLattice_Diamond"]

Gui.addCommand("FreeLattice_Gyroid",
               CommandGenerateTPMS("gyroid", "tpms_gyroid.svg"))
Gui.addCommand("FreeLattice_SchwarzP",
               CommandGenerateTPMS("schwarz_p", "tpms_schwarz_p.svg"))
Gui.addCommand("FreeLattice_Diamond",
               CommandGenerateTPMS("diamond", "tpms_diamond.svg"))
# V0 compatibility: il vecchio nome apre il giroide
Gui.addCommand("FreeLattice_Genera",
               CommandGenerateTPMS("gyroid", "tpms_gyroid.svg"))
