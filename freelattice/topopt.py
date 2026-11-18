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
LatticeFree — topology optimization for variable-density TPMS lattices
======================================================================
Computes the optimal relative-density field rho(x) of a TPMS lattice
infill (minimum compliance = maximum stiffness for a given mass),
which the generator then turns into a variable wall thickness.

Unlike classic SIMP (which penalizes intermediate densities to obtain
solid/void), intermediate densities here ARE the lattice: the
stiffness of every element follows the homogenized Gibson-Ashby law
of the TPMS

        E(rho) = Es * C * rho^n          rho in [rho_min, rho_max]

with C, n from the lattice characterization (gyroid: C = 0.414,
n = 1.386, Mastrofini 2022, shell FEA of a cubic cell). rho_min is
the printable minimum (wall >= nozzle), rho_max keeps the pores open.

Model
-----
- Structured grid of 8-node hexahedra (H8, 2x2x2 Gauss) on the part
  bounding box; only the elements whose centre is inside the part are
  active. Element size is unrelated to the marching grid: the density
  varies on the scale of the TPMS cell, so a coarse grid is enough.
- Outer shell (solid): each element carries a fraction phi of solid
  material, estimated from the distance to the surface, so the skin
  stiffness is accounted for and excluded from the design.
- Matrix-free solver: preconditioned (Jacobi) conjugate gradient,
  warm-started from the previous iteration. Only numpy is required.
- Density filter (cone, radius R ~ TPMS cell): the density cannot
  vary faster than one cell, which is also what makes the result
  printable as a lattice. Implemented as an FFT convolution.
- Update: Optimality Criteria with move limit and bisection on the
  mean-density (mass) constraint.

No Qt and no FreeCAD dependency: everything here is pure numpy and
testable outside FreeCAD. The FreeCAD/GUI side is in topopt_gui.py.

Units: mm, N, MPa  ->  displacements in mm, compliance in N*mm.
"""

import os
import time
import numpy as np

# ----------------------------------------------------------------------
# Homogenized stiffness laws E*/Es = C * rho^n per TPMS.
# rho_fit = density range of the fit: outside it the law is an
# extrapolation (reported in the log).
# ----------------------------------------------------------------------
STIFFNESS_LAW = {
    "gyroid": dict(
        C=0.414, n=1.386, rho_fit=(0.064, 0.322),
        source="Mastrofini 2022 (shell FEA, 20 mm cell)"),
    # No dedicated characterization yet: gyroid values as a starting
    # point. Override them in the dialog when you have measured data.
    "schwarz_p": dict(
        C=0.414, n=1.386, rho_fit=(0.064, 0.322),
        source="PROVISIONAL: gyroid values, to be calibrated"),
    "diamond": dict(
        C=0.414, n=1.386, rho_fit=(0.064, 0.322),
        source="PROVISIONAL: gyroid values, to be calibrated"),
}

# ----------------------------------------------------------------------
# Homogenized strength laws  sigma*/sigma_s = C2 * rho^m  per TPMS.
# Gibson-Ashby form for sheet/bending-dominated cellular solids
# (C2 ~ 0.3, m = 1.5): conservative default. NB: the yield fit in
# Mastrofini 2022 is weak (the paper's own C2, m do not reproduce its
# table), so it is NOT used here. Override with test data if available.
# ----------------------------------------------------------------------
STRENGTH_LAW = {
    "gyroid": dict(C2=0.30, m=1.5,
                   source="Gibson-Ashby, sheet TPMS (indicative)"),
    "schwarz_p": dict(C2=0.30, m=1.5,
                      source="Gibson-Ashby (indicative)"),
    "diamond": dict(C2=0.30, m=1.5,
                    source="Gibson-Ashby (indicative)"),
}

# ----------------------------------------------------------------------
# FDM material library: TYPICAL indicative values for printed parts in
# the XY plane (E, nu, yield, ultimate in MPa; density in g/cm3).
# Every filament differs: use your manufacturer's datasheet.
# z_factor = interlayer strength / XY strength (tension across layers).
# ----------------------------------------------------------------------
MATERIALS = {
    "PLA":  dict(E=3500.0, nu=0.36, sy=55.0, su=60.0, dens=1.24, z_factor=0.5),
    "PETG": dict(E=2000.0, nu=0.38, sy=48.0, su=52.0, dens=1.27, z_factor=0.6),
    "ABS":  dict(E=2100.0, nu=0.35, sy=40.0, su=43.0, dens=1.04, z_factor=0.5),
    "ASA":  dict(E=2000.0, nu=0.35, sy=42.0, su=45.0, dens=1.07, z_factor=0.5),
    "PA12": dict(E=1600.0, nu=0.40, sy=45.0, su=50.0, dens=1.01, z_factor=0.6),
}

# Last result of the session (read by the generator dialog)
LAST = None


# ======================================================================
# ELEMENT
# ======================================================================

_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                    dtype=np.float64)


def hex8_stiffness(hx, hy, hz, nu):
    """24x24 stiffness of a hx*hy*hz brick with E = 1 (2x2x2 Gauss)."""
    lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = 1.0 / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    inv_j = np.array([2.0 / hx, 2.0 / hy, 2.0 / hz])
    det_j = hx * hy * hz / 8.0
    g = 1.0 / np.sqrt(3.0)
    KE = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                p = np.array([xi, eta, zeta])
                dN = np.empty((8, 3))
                for a in range(8):
                    s = _CORNERS[a]
                    f = 1.0 + s * p
                    dN[a, 0] = s[0] * f[1] * f[2] / 8.0
                    dN[a, 1] = s[1] * f[0] * f[2] / 8.0
                    dN[a, 2] = s[2] * f[0] * f[1] / 8.0
                dN *= inv_j[None, :]
                B = np.zeros((6, 24))
                for a in range(8):
                    dx, dy, dz = dN[a]
                    c = 3 * a
                    B[0, c] = dx
                    B[1, c + 1] = dy
                    B[2, c + 2] = dz
                    B[3, c] = dy; B[3, c + 1] = dx
                    B[4, c + 1] = dz; B[4, c + 2] = dy
                    B[5, c] = dz; B[5, c + 2] = dx
                KE += B.T @ D @ B * det_j
    return KE


def hex8_B_center(hx, hy, hz):
    """6x24 strain-displacement matrix at the element centre.
    Voigt order: xx, yy, zz, gamma_xy, gamma_yz, gamma_xz."""
    dN = _CORNERS / 8.0 * np.array([2.0 / hx, 2.0 / hy, 2.0 / hz])
    B = np.zeros((6, 24))
    for a in range(8):
        dx, dy, dz = dN[a]
        c = 3 * a
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c] = dy; B[3, c + 1] = dx
        B[4, c + 1] = dz; B[4, c + 2] = dy
        B[5, c] = dz; B[5, c + 2] = dx
    return B


def elasticity_unit(nu):
    lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = 1.0 / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


# ======================================================================
# GRID AND MODEL
# ======================================================================

def make_grid(bb_min, bb_max, h_target):
    """Node axes aligned to the bounding box (box faces fall exactly on
    grid nodes, so loads/constraints on planar faces map cleanly)."""
    axes = []
    for a in range(3):
        L = float(bb_max[a] - bb_min[a])
        n = max(2, int(round(L / h_target)))
        axes.append(np.linspace(bb_min[a], bb_max[a], n + 1))
    return axes


def auto_element_size(bb_min, bb_max, cell_size, max_elements=40000):
    """Element size: ~cell/2 (the density varies on the cell scale),
    but at least ~8 elements across the thinnest dimension of the part
    (thin plates/arms need a few elements through the thickness to bend
    correctly), and never more than max_elements over the bounding box."""
    V = 1.0
    dims = []
    for a in range(3):
        d = max(bb_max[a] - bb_min[a], 1e-6)
        dims.append(d)
        V *= d
    h_budget = (V / float(max_elements)) ** (1.0 / 3.0)
    return max(h_budget, min(cell_size / 2.0, min(dims) / 8.0))


def centers(axis):
    return 0.5 * (axis[1:] + axis[:-1])


class Model(object):
    """Voxel FE model of the part.

    active : bool (nex, ney, nez) - element centre inside the part
    phi    : float (nex, ney, nez) - solid shell fraction of the element
    """

    def __init__(self, xn, yn, zn, active, phi, Es, nu, C, n):
        self.xn, self.yn, self.zn = xn, yn, zn
        self.nex, self.ney, self.nez = len(xn) - 1, len(yn) - 1, len(zn) - 1
        self.hx = (xn[-1] - xn[0]) / self.nex
        self.hy = (yn[-1] - yn[0]) / self.ney
        self.hz = (zn[-1] - zn[0]) / self.nez
        self.active = np.asarray(active, dtype=bool)
        if not self.active.any():
            raise ValueError("No active element: is the part a closed solid?")
        self.phi = np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)
        self.Es, self.nu, self.C, self.n = float(Es), float(nu), float(C), float(n)
        self.KE = hex8_stiffness(self.hx, self.hy, self.hz, self.nu)
        self.KEdiag = np.diag(self.KE).copy()

        # --- connectivity of the active elements -----------------------
        ny1, nz1 = self.ney + 1, self.nez + 1
        ie, je, ke = np.nonzero(self.active)
        self.elem_ijk = np.stack([ie, je, ke], axis=1)
        off = ((_CORNERS + 1) // 2).astype(np.int64)        # (8,3)
        ni = ie[:, None] + off[None, :, 0]
        nj = je[:, None] + off[None, :, 1]
        nk = ke[:, None] + off[None, :, 2]
        gnode = (ni * ny1 + nj) * nz1 + nk                   # (nel, 8)
        used = np.unique(gnode)
        self.node_gid = used                                  # compact -> global
        nmap = -np.ones((self.nex + 1) * ny1 * nz1, dtype=np.int64)
        nmap[used] = np.arange(len(used))
        self.node_map = nmap                                  # global -> compact
        cn = nmap[gnode]                                      # (nel, 8)
        self.edof = (3 * cn[:, :, None] + np.arange(3)[None, None, :]
                     ).reshape(-1, 24)
        self.nel = len(ie)
        self.nnode = len(used)
        self.ndof = 3 * self.nnode
        gi = used // (ny1 * nz1)
        gj = (used // nz1) % ny1
        gk = used % nz1
        self.node_xyz = np.stack([xn[gi], yn[gj], zn[gk]], axis=1)
        self.node_ijk = np.stack([gi, gj, gk], axis=1)
        self.phi_e = self.phi[self.active]
        self.vol_e = self.hx * self.hy * self.hz
        self.fixed = np.zeros(self.ndof, dtype=bool)
        self.f = np.zeros(self.ndof)

    # ---------------------------------------------------------- material
    def young(self, rho):
        """Element Young modulus: solid shell fraction + lattice."""
        return self.Es * (self.phi_e +
                          (1.0 - self.phi_e) * self.C * rho ** self.n)

    def dyoung(self, rho):
        return self.Es * (1.0 - self.phi_e) * self.C * self.n * \
            rho ** (self.n - 1.0)

    # ----------------------------------------------------- point mapping
    def nearest_nodes(self, pts):
        """Compact index of the nearest ACTIVE node for each point
        (search in the 4x4x4 node neighbourhood, then global fallback)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        o = np.array([self.xn[0], self.yn[0], self.zn[0]])
        h = np.array([self.hx, self.hy, self.hz])
        nmax = np.array([self.nex, self.ney, self.nez])
        base = np.floor((pts - o) / h).astype(np.int64) - 1
        ny1, nz1 = self.ney + 1, self.nez + 1
        best_d = np.full(len(pts), np.inf)
        best_n = -np.ones(len(pts), dtype=np.int64)
        for di in range(4):
            for dj in range(4):
                for dk in range(4):
                    ijk = base + np.array([di, dj, dk])
                    ok = np.all((ijk >= 0) & (ijk <= nmax), axis=1)
                    ijk = np.clip(ijk, 0, nmax)
                    gid = (ijk[:, 0] * ny1 + ijk[:, 1]) * nz1 + ijk[:, 2]
                    cid = self.node_map[gid]
                    ok &= cid >= 0
                    p = o + ijk * h
                    d = np.sum((p - pts) ** 2, axis=1)
                    d[~ok] = np.inf
                    better = d < best_d
                    best_d[better] = d[better]
                    best_n[better] = cid[better]
        miss = np.nonzero(best_n < 0)[0]
        for s in range(0, len(miss), 256):
            idx = miss[s:s + 256]
            d = ((pts[idx, None, :] - self.node_xyz[None, :, :]) ** 2).sum(-1)
            best_n[idx] = np.argmin(d, axis=1)
            best_d[idx] = d[np.arange(len(idx)), best_n[idx]]
        return best_n, np.sqrt(best_d)

    def add_fixed(self, pts):
        """Clamps (all 3 DOF) the active nodes nearest to the points."""
        nodes, _ = self.nearest_nodes(pts)
        nodes = np.unique(nodes)
        for c in range(3):
            self.fixed[3 * nodes + c] = True
        return len(nodes)

    def add_forces(self, pts, vecs):
        """Adds point forces (N) lumped on the nearest active node."""
        nodes, _ = self.nearest_nodes(pts)
        vecs = np.atleast_2d(np.asarray(vecs, dtype=np.float64))
        for c in range(3):
            np.add.at(self.f, 3 * nodes + c, vecs[:, c])
        return len(np.unique(nodes))

    # ------------------------------------------------------------ solver
    def matvec(self, E, u):
        ue = u[self.edof]                                  # (nel, 24)
        fe = (ue @ self.KE) * E[:, None]
        return np.bincount(self.edof.ravel(), fe.ravel(),
                           minlength=self.ndof)

    def solve(self, E, u0=None, tol=1e-6, maxit=8000):
        """K(E) u = f with clamped DOF: Jacobi-preconditioned CG,
        matrix-free (projected onto the free DOF)."""
        free = ~self.fixed
        diag = np.bincount(self.edof.ravel(),
                           (E[:, None] * self.KEdiag[None, :]).ravel(),
                           minlength=self.ndof)
        Minv = np.where(free & (diag > 0), 1.0 / np.maximum(diag, 1e-300), 0.0)
        f = self.f * free
        u = np.zeros(self.ndof) if u0 is None else u0 * free
        r = (f - self.matvec(E, u)) * free
        fn = np.linalg.norm(f)
        if fn == 0.0:
            return u, 0
        z = Minv * r
        p = z.copy()
        rz = r @ z
        it = 0
        for it in range(1, maxit + 1):
            Ap = self.matvec(E, p) * free
            pAp = p @ Ap
            if pAp <= 0.0:
                break
            a = rz / pAp
            u += a * p
            r -= a * Ap
            if np.linalg.norm(r) < tol * fn:
                break
            z = Minv * r
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
        return u, it

    def element_energy(self, u):
        """u_e^T KE u_e per element (E = 1)."""
        ue = u[self.edof]
        return np.einsum("ij,jk,ik->i", ue, self.KE, ue)

    def check(self):
        """Basic sanity of loads/constraints before solving."""
        if not self.fixed.any():
            raise ValueError("No constrained node: add a Fixed constraint.")
        if not np.any(self.f != 0.0):
            raise ValueError("No load: add a Force or Pressure constraint.")
        nfix = int(self.fixed.sum() // 3)
        if nfix < 3:
            raise ValueError("Only {} constrained nodes: the part is "
                             "not restrained against rotation.".format(nfix))


# ======================================================================
# DENSITY FILTER (cone, FFT convolution on the element grid)
# ======================================================================

class DensityFilter(object):
    def __init__(self, model, radius):
        self.m = model
        h = np.array([model.hx, model.hy, model.hz])
        r = np.maximum(np.ceil(radius / h).astype(int) - 1, 0)
        self.r = r
        ax = [np.arange(-r[a], r[a] + 1) * h[a] for a in range(3)]
        DX, DY, DZ = np.meshgrid(*ax, indexing="ij")
        w = np.maximum(radius - np.sqrt(DX ** 2 + DY ** 2 + DZ ** 2), 0.0)
        self.shape = tuple(np.array(model.active.shape) + 2 * r)
        self.Wk = np.fft.rfftn(w, s=self.shape)
        mask = model.active.astype(np.float64)
        self.wsum = self._conv(mask)[model.active]
        self.wsum = np.maximum(self.wsum, 1e-12)

    def _conv(self, a):
        A = np.fft.rfftn(a, s=self.shape)
        full = np.fft.irfftn(A * self.Wk, s=self.shape)
        r = self.r
        n = self.m.active.shape
        return full[r[0]:r[0] + n[0], r[1]:r[1] + n[1], r[2]:r[2] + n[2]]

    def _scatter(self, v):
        g = np.zeros(self.m.active.shape)
        g[self.m.active] = v
        return g

    def forward(self, x):
        """x (active elements) -> filtered density."""
        return self._conv(self._scatter(x))[self.m.active] / self.wsum

    def backward(self, d):
        """Chain rule: d(.)/d rho  ->  d(.)/d x."""
        return self._conv(self._scatter(d / self.wsum))[self.m.active]


# ======================================================================
# OPTIMIZATION
# ======================================================================

def optimize(model, rho_target, rho_min, rho_max, filter_radius,
             maxit=60, tol_change=0.01, move=0.05, cg_tol=1e-6,
             callback=None, log=None):
    """Minimum compliance with a mean-lattice-density constraint.

    callback(it, compliance, mean_rho, change) -> False to cancel.
    Returns a dict with the density field and the run statistics.
    """
    log = log or (lambda s: None)
    t0 = time.time()
    model.check()
    if not (rho_min <= rho_target <= rho_max):
        raise ValueError("Target density {:.1f}% outside the allowed "
                         "range {:.1f}-{:.1f}%".format(
                             rho_target * 100, rho_min * 100, rho_max * 100))
    filt = DensityFilter(model, filter_radius)
    wv = (1.0 - model.phi_e) * model.vol_e
    wv_sum = max(wv.sum(), 1e-12)

    def mean_rho(rho):
        return float(np.sum(wv * rho) / wv_sum)

    # --- reference: uniform lattice at the same mass ------------------
    rho_u = np.full(model.nel, rho_target)
    u, its = model.solve(model.young(rho_u), tol=cg_tol)
    u_uniform = u.copy()
    c_uniform = float(model.f @ u)
    log("Uniform lattice {:.0f}%: compliance {:.4g} N*mm, max "
        "displacement {:.4g} mm  (CG {} it.)\n".format(
            rho_target * 100, c_uniform, _max_disp(u), its))

    x = rho_u.copy()
    rho = filt.forward(x)
    history = []
    change = 1.0
    cancelled = False
    c = c_uniform
    dv = filt.backward(wv / wv_sum)
    for it in range(1, maxit + 1):
        E = model.young(rho)
        u, its = model.solve(E, u0=u, tol=cg_tol)
        c = float(model.f @ u)
        dc = -model.dyoung(rho) * model.element_energy(u)
        dc = filt.backward(dc)
        dc = np.minimum(dc, -1e-30)
        # --- OC with bisection on the volume constraint ----------------
        l1, l2 = 0.0, 1e9
        lo = np.maximum(rho_min, x - move)
        hi = np.minimum(rho_max, x + move)
        xnew = x
        B = -dc / np.maximum(dv, 1e-30)
        for _ in range(200):
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(x * np.sqrt(B / lmid), lo, hi)
            if mean_rho(filt.forward(xnew)) > rho_target:
                l1 = lmid
            else:
                l2 = lmid
            if (l2 - l1) <= 1e-6 * (l1 + l2):
                break
        change = float(np.max(np.abs(xnew - x)))
        x = xnew
        rho = filt.forward(x)
        mr = mean_rho(rho)
        history.append((it, c, mr, change))
        log("  it {:3d}  compliance {:.4g}  mean density {:.1f}%  "
            "change {:.3f}  (CG {} it.)\n".format(it, c, mr * 100, change, its))
        if callback is not None and callback(it, c, mr, change) is False:
            cancelled = True
            break
        if change < tol_change:
            break
        # OC can oscillate on the bounds with a stalled objective: stop
        # when the compliance has not moved for 3 iterations
        if len(history) >= 4:
            cs = [h_[1] for h_ in history[-4:]]
            if (max(cs) - min(cs)) < 1e-4 * abs(cs[-1]):
                break

    # final consistent solution
    E = model.young(rho)
    u, _ = model.solve(E, u0=u, tol=cg_tol)
    c = float(model.f @ u)

    grid = np.full(model.active.shape, np.nan)
    grid[model.active] = rho
    res = dict(
        xc=centers(model.xn), yc=centers(model.yn), zc=centers(model.zn),
        rho=grid, rho_min=float(rho_min), rho_max=float(rho_max),
        rho_target=float(rho_target), mean_rho=mean_rho(rho),
        compliance=c, compliance_uniform=c_uniform,
        gain=1.0 - c / c_uniform if c_uniform > 0 else 0.0,
        max_disp=_max_disp(u), max_disp_uniform=None,
        iterations=len(history), cancelled=cancelled,
        history=np.array(history, dtype=np.float64).reshape(-1, 4),
        filter_radius=float(filter_radius),
        Es=model.Es, nu=model.nu, C=model.C, n=model.n,
        elapsed=time.time() - t0,
        # not saved: needed by verify()
        u=u, u_uniform=u_uniform, rho_e=rho, rho_uniform=rho_u,
    )
    return res


# ======================================================================
# STRENGTH VERIFICATION
# ======================================================================

def element_stress_unit(model, u):
    """Macro strain-derived stress per element with E = 1 (MPa per MPa
    of modulus): sigma = E_e * this. Shape (nel, 6)."""
    B = hex8_B_center(model.hx, model.hy, model.hz)
    D = elasticity_unit(model.nu)
    eps = u[model.edof] @ B.T                      # (nel, 6)
    return eps @ D.T


def von_mises(s):
    sx, sy, sz, txy, tyz, txz = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                   + 3.0 * (txy ** 2 + tyz ** 2 + txz ** 2))


def boundary_elements(model):
    """Elements touching a loaded or clamped node: stresses there are
    dominated by the point-like application of loads/constraints."""
    f = model.f.reshape(-1, 3)
    fx = model.fixed.reshape(-1, 3)
    special = np.any(f != 0.0, axis=1) | np.any(fx, axis=1)
    nodes = model.edof[:, 0::3] // 3
    return special[nodes].any(axis=1)


def verify(model, rho, u, sy, su, C2, m, z_factor=1.0, build_axis=2,
           dens_g_cm3=None):
    """Strength check of the shell + lattice.

    Lattice (each element, fraction 1 - phi):
        macro stress  s_lat = Es C rho^n * S_unit
        strength      sigma*(rho) = C2 rho^m * sigma_s
    Solid shell (fraction phi):
        stress        s_sol = Es * S_unit,  strength sigma_s
    In-plane (XY) check with von Mises; interlayer check with the
    TENSILE normal stress along the build axis against z_factor*sigma_s.
    Returns per-element safety factors (yield / ultimate) and summary.
    """
    S = element_stress_unit(model, u)
    vm1 = von_mises(S)
    sz1 = np.maximum(S[:, build_axis], 0.0)
    Elat = model.Es * model.C * rho ** model.n
    strength_rel = C2 * rho ** m
    big = 1e9
    lat = model.phi_e < 0.999
    sol = model.phi_e > 0.001

    def fs(sig):
        fs_l = np.full(model.nel, big)
        fs_s = np.full(model.nel, big)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = strength_rel * sig / (Elat * vm1)
            b = strength_rel * sig * z_factor / (Elat * sz1)
            fs_l[lat] = np.minimum(a, b)[lat]
            a = sig / (model.Es * vm1)
            b = sig * z_factor / (model.Es * sz1)
            fs_s[sol] = np.minimum(a, b)[sol]
        fs_l = np.nan_to_num(fs_l, nan=big, posinf=big)
        fs_s = np.nan_to_num(fs_s, nan=big, posinf=big)
        return fs_l, fs_s

    fy_l, fy_s = fs(sy)
    fu_l, fu_s = fs(su)
    fy = np.minimum(fy_l, fy_s)
    fu = np.minimum(fu_l, fu_s)
    bnd = boundary_elements(model)
    inner = ~bnd if (~bnd).any() else np.ones(model.nel, bool)

    # which criterion governs the critical element
    with np.errstate(divide="ignore", invalid="ignore"):
        gov_z = (sz1 * 1.0 / max(z_factor, 1e-9)) > vm1
    i_crit = int(np.argmin(np.where(inner, fy, big)))

    vol_lat = float(np.sum((1.0 - model.phi_e) * rho) * model.vol_e)
    vol_sol = float(np.sum(model.phi_e) * model.vol_e)
    out = dict(
        fs_yield=fy, fs_ult=fu,
        fs_yield_min=float(fy[inner].min()),
        fs_ult_min=float(fu[inner].min()),
        fs_yield_min_all=float(fy.min()),
        fs_yield_lattice_min=float(fy_l[inner].min()),
        fs_yield_shell_min=float(fy_s[inner].min()),
        crit_xyz=None, crit_mode="interlayer (Z)" if gov_z[i_crit]
        else "von Mises (XY)",
        crit_in_shell=bool(fy_s[i_crit] <= fy_l[i_crit]),
        crit_rho=float(rho[i_crit]),
        n_below_1=int(np.sum(fy[inner] < 1.0)),
        vm_lattice_max=float(np.max(np.where(lat, Elat * vm1, 0.0))),
        volume_mm3=vol_lat + vol_sol,
        mass_g=None if dens_g_cm3 is None else
        (vol_lat + vol_sol) / 1000.0 * dens_g_cm3,
    )
    ijk = model.elem_ijk[i_crit]
    out["crit_xyz"] = (float(centers(model.xn)[ijk[0]]),
                       float(centers(model.yn)[ijk[1]]),
                       float(centers(model.zn)[ijk[2]]))
    return out


# ======================================================================
# STRENGTH-DRIVEN SIZING (minimum mass for a target safety factor)
# ======================================================================

def lattice_fs(model, rho, u, sy, C2, m, z_factor=1.0, build_axis=2):
    """Yield safety factor of the LATTICE part of each element (same
    criteria as verify(): von Mises in-plane, tensile stress along the
    build axis against z_factor * strength). 1e9 where unloaded."""
    S = element_stress_unit(model, u)
    vm1 = von_mises(S)
    sz1 = np.maximum(S[:, build_axis], 0.0)
    Elat = model.Es * model.C * rho ** model.n
    strength = C2 * rho ** m * sy
    with np.errstate(divide="ignore", invalid="ignore"):
        a = strength / (Elat * vm1)
        b = strength * z_factor / (Elat * sz1)
    fs = np.minimum(a, b)
    return np.nan_to_num(fs, nan=1e9, posinf=1e9)


def size_for_strength(model, fs_target, sy, C2, m, rho_min, rho_max,
                      filter_radius, z_factor=1.0, build_axis=2,
                      maxit=40, tol_rho=0.01, cg_tol=1e-6, callback=None,
                      log=None):
    """Minimum lattice mass such that the yield safety factor of the
    lattice is >= fs_target (within the printable density bounds).

    Bisection on the mean density: at each step the stiffness-optimal
    density map for that mass is computed with optimize(), then its
    lattice safety factor is checked (same criteria as verify()). The
    lightest map that passes is returned.

    Why not a classic fully-stressed (stress-ratio) update: with a
    solid shell the lattice is largely strain-controlled, and since
    stiffness ~ rho^n and strength ~ rho^m with n close to m, raising
    the local density barely raises the local safety factor; the
    stress-ratio iteration then converges to an infeasible design.

    Also finds, by bisection, the UNIFORM lattice density that meets
    the same fs_target, for a like-for-like mass comparison.
    """
    log = log or (lambda s: None)
    t0 = time.time()
    model.check()
    design = model.phi_e < 0.999
    inner = ~boundary_elements(model)
    probe = inner & design
    if not probe.any():
        probe = design if design.any() else np.ones(model.nel, bool)
    quiet = lambda s: None
    step = [0]
    cancelled = [False]

    def fs_min_of(rho, u):
        return float(lattice_fs(model, rho, u, sy, C2, m, z_factor,
                                build_axis)[probe].min())

    def evaluate(mean):
        step[0] += 1
        r = optimize(model, mean, rho_min, rho_max, filter_radius,
                     maxit=maxit, cg_tol=cg_tol, log=quiet)
        f = fs_min_of(r["rho_e"], r["u"])
        log("  step {:2d}: mean density {:.1f}%  ->  lattice FS min {:.2f}"
            "  ({} it.)\n".format(step[0], mean * 100, f, r["iterations"]))
        if callback is not None and callback(step[0], r["compliance"],
                                             mean, f) is False:
            cancelled[0] = True
        return r, f

    lo = rho_min
    hi = rho_max - 1e-4
    r_hi, f_hi = evaluate(hi)
    best, f_best, feasible = r_hi, f_hi, f_hi >= fs_target
    if feasible and not cancelled[0]:
        r_lo, f_lo = evaluate(lo)
        if f_lo >= fs_target:
            best, f_best = r_lo, f_lo
        else:
            while hi - lo > tol_rho and not cancelled[0]:
                mid = 0.5 * (lo + hi)
                r_mid, f_mid = evaluate(mid)
                if f_mid >= fs_target:
                    hi, best, f_best = mid, r_mid, f_mid
                else:
                    lo = mid

    rho, u = best["rho_e"], best["u"]
    mr = best["mean_rho"]

    # --- uniform lattice meeting the same safety factor ---------------
    def fs_uniform(r):
        ru = np.full(model.nel, r)
        uu, _ = model.solve(model.young(ru), u0=u, tol=cg_tol)
        return fs_min_of(ru, uu), uu
    fu_hi, u_hi = fs_uniform(rho_max)
    if fu_hi < fs_target:
        r_u, u_u, uniform_ok = rho_max, u_hi, False
    else:
        a, b, u_u = rho_min, rho_max, u_hi
        fu_lo, u_lo = fs_uniform(rho_min)
        if fu_lo >= fs_target:
            b, u_u = rho_min, u_lo
        else:
            while b - a > 0.002:
                mid = 0.5 * (a + b)
                f_mid, u_mid = fs_uniform(mid)
                if f_mid >= fs_target:
                    b, u_u = mid, u_mid
                else:
                    a = mid
        r_u, uniform_ok = b, True
    c_u = float(model.f @ u_u)

    res = dict(best)
    res.update(
        mode="strength",
        compliance_uniform=c_u,
        gain=1.0 - best["compliance"] / c_u if c_u > 0 else 0.0,
        iterations=step[0], cancelled=cancelled[0],
        elapsed=time.time() - t0,
        fs_target=float(fs_target), fs_lattice_min=f_best,
        feasible=bool(feasible),
        rho_uniform_equiv=float(r_u), uniform_feasible=uniform_ok,
        lattice_mass_saving=(1.0 - mr / r_u) if (uniform_ok and feasible)
        else None,
        u_uniform=u_u, rho_uniform=np.full(model.nel, r_u),
    )
    return res


# ======================================================================
# STIFFNESS-DRIVEN SIZING (minimum mass for a required stiffness)
# ======================================================================

def load_stiffness(model, u):
    """Stiffness seen by the applied load [N/mm]: resultant force over
    the work-conjugate displacement, k = F_res^2 / (f.u). For a load
    spread on a face along one direction this is F / (mean displacement
    of the face along the load)."""
    F = model.f.reshape(-1, 3).sum(axis=0)
    F_res = float(np.linalg.norm(F))
    work = float(model.f @ u)
    if F_res <= 0.0 or work <= 0.0:
        return 0.0, F_res
    return F_res * F_res / work, F_res


def size_for_stiffness(model, k_target, rho_min, rho_max, filter_radius,
                       maxit=40, tol_rho=0.01, cg_tol=1e-6, callback=None,
                       log=None):
    """Minimum lattice mass such that the stiffness under the applied
    load is >= k_target [N/mm] (bisection on the mean density with the
    stiffness-optimal map at every step). Also returns the uniform
    lattice density giving the same stiffness."""
    log = log or (lambda s: None)
    t0 = time.time()
    model.check()
    quiet = lambda s: None
    step = [0]
    cancelled = [False]

    def evaluate(mean):
        step[0] += 1
        r = optimize(model, mean, rho_min, rho_max, filter_radius,
                     maxit=maxit, cg_tol=cg_tol, log=quiet)
        k, _ = load_stiffness(model, r["u"])
        log("  step {:2d}: mean density {:.1f}%  ->  stiffness {:.4g} N/mm"
            "  ({} it.)\n".format(step[0], mean * 100, k, r["iterations"]))
        if callback is not None and callback(step[0], r["compliance"],
                                             mean, k) is False:
            cancelled[0] = True
        return r, k

    lo = rho_min
    hi = rho_max - 1e-4
    r_hi, k_hi = evaluate(hi)
    best, k_best, feasible = r_hi, k_hi, k_hi >= k_target
    if feasible and not cancelled[0]:
        r_lo, k_lo = evaluate(lo)
        if k_lo >= k_target:
            best, k_best = r_lo, k_lo
        else:
            while hi - lo > tol_rho and not cancelled[0]:
                mid = 0.5 * (lo + hi)
                r_mid, k_mid = evaluate(mid)
                if k_mid >= k_target:
                    hi, best, k_best = mid, r_mid, k_mid
                else:
                    lo = mid
    u = best["u"]
    mr = best["mean_rho"]

    def k_uniform(r):
        ru = np.full(model.nel, r)
        uu, _ = model.solve(model.young(ru), u0=u, tol=cg_tol)
        return load_stiffness(model, uu)[0], uu
    ku_hi, u_hi = k_uniform(rho_max)
    if ku_hi < k_target:
        r_u, u_u, uniform_ok = rho_max, u_hi, False
    else:
        a, b, u_u = rho_min, rho_max, u_hi
        ku_lo, u_lo = k_uniform(rho_min)
        if ku_lo >= k_target:
            b, u_u = rho_min, u_lo
        else:
            while b - a > 0.002:
                mid = 0.5 * (a + b)
                k_mid, u_mid = k_uniform(mid)
                if k_mid >= k_target:
                    b, u_u = mid, u_mid
                else:
                    a = mid
        r_u, uniform_ok = b, True
    c_u = float(model.f @ u_u)
    res = dict(best)
    res.update(
        mode="stiffness",
        compliance_uniform=c_u,
        gain=1.0 - best["compliance"] / c_u if c_u > 0 else 0.0,
        iterations=step[0], cancelled=cancelled[0],
        elapsed=time.time() - t0,
        k_target=float(k_target), k_found=float(k_best),
        feasible=bool(feasible),
        rho_uniform_equiv=float(r_u), uniform_feasible=uniform_ok,
        lattice_mass_saving=(1.0 - mr / r_u) if (uniform_ok and feasible)
        else None,
        u_uniform=u_u, rho_uniform=np.full(model.nel, r_u),
    )
    return res


# ======================================================================
# STAGE 1: OUTER SHAPE (solid/void SIMP) AND SOLID "KEEP" ZONES
# ======================================================================

def _heaviside(xt, beta, eta=0.5):
    tb = np.tanh(beta * eta)
    den = tb + np.tanh(beta * (1.0 - eta))
    xb = (tb + np.tanh(beta * (xt - eta))) / den
    dxb = beta * (1.0 - np.tanh(beta * (xt - eta)) ** 2) / den
    return xb, dxb


def simp_shape(model, volfrac, filter_radius, keep=None, penal=3.0,
               emin=1e-2, maxit=150, move=0.2, tol_change=0.01,
               cg_tol=1e-4, beta_max=16.0, callback=None, log=None):
    """SIMP (solid/void) on the voxel model with density filter and
    Heaviside projection (beta continuation 1 -> beta_max): minimum
    compliance with volume fraction volfrac of the part. Elements in
    `keep` (bool per active element) are solid and outside the design.
    Returns the projected physical density per element (~0/1) and the
    number of iterations.

    Stage 1 of the two-stage method (Giacalone 2023, ch. 4): it finds
    the outer shape of the component and removes the volume that does
    not carry load; the lattice is then graded inside that shape. The
    projection drives the result to crisp solid/void, so that the 0.5
    level used as outer surface does not cut grey load paths. The
    filter radius sets the minimum member size."""
    log = log or (lambda s: None)
    model.check()
    keep = np.zeros(model.nel, bool) if keep is None else np.asarray(keep)
    design = ~keep
    filt = DensityFilter(model, filter_radius)
    Es = model.Es
    beta = 1.0

    def phys(xd):
        xt = filt.forward(xd)
        xb, dxb = _heaviside(xt, beta)
        return np.where(keep, 1.0, xb), np.where(keep, 0.0, dxb)

    x = np.where(keep, 1.0, volfrac)
    xp, dxb = phys(x)
    u = None
    it = 0
    hist = []
    since = 0
    for it in range(1, maxit + 1):
        E = Es * (emin + (1.0 - emin) * xp ** penal)
        u, its = model.solve(E, u0=u, tol=cg_tol)
        c = float(model.f @ u)
        dc = -penal * (1.0 - emin) * Es * xp ** (penal - 1.0) * \
            model.element_energy(u)
        dc = np.minimum(filt.backward(dc * dxb), -1e-30)
        dv = np.maximum(filt.backward(dxb / float(model.nel)), 1e-30)
        l1, l2 = 0.0, 1e15
        lo = np.maximum(0.0, x - move)
        hi = np.minimum(1.0, x + move)
        B = -dc / dv
        xnew = x
        for _ in range(200):
            lmid = 0.5 * (l1 + l2)
            xnew = np.where(keep, 1.0,
                            np.clip(x * np.sqrt(B / lmid), lo, hi))
            if phys(xnew)[0].mean() > volfrac:
                l1 = lmid
            else:
                l2 = lmid
            if (l2 - l1) <= 1e-6 * (l1 + l2):
                break
        change = float(np.max(np.abs(xnew - x)[design])) if design.any() \
            else 0.0
        x = xnew
        xp, dxb = phys(x)
        grey = float(np.mean(4.0 * xp * (1.0 - xp)))
        log("  shape it {:3d}  compliance {:.4g}  volume {:.1f}%  beta "
            "{:.0f}  grey {:.2f}  change {:.3f}  (CG {} it.)\n".format(
                it, c, 100 * xp.mean(), beta, grey, change, its))
        if callback is not None and callback(it, c, float(xp.mean()),
                                             change) is False:
            break
        hist.append(c)
        since += 1
        stalled = len(hist) >= 6 and (max(hist[-6:]) - min(hist[-6:])) < \
            2e-3 * hist[-1]
        if (change < tol_change or stalled or since >= 25) and since >= 5:
            if beta >= beta_max:
                break
            beta = min(2.0 * beta, beta_max)
            xp, dxb = phys(x)
            since = 0
            hist = []
    return xp, it


def shape_size_check(bb_min, bb_max, member):
    """The shape stage needs room for members of size `member`: returns
    the ratio (smallest part dimension / member size)."""
    dmin = min(bb_max[a] - bb_min[a] for a in range(3))
    return dmin / max(member, 1e-9)


def grid_from_elements(model, values, fill=np.nan):
    g = np.full(model.active.shape, fill, dtype=np.float64)
    g[model.active] = values
    return g


def keep_mask_from_points(model, pts, ring):
    """Active elements within ~ring [mm] of the given surface points
    (element containing each point, dilated by round(ring/h) layers)."""
    if pts is None or len(pts) == 0:
        return np.zeros(model.nel, bool)
    o = np.array([model.xn[0], model.yn[0], model.zn[0]])
    h = np.array([model.hx, model.hy, model.hz])
    nmax = np.array(model.active.shape) - 1
    ijk = np.clip(np.floor((np.asarray(pts) - o) / h).astype(int), 0, nmax)
    seed = np.zeros(model.active.shape, bool)
    seed[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    k = int(round(ring / float(h.min())))
    for _ in range(k):
        d = seed.copy()
        for ax in range(3):
            d |= _shift(seed.astype(np.float64), 1, ax) > 0
            d |= _shift(seed.astype(np.float64), -1, ax) > 0
        seed = d
    return (seed & model.active)[model.active]


def distance_to_seeds(seed, spacing, cap):
    """Distance [mm] from each voxel to the nearest seed voxel (exact
    EDT with scipy, else 6-neighbour dilation up to `cap`)."""
    try:
        from scipy import ndimage as ndi
        return ndi.distance_transform_edt(~seed, sampling=spacing).astype(
            np.float32)
    except ImportError:
        K = int(np.ceil(cap / spacing)) + 1
        d = np.full(seed.shape, (K + 1) * spacing, dtype=np.float32)
        d[seed] = 0.0
        reached = seed.copy()
        for k in range(1, K + 1):
            r = reached.copy()
            r[1:] |= reached[:-1]; r[:-1] |= reached[1:]
            r[:, 1:] |= reached[:, :-1]; r[:, :-1] |= reached[:, 1:]
            r[:, :, 1:] |= reached[:, :, :-1]; r[:, :, :-1] |= reached[:, :, 1:]
            new = r & ~reached
            d[new] = k * spacing
            reached = r
            if reached.all():
                break
        return d


def sample_grid_axes(res, key, x, y, z, fill=0.0):
    """Like sample_density_axes, for any element-centre grid stored in
    res[key] (NaN replaced by `fill`, no clipping)."""
    g = np.nan_to_num(np.asarray(res[key], dtype=np.float64), nan=fill)
    Wx = _lin_weights(x, res["xc"])
    Wy = _lin_weights(y, res["yc"])
    Wz = _lin_weights(z, res["zc"])
    a = np.tensordot(Wx, g, axes=(1, 0))
    a = np.tensordot(a, Wy, axes=(1, 1))
    a = np.tensordot(a, Wz, axes=(1, 1))
    return a.astype(np.float32)


def shape_distance(xs, spacing):
    """Approximate signed distance [mm] (>0 inside) of the 0.5 level of
    a smooth density field xs on a regular grid: (xs - 0.5) / |grad xs|
    with the gradient magnitude measured on the 0.5 level."""
    gx, gy, gz = np.gradient(xs.astype(np.float32), spacing)
    gm = np.sqrt(gx * gx + gy * gy + gz * gz)
    band = np.abs(xs - 0.5) < 0.1
    g = float(np.median(gm[band])) if band.any() else 0.0
    if g <= 1e-9:
        g = 1.0 / max(spacing, 1e-6)
    return ((xs - 0.5) / g).astype(np.float32)


def local_thickness_check(sdf_centres, active, cell_size):
    """Fraction of the part where the local half-thickness (distance
    to the surface at the medial axis, approximated by local maxima of
    the SDF) is below half a cell: there a complete TPMS cell does not
    fit (Giacalone 2023: lattice stiffness drops with few cells)."""
    s = np.where(active, sdf_centres, -1e9)
    ridge = active.copy()
    for ax in range(3):
        for sh in (1, -1):
            ridge &= s >= _shift(s, sh, ax)
    if not ridge.any():
        return 0.0, 0.0
    t = 2.0 * s[ridge]
    frac_thin = float(np.mean(t < cell_size))
    return frac_thin, float(np.median(t))


def _max_disp(u):
    if u.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(u.reshape(-1, 3), axis=1)))


# ======================================================================
# RESULT: sampling on the generator grid, save/load, VTK
# ======================================================================

def _filled(rho):
    """Fills the cells outside the part with the mean of their
    neighbours (repeated), so the interpolation near the skin does not
    see NaN/zero."""
    g = np.array(rho, dtype=np.float64)
    nan = np.isnan(g)
    if nan.all():
        return np.zeros_like(g)
    for _ in range(sum(g.shape)):
        if not nan.any():
            break
        acc = np.zeros_like(g)
        cnt = np.zeros_like(g)
        v = np.where(nan, 0.0, g)
        ok = (~nan).astype(np.float64)
        for ax in range(3):
            for sh in (1, -1):
                acc += _shift(v, sh, ax)
                cnt += _shift(ok, sh, ax)
        newly = nan & (cnt > 0)
        g[newly] = acc[newly] / cnt[newly]
        nan = nan & ~newly
    g[nan] = np.nanmean(rho)
    return g


def _shift(a, sh, ax):
    out = np.zeros_like(a)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if sh > 0:
        src[ax] = slice(0, -sh); dst[ax] = slice(sh, None)
    else:
        src[ax] = slice(-sh, None); dst[ax] = slice(0, sh)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _lin_weights(q, c):
    """(len(q), len(c)) linear interpolation matrix, clamped at the ends."""
    q = np.asarray(q, dtype=np.float64)
    W = np.zeros((len(q), len(c)))
    if len(c) == 1:
        W[:, 0] = 1.0
        return W
    t = np.interp(q, c, np.arange(len(c), dtype=np.float64))
    i0 = np.clip(np.floor(t).astype(int), 0, len(c) - 2)
    a = t - i0
    W[np.arange(len(q)), i0] = 1.0 - a
    W[np.arange(len(q)), i0 + 1] += a
    return W


def sample_density_axes(res, x, y, z):
    """Density on the generator grid (meshgrid 'ij' of x, y, z), by
    separable trilinear interpolation. Returns float32 (nx, ny, nz)."""
    g = res.get("_filled")
    if g is None:
        g = _filled(res["rho"])
        res["_filled"] = g
    Wx = _lin_weights(x, res["xc"])
    Wy = _lin_weights(y, res["yc"])
    Wz = _lin_weights(z, res["zc"])
    a = np.tensordot(Wx, g, axes=(1, 0))              # (nx, nyc, nzc)
    a = np.tensordot(a, Wy, axes=(1, 1))              # (nx, nzc, ny)
    a = np.tensordot(a, Wz, axes=(1, 1))              # (nx, ny, nz)
    a = np.clip(a, res["rho_min"], res["rho_max"])
    return a.astype(np.float32)


_SAVE_KEYS = ("xc", "yc", "zc", "rho", "fs_grid", "fs_yield_min",
              "shape", "keep_P", "keep_T", "keep_ring", "mass_pred_g",
              "volume_pred_mm3", "dens",
              "fs_ult_min", "mass_g", "material", "rho_min", "rho_max", "rho_target",
              "mean_rho", "compliance", "compliance_uniform", "gain",
              "max_disp", "iterations", "history", "filter_radius",
              "Es", "nu", "C", "n", "elapsed")


def save_npz(res, path):
    data = {k: np.asarray(res[k]) for k in _SAVE_KEYS
            if res.get(k) is not None and k != "material"}
    if res.get("material"):
        data["material"] = np.asarray(str(res["material"]))
    for k in ("tpms", "target", "cell_size", "t_shell", "nozzle",
              "line_width", "min_lines", "min_pore"):
        if res.get(k) is not None:
            data[k] = np.asarray(res[k])
    np.savez_compressed(path, **data)


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    res = {}
    for k in d.files:
        v = d[k]
        res[k] = v.item() if v.ndim == 0 else v
    for k in ("tpms", "target", "material"):
        if k in res:
            res[k] = str(res[k])
    return res


def write_vtk(res, path):
    """Legacy VTK STRUCTURED_POINTS (ParaView, FreeCAD FEM post):
    'density' (0 outside the part) and 'inside' on the element centres."""
    xc, yc, zc = res["xc"], res["yc"], res["zc"]
    rho = res["rho"]
    nx, ny, nz = rho.shape
    sp = [(c[-1] - c[0]) / max(len(c) - 1, 1) for c in (xc, yc, zc)]
    dens = np.nan_to_num(rho, nan=0.0)
    inside = (~np.isnan(rho)).astype(np.int32)
    fsy = res.get("fs_grid")
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("LatticeFree topology optimization density\n")
        fh.write("ASCII\nDATASET STRUCTURED_POINTS\n")
        fh.write("DIMENSIONS {} {} {}\n".format(nx, ny, nz))
        fh.write("ORIGIN {:.6g} {:.6g} {:.6g}\n".format(xc[0], yc[0], zc[0]))
        fh.write("SPACING {:.6g} {:.6g} {:.6g}\n".format(*sp))
        fh.write("POINT_DATA {}\n".format(nx * ny * nz))
        fh.write("SCALARS density float 1\nLOOKUP_TABLE default\n")
        # VTK order: x fastest
        np.savetxt(fh, dens.transpose(2, 1, 0).ravel(), fmt="%.4f")
        fh.write("SCALARS inside int 1\nLOOKUP_TABLE default\n")
        np.savetxt(fh, inside.transpose(2, 1, 0).ravel(), fmt="%d")
        if fsy is not None:
            fh.write("SCALARS safety_factor_yield float 1\n"
                     "LOOKUP_TABLE default\n")
            np.savetxt(fh, np.nan_to_num(np.minimum(fsy, 99.0), nan=0.0)
                       .transpose(2, 1, 0).ravel(), fmt="%.3f")


# ======================================================================
# GEOMETRY HELPERS (pure numpy, used by the GUI)
# ======================================================================

def sample_triangles(P, T, step):
    """Uniform samples on a triangulated surface: returns
    (points, area weights, triangle index). Each triangle is split
    into m^2 sub-triangles with edge <= step; one sample per centroid."""
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64).reshape(-1, 3)
    pts, wts, tid = [], [], []
    for t, (i, j, k) in enumerate(T):
        a, b, c = P[i], P[j], P[k]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if area <= 0.0:
            continue
        L = max(np.linalg.norm(b - a), np.linalg.norm(c - b),
                np.linalg.norm(a - c))
        m = max(1, int(np.ceil(L / step)))
        bc = []
        for p in range(m):
            for q in range(m - p):
                bc.append(((p + 1.0 / 3.0) / m, (q + 1.0 / 3.0) / m))
                if p + q < m - 1:
                    bc.append(((p + 2.0 / 3.0) / m, (q + 2.0 / 3.0) / m))
        bc = np.array(bc)
        pts.append(a + bc[:, :1] * (b - a) + bc[:, 1:] * (c - a))
        wts.append(np.full(len(bc), area / len(bc)))
        tid.append(np.full(len(bc), t))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)
    return np.vstack(pts), np.concatenate(wts), np.concatenate(tid)


def shell_fraction(sdf_centres, t_shell, h):
    """Fraction of each element occupied by the solid shell, from the
    signed distance of its centre (sdf > 0 inside)."""
    if t_shell <= 0.0:
        return np.zeros_like(sdf_centres, dtype=np.float64)
    return np.clip((t_shell - (sdf_centres - 0.5 * h)) / h, 0.0, 1.0)


def density_bounds(tpms, cell_size, nozzle_mm, rho_max=None):
    """Allowed density range: printable minimum .. open pores."""
    from . import engine
    rmin = max(engine.RHO_MIN,
               engine.min_printable_density(cell_size, nozzle_mm, tpms))
    rmax = engine.RHO_MAX if rho_max is None else rho_max
    return rmin, rmax


def last_path():
    """Copy of the last result, so it survives a FreeCAD restart."""
    try:
        import FreeCAD
        base = FreeCAD.getUserAppDataDir()
    except Exception:
        base = os.path.expanduser("~")
    return os.path.join(base, "LatticeFree_last_topopt.npz")


def get_result():
    """Last result of the session, or the one saved on disk."""
    global LAST
    if LAST is None:
        p = last_path()
        if os.path.isfile(p):
            try:
                LAST = load_npz(p)
            except Exception:
                LAST = None
    return LAST


def set_result(res, save=True):
    global LAST
    LAST = res
    if save:
        try:
            save_npz(res, last_path())
        except Exception:
            pass


# ======================================================================
# PRINTABILITY (FEASIBILITY) LIMITS
# ======================================================================
# Pore size of each TPMS: diameter of the largest sphere inscribed in
# the void, relative to the cell size, MEASURED (exact EDT, 72^3 per
# cell, periodic) at the isovalues of engine.TPMS[...]["cal_t"].
PORE_TABLE = {
    "gyroid": [0.4401, 0.4321, 0.4213, 0.4111, 0.4025, 0.3909, 0.3819,
               0.3716, 0.3579, 0.3525, 0.3391, 0.3239, 0.3155, 0.3056,
               0.2913, 0.2792, 0.2693, 0.2576, 0.2406, 0.2257, 0.2170,
               0.2079, 0.1925, 0.1779, 0.1620, 0.1443, 0.1211, 0.0962,
               0.0680, 0.0],
    "schwarz_p": [0.8343, 0.8179, 0.7940, 0.7698, 0.7541, 0.7381, 0.7217,
                  0.6989, 0.6747, 0.6579, 0.6419, 0.6224, 0.5964, 0.5774,
                  0.5583, 0.5336, 0.5137, 0.4922, 0.4656, 0.4453, 0.4176,
                  0.3909, 0.3622, 0.3333, 0.3056, 0.2693, 0.2257, 0.1822,
                  0.1211, 0.0],
    "diamond": [0.4176, 0.4176, 0.4016, 0.4016, 0.3849, 0.3849, 0.3696,
                0.3643, 0.3536, 0.3368, 0.3368, 0.3216, 0.3179, 0.3056,
                0.2887, 0.2846, 0.2736, 0.2576, 0.2500, 0.2390, 0.2257,
                0.2097, 0.1964, 0.1822, 0.1667, 0.1496, 0.1273, 0.1002,
                0.0680, 0.0],
}

# Defaults: FDM with a 0.4 mm nozzle
DEFAULT_LINE_WIDTH = 0.45      # extrusion width [mm]
DEFAULT_MIN_LINES = 2          # minimum extrusion lines per wall
DEFAULT_MIN_PORE = 1.5         # minimum pore diameter [mm]
MAX_GENERATOR_POINTS = 25e6    # generation grid budget


def pore_mm(rho, cell_size, tpms):
    from . import engine
    d = engine.TPMS[tpms]
    t = np.interp(rho, d["cal_rho"], d["cal_t"])
    return float(np.interp(t, d["cal_t"], PORE_TABLE[tpms])) * cell_size


def cell_for_min_density(rho, w_min, tpms):
    """Cell size [mm] at which a wall of w_min corresponds to rho."""
    from . import engine
    t = engine.density_to_isovalue(rho, tpms)
    cf = engine.TPMS[tpms]["c_forma"]
    return w_min * 2.0 * np.pi * cf / (2.0 * max(t, 1e-9))


def feasibility(tpms, cell_size, line_width=DEFAULT_LINE_WIDTH,
                min_lines=DEFAULT_MIN_LINES, min_pore=DEFAULT_MIN_PORE,
                rho_cap=None, bb=None):
    """Printable density range of the lattice for this cell:

    - rho_min: the thinnest wall is min_lines extrusion lines wide
      (w_min = min_lines * line_width);
    - rho_max: the pores stay >= min_pore (and <= rho_cap, default
      engine.RHO_MAX = open pores / valid calibration);
    - generator grid needed to resolve w_min (element ~ w_min/2.5)
      and its point count over the bounding box bb (if given).
    """
    from . import engine
    d = engine.TPMS[tpms]
    cap = engine.RHO_MAX if rho_cap is None else rho_cap
    w_min = line_width * min_lines
    t_min = engine.isovalue_from_thickness(w_min, cell_size, tpms)
    rho_min = max(engine.RHO_MIN, engine.isovalue_to_density(t_min, tpms))
    pore = np.asarray(PORE_TABLE[tpms]) * cell_size
    # pore decreases with t: largest t with pore >= min_pore
    if pore[0] < min_pore:
        rho_pore = 0.0
    else:
        t_p = float(np.interp(-min_pore, -pore, d["cal_t"]))
        rho_pore = engine.isovalue_to_density(t_p, tpms)
    rho_max = min(cap, rho_pore)
    out = dict(w_min=w_min, rho_min=rho_min, rho_max=rho_max,
               rho_pore=rho_pore, pore_at_max=pore_mm(rho_max, cell_size, tpms)
               if rho_max > 0 else 0.0,
               ok=rho_max - rho_min >= 0.03)
    out["spacing"] = engine.recommended_spacing(cell_size, rho_min, tpms,
                                                min_el_per_wall=2.5)
    if bb is not None:
        out["points"] = engine.estimate_points(bb[0], bb[1], out["spacing"])
    msgs = []
    if not out["ok"]:
        msgs.append("cell {:.1f} mm too small for walls of {:.2f} mm: "
                    "use a cell >= {:.1f} mm".format(
                        cell_size, w_min,
                        cell_for_min_density(max(engine.RHO_MIN,
                                                 rho_max - 0.15), w_min, tpms)))
    out["msg"] = "; ".join(msgs)
    return out


def default_paths(doc_file=None, tag="topopt"):
    """Where to save the result: next to the document if saved,
    else in the home folder."""
    if doc_file:
        base = os.path.splitext(doc_file)[0]
    else:
        base = os.path.join(os.path.expanduser("~"), "LatticeFree")
    return base + "_" + tag + ".npz", base + "_" + tag + ".vtk"
