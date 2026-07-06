# -*- coding: utf-8 -*-
"""
FreeLattice — computation engine
=================================
Implicit pipeline for TPMS infill (Gyroid, Schwarz P, Diamond):
SDF fields, two-sheet TPMS field, shell, fillet, density grading,
marching tetrahedra, mesh cleanup. Kept separate from the GUI
(NFR6): no Qt dependency.
Requires only numpy; uses scipy when available.
"""

import os
import numpy as np

try:
    import FreeCAD as App
    import Part
    import Mesh
    _HAS_FREECAD = True
except ImportError:
    _HAS_FREECAD = False

try:
    from scipy import ndimage as _ndi
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ======================================================================
# MARCHING TETRAHEDRA
# ======================================================================

_CUBE_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
         (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]

_CASES = {
    1:  [[(0, 1), (0, 2), (0, 3)]],
    2:  [[(1, 0), (1, 2), (1, 3)]],
    4:  [[(2, 0), (2, 1), (2, 3)]],
    8:  [[(3, 0), (3, 1), (3, 2)]],
    3:  [[(0, 2), (0, 3), (1, 3)], [(0, 2), (1, 3), (1, 2)]],
    5:  [[(0, 1), (0, 3), (2, 3)], [(0, 1), (2, 3), (2, 1)]],
    9:  [[(0, 1), (0, 2), (3, 2)], [(0, 1), (3, 2), (3, 1)]],
    6:  [[(1, 0), (1, 3), (2, 3)], [(1, 0), (2, 3), (2, 0)]],
    10: [[(1, 0), (1, 2), (3, 2)], [(1, 0), (3, 2), (3, 0)]],
    12: [[(2, 0), (2, 1), (3, 1)], [(2, 0), (3, 1), (3, 0)]],
}
for c in (1, 2, 4, 8, 3, 5, 9, 6, 10, 12):
    _CASES[15 - c] = _CASES[c]


def _interp(pa, pb, va, vb):
    t = va / (va - vb)
    t = np.clip(t, 0.0, 1.0)
    return pa + t[:, None] * (pb - pa)


def marching_tetrahedra(x, y, z, vol):
    nx, ny, nz = vol.shape
    nynz = ny * nz
    ii, jj, kk = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1),
                             np.arange(nz - 1), indexing="ij")
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()

    vals8 = np.empty((ii.size, 8), dtype=np.float32)
    pts8 = np.empty((ii.size, 8, 3), dtype=np.float32)
    # GLOBAL grid-node index for each cube corner: needed to always
    # interpolate every edge in the same direction (see below).
    gid8 = np.empty((ii.size, 8), dtype=np.int64)
    for c, (dx, dy, dz) in enumerate(_CUBE_CORNERS):
        vals8[:, c] = vol[ii + dx, jj + dy, kk + dz]
        pts8[:, c, 0] = x[ii + dx]
        pts8[:, c, 1] = y[jj + dy]
        pts8[:, c, 2] = z[kk + dz]
        gid8[:, c] = (ii + dx) * nynz + (jj + dy) * nz + (kk + dz)

    tris = []
    for tet in _TETS:
        vals = vals8[:, tet]
        pts = pts8[:, tet, :]
        gids = gid8[:, tet]
        inside = vals > 0.0
        code = (inside[:, 0].astype(np.int8) +
                (inside[:, 1] << 1) + (inside[:, 2] << 2) +
                (inside[:, 3] << 3))
        for c, triangles in _CASES.items():
            m = np.flatnonzero(code == c)
            if m.size == 0:
                continue
            v, p, g = vals[m], pts[m], gids[m]
            # "Outward" direction for this configuration: from the
            # barycenter of the INSIDE corners (field > 0) to the OUTSIDE
            # ones. Each triangle normal must agree with this direction,
            # otherwise two vertices are swapped. This keeps the orientation
            # consistent and correct regardless of the _CASES table.
            in_idx = [q for q in range(4) if (c >> q) & 1]
            out_idx = [q for q in range(4) if not (c >> q) & 1]
            outward = p[:, out_idx, :].mean(axis=1) - p[:, in_idx, :].mean(axis=1)
            for tri in triangles:
                corners = []
                for (a, b) in tri:
                    # ALWAYS interpolate from the grid node with the lower
                    # global id to the higher one: the same edge, shared by
                    # adjacent tetrahedra, yields a bit-identical point in
                    # both -> welding always merges it (no cracks).
                    swap = g[:, a] > g[:, b]
                    pa = np.where(swap[:, None], p[:, b], p[:, a])
                    pb = np.where(swap[:, None], p[:, a], p[:, b])
                    va = np.where(swap, v[:, b], v[:, a])
                    vb = np.where(swap, v[:, a], v[:, b])
                    corners.append(_interp(pa, pb, va, vb))
                tri_pts = np.stack(corners, axis=1)
                n = np.cross(tri_pts[:, 1] - tri_pts[:, 0],
                             tri_pts[:, 2] - tri_pts[:, 0])
                flip = np.einsum('ij,ij->i', n, outward) < 0.0
                tri_pts[flip] = tri_pts[flip][:, [0, 2, 1], :]
                tris.append(tri_pts)
    if not tris:
        raise RuntimeError("Nessuna superficie trovata: controlla i parameters.")
    return np.concatenate(tris, axis=0)


# ======================================================================
# CAMPI IMPLICITI
# ======================================================================

# ----------------------------------------------------------------------
# Nodal functions of the supported TPMS. Arguments already scaled (u = k*x).
# ----------------------------------------------------------------------

def _g_gyroid(U, V, W):
    return (np.sin(U) * np.cos(V) +
            np.sin(V) * np.cos(W) +
            np.sin(W) * np.cos(U))


def _g_schwarz_p(U, V, W):
    return np.cos(U) + np.cos(V) + np.cos(W)


def _g_diamond(U, V, W):
    return (np.sin(U) * np.sin(V) * np.sin(W) +
            np.sin(U) * np.cos(V) * np.cos(W) +
            np.cos(U) * np.sin(V) * np.cos(W) +
            np.cos(U) * np.cos(V) * np.sin(W))


# ----------------------------------------------------------------------
# TPMS registry.
# For each tpms_type: nodal function g, density <-> isovalue calibration
# tables (volume fraction of {|g| < t}, numerically MEASURED on a 220^3
# cell-centered grid, dimensionless: valid for any cell size) and
# C_FORMA = surface average of |grad g| on g=0 (same method for all
# types; on the gyroid it reproduces the V0 table within 0.2%).
# Adding a TPMS = adding one entry here (+ one command in
# commands.py): the rest of the pipeline is fully generic.
# ----------------------------------------------------------------------
TPMS = {
    "gyroid": dict(
        label="Gyroid",
        g=_g_gyroid,
        c_forma=1.534,
        cal_t=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
               0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
               1.40, 1.45, 1.50],
        cal_rho=[0.0324, 0.0647, 0.0963, 0.1286, 0.1612, 0.1938, 0.2256,
                 0.2582, 0.2910, 0.3235, 0.3565, 0.3895, 0.4225, 0.4559,
                 0.4890, 0.5228, 0.5564, 0.5906, 0.6247, 0.6595, 0.6944,
                 0.7299, 0.7658, 0.8021, 0.8390, 0.8768, 0.9154, 0.9552,
                 0.9868, 1.0000],
    ),
    "schwarz_p": dict(
        label="Schwarz P",
        g=_g_schwarz_p,
        c_forma=1.337,
        cal_t=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
               1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80,
               1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70,
               2.80, 2.90, 3.00],
        cal_rho=[0.0575, 0.1141, 0.1716, 0.2282, 0.2855, 0.3430, 0.4004,
                 0.4576, 0.5153, 0.5736, 0.6222, 0.6641, 0.7012, 0.7352,
                 0.7660, 0.7943, 0.8203, 0.8442, 0.8662, 0.8863, 0.9049,
                 0.9219, 0.9372, 0.9511, 0.9634, 0.9743, 0.9835, 0.9912,
                 0.9969, 1.0000],
    ),
    "diamond": dict(
        label="Diamond",
        g=_g_diamond,
        c_forma=1.497,
        cal_t=[0.0471, 0.0943, 0.1414, 0.1886, 0.2357, 0.2828, 0.3300,
               0.3771, 0.4243, 0.4714, 0.5185, 0.5657, 0.6128, 0.6600,
               0.7071, 0.7542, 0.8014, 0.8485, 0.8957, 0.9428, 0.9899,
               1.0371, 1.0842, 1.1314, 1.1785, 1.2257, 1.2728, 1.3199,
               1.3671, 1.4142],
        cal_rho=[0.0389, 0.0772, 0.1158, 0.1547, 0.1928, 0.2318, 0.2710,
                 0.3100, 0.3485, 0.3873, 0.4271, 0.4666, 0.5058, 0.5454,
                 0.5854, 0.6253, 0.6661, 0.7064, 0.7471, 0.7884, 0.8303,
                 0.8664, 0.8951, 0.9197, 0.9409, 0.9588, 0.9740, 0.9862,
                 0.9953, 1.0000],
    ),
}

TPMS_DEFAULT = "gyroid"

# recommended relative-density range (printable wall, open pores)
RHO_MIN = 0.10
RHO_MAX = 0.70


def tpms_field(X, Y, Z, period, thickness, tpms_type=TPMS_DEFAULT):
    """TWO-SHEET implicit TPMS field (positive inside the wall).

    The slab {|g| < t} is encoded as the PRODUCT of the two sheets:

        f = (t - g)(t + g) / (2 t k) = (t^2 - g^2) / (2 t k)

    Exactly the same solid as (t - |g|)/k, but f is SMOOTH everywhere
    (polynomial in g): no absolute-value kink on the mid-surface g=0.
    The two surfaces g=+t and g=-t are thus extracted by the marching
    as separate, regular sheets — the pinching/non-manifold artifacts
    the kink produced on thin low-density walls are gone. Near the
    surface f has the same gradient (|grad g|/k) as the old slab
    field, so calibrations, wall thicknesses and the smooth-max fillet
    are unchanged. `thickness` may be a scalar or a field (grading).
    """
    k = 2.0 * np.pi / period
    g = TPMS[tpms_type]["g"](k * X, k * Y, k * Z)
    t = np.maximum(thickness, 1.0e-6)
    return (t * t - g * g) / (2.0 * t * k)


def gyroid_mm(X, Y, Z, period, thickness):
    """V0 compatibility: gyroid field (now two-sheet)."""
    return tpms_field(X, Y, Z, period, thickness, "gyroid")


def density_to_isovalue(rho, tpms_type=TPMS_DEFAULT):
    """Relative density (0-1) -> TPMS thickness isovalue."""
    d = TPMS[tpms_type]
    return float(np.interp(rho, d["cal_rho"], d["cal_t"]))


def isovalue_to_density(t, tpms_type=TPMS_DEFAULT):
    """Isovalue -> estimated relative density (0-1)."""
    d = TPMS[tpms_type]
    return float(np.interp(t, d["cal_t"], d["cal_rho"]))


# Isovalue <-> PHYSICAL wall thickness (mm) relation.
# The wall is the band |g| < t; thickness w = 2t/|grad g|, and
# |grad g| = (2pi/L) * C_FORMA(tpms_type), average measured on g=0.
C_FORMA = TPMS["gyroid"]["c_forma"]   # V0 compatibility


def wall_thickness_mm(t, cell_size, tpms_type=TPMS_DEFAULT):
    """Isovalue -> physical wall thickness in mm, for a given cell."""
    cf = TPMS[tpms_type]["c_forma"]
    return 2.0 * t * cell_size / (2.0 * np.pi * cf)


def isovalue_from_thickness(w_mm, cell_size, tpms_type=TPMS_DEFAULT):
    """Physical thickness (mm) -> isovalue, for a given cell."""
    cf = TPMS[tpms_type]["c_forma"]
    return w_mm * 2.0 * np.pi * cf / (2.0 * cell_size)


def min_printable_density(cell_size, nozzle_mm, tpms_type=TPMS_DEFAULT):
    """Minimum relative density for wall >= nozzle, given the cell."""
    t_min = isovalue_from_thickness(nozzle_mm, cell_size, tpms_type)
    return isovalue_to_density(t_min, tpms_type)


def recommended_spacing(cell_size, rho_min_used, tpms_type=TPMS_DEFAULT,
                        min_el_per_wall=3.0):
    """Recommended grid element: the finer of cell/15 and
    min_wall/3. This is the cure for the 'hexagonal holes' at low
    densities: with the cell/15 auto a 10% wall is ~half an element
    (under-sampled by construction); here the resolution tracks the
    thinnest wall actually present (grading included)."""
    t_min = density_to_isovalue(rho_min_used, tpms_type)
    w_min = wall_thickness_mm(t_min, cell_size, tpms_type)
    return max(0.05, min(cell_size / 15.0, w_min / min_el_per_wall))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def occupancy_from_tris(P, F, x, y, z, spacing, log=None):
    """Occupancy by crossing parity along Z."""
    nx, ny, nz = len(x), len(y), len(z)
    C = np.zeros((nx, ny, nz + 1), dtype=np.int16)
    gx = x.astype(np.float64) + spacing * 1.3e-4
    gy = y.astype(np.float64) + spacing * 0.7e-4

    for f in F:
        a, b, c3 = P[f[0]], P[f[1]], P[f[2]]
        v0 = b[:2] - a[:2]
        v1 = c3[:2] - a[:2]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            continue
        xmin = min(a[0], b[0], c3[0]); xmax = max(a[0], b[0], c3[0])
        ymin = min(a[1], b[1], c3[1]); ymax = max(a[1], b[1], c3[1])
        i0 = np.searchsorted(gx, xmin); i1 = np.searchsorted(gx, xmax)
        j0 = np.searchsorted(gy, ymin); j1 = np.searchsorted(gy, ymax)
        if i0 >= i1 or j0 >= j1:
            continue
        GX, GY = np.meshgrid(gx[i0:i1], gy[j0:j1], indexing="ij")
        d0 = GX - a[0]
        d1 = GY - a[1]
        u = (d0 * v1[1] - v1[0] * d1) / den
        v = (v0[0] * d1 - d0 * v0[1]) / den
        dentro = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        if not dentro.any():
            continue
        zc = a[2] + u * (b[2] - a[2]) + v * (c3[2] - a[2])
        iz = np.searchsorted(z, zc[dentro])
        IX, IY = np.nonzero(dentro)
        np.add.at(C, (IX + i0, IY + j0, np.clip(iz, 0, nz)), 1)

    occ = (np.cumsum(C, axis=2, dtype=np.int32)[:, :, :nz] % 2) == 1
    return occ


def sdf_coarse(occ, spacing, cap_mm):
    """Signed distance (mm), coarse (~half-voxel accuracy)."""
    if HAVE_SCIPY:
        d_in = _ndi.distance_transform_edt(occ, sampling=spacing)
        d_out = _ndi.distance_transform_edt(~occ, sampling=spacing)
        return (d_in - d_out).astype(np.float32)
    K = int(np.ceil(cap_mm / spacing)) + 2
    reached = ~occ
    d = np.zeros(occ.shape, dtype=np.float32)
    remaining = occ.copy()

    def dilate(m):
        r = m.copy()
        r[1:, :, :] |= m[:-1, :, :]; r[:-1, :, :] |= m[1:, :, :]
        r[:, 1:, :] |= m[:, :-1, :]; r[:, :-1, :] |= m[:, 1:, :]
        r[:, :, 1:] |= m[:, :, :-1]; r[:, :, :-1] |= m[:, :, 1:]
        return r

    for k in range(1, K + 1):
        reached = dilate(reached)
        new = reached & remaining
        d[new] = k * spacing
        remaining &= ~new
        if not remaining.any():
            break
    d[remaining] = (K + 1) * spacing
    return np.where(occ, d - 0.5 * spacing,
                    -0.5 * spacing).astype(np.float32)


def _point_triangle_dist(Q, a, b, c):
    """Exact point->triangle distance (Ericson), vectorized over points."""
    ab = b - a; ac = c - a; bc = c - b
    ap = Q - a; bp = Q - b; cp = Q - c
    d1 = ap @ ab; d2 = ap @ ac
    d3 = bp @ ab; d4 = bp @ ac
    d5 = cp @ ab; d6 = cp @ ac
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    closest = np.empty_like(Q)
    done = np.zeros(len(Q), dtype=bool)

    m = (d1 <= 0) & (d2 <= 0)                      # vertice A
    closest[m] = a; done |= m
    m = (~done) & (d3 >= 0) & (d4 <= d3)           # vertice B
    closest[m] = b; done |= m
    m = (~done) & (d6 >= 0) & (d5 <= d6)           # vertice C
    closest[m] = c; done |= m

    m = (~done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)        # lato AB
    t = d1[m] / (d1[m] - d3[m])
    closest[m] = a + t[:, None] * ab; done |= m

    m = (~done) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)        # lato AC
    t = d2[m] / (d2[m] - d6[m])
    closest[m] = a + t[:, None] * ac; done |= m

    m = (~done) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)  # lato BC
    t = (d4[m] - d3[m]) / ((d4[m] - d3[m]) + (d5[m] - d6[m]))
    closest[m] = b + t[:, None] * bc; done |= m

    m = ~done                                       # interno
    if m.any():
        den = 1.0 / (va[m] + vb[m] + vc[m])
        v = vb[m] * den
        w = vc[m] * den
        closest[m] = a + v[:, None] * ab + w[:, None] * ac

    diff = Q - closest
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def refine_band(P, F, x, y, z, sdf, band, log=None):
    """Replaces |sdf| with the exact distance in voxels where
    |sdf| <= band. The sign stays the occupancy one."""
    in_band = np.abs(sdf) <= band
    n_band = int(in_band.sum())
    if log:
        log("Exact band: {} voxels (radius {:.2f} mm)\n"
            .format(n_band, band))
    d_exact = np.full(sdf.shape, np.float32(np.inf))

    nx, ny, nz = sdf.shape
    passo = max(1, len(F) // 10)
    for nf, f in enumerate(F):
        if log and nf % passo == 0:
            log("  exact band: {}/{} triangles\n".format(nf, len(F)))
        a = P[f[0]].astype(np.float64)
        b = P[f[1]].astype(np.float64)
        c3 = P[f[2]].astype(np.float64)
        lo = np.minimum(np.minimum(a, b), c3) - band
        hi = np.maximum(np.maximum(a, b), c3) + band
        i0 = np.searchsorted(x, lo[0]); i1 = np.searchsorted(x, hi[0])
        j0 = np.searchsorted(y, lo[1]); j1 = np.searchsorted(y, hi[1])
        k0 = np.searchsorted(z, lo[2]); k1 = np.searchsorted(z, hi[2])
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        sub_mask = in_band[i0:i1, j0:j1, k0:k1]
        if not sub_mask.any():
            continue
        li, lj, lk = np.nonzero(sub_mask)
        Q = np.stack([x[i0 + li], y[j0 + lj], z[k0 + lk]],
                     axis=1).astype(np.float64)
        d = _point_triangle_dist(Q, a, b, c3)
        sub_d = d_exact[i0:i1, j0:j1, k0:k1]
        cur = sub_d[li, lj, lk]
        upd = d < cur
        if upd.any():
            sub_d[li[upd], lj[upd], lk[upd]] = d[upd]

    valid = in_band & np.isfinite(d_exact)
    sdf[valid] = np.sign(sdf[valid]) * d_exact[valid].astype(np.float32)
    # band voxels never touched (should not exist) stay coarse
    return sdf


def sdf_box_demo(X, Y, Z, lato):
    h = lato / 2.0
    return np.minimum.reduce([h - np.abs(X), h - np.abs(Y),
                              h - np.abs(Z)]).astype(np.float32)


def smooth_max(a, b, k):
    """Soft union (R-function): ~k-radius fillet between the fields."""
    if k <= 0.0:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + 0.25 * k * h * h


def final_field(sdf, gy, t_shell, r_blend=0.0):
    if t_shell > 0.0:
        skin = np.minimum(sdf, t_shell - sdf)
        core = sdf - t_shell
        infill = np.minimum(core, gy)
        # soft union: the shell-lattice junction is born filleted
        return smooth_max(skin, infill, r_blend)
    return np.minimum(sdf, gy)


# ======================================================================
# POST-PROCESSING MESH
# ======================================================================

def weld_vertices(tris, eps):
    """Triangle soup -> (V, T) with unified vertices."""
    Pf = tris.reshape(-1, 3).astype(np.float64)
    Pq = np.round(Pf / eps).astype(np.int64)
    uniq, inv = np.unique(Pq, axis=0, return_inverse=True)
    nv = len(uniq)
    V = np.zeros((nv, 3), dtype=np.float64)
    cnt = np.zeros(nv, dtype=np.float64)
    np.add.at(V, inv, Pf)
    np.add.at(cnt, inv, 1.0)
    V /= cnt[:, None]
    T = inv.reshape(-1, 3)
    ok = ((T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) &
          (T[:, 2] != T[:, 0]))
    return V, T[ok]


def taubin(V, T, iterazioni=8, lam=0.5, mu=-0.53):
    """Taubin smoothing (smooths without shrinking the volume)."""
    if iterazioni <= 0:
        return V
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    src = np.concatenate([E[:, 0], E[:, 1]])
    dst = np.concatenate([E[:, 1], E[:, 0]])
    nv = len(V)
    deg = np.bincount(src, minlength=nv).astype(np.float64)
    deg[deg == 0] = 1.0
    V = V.copy()
    for _ in range(iterazioni):
        for f in (lam, mu):
            Sx = np.bincount(src, weights=V[dst, 0], minlength=nv)
            Sy = np.bincount(src, weights=V[dst, 1], minlength=nv)
            Sz = np.bincount(src, weights=V[dst, 2], minlength=nv)
            L = np.stack([Sx, Sy, Sz], axis=1) / deg[:, None] - V
            V = V + f * L
    return V


# ======================================================================
# FEM EXPORT — mid-surface (single wall) + Gmsh remeshing
# ======================================================================

def mid_surface(bb_min, bb_max, cell_size, spacing_gen,
                     target_tris=None, target_P=None, log=None,
                     tpms_type=TPMS_DEFAULT):
    """Extracts the TPMS mid-surface: the g=0 isosurface,
    with NO thickness (single wall), clipped to the component.
    Returns welded (V, T). The thickness is NOT in the geometry:
    it will be a shell-element property in the solver."""
    pad = 2 * spacing_gen
    x = _axis_samples(bb_min[0] - pad, bb_max[0] + pad, spacing_gen)
    y = _axis_samples(bb_min[1] - pad, bb_max[1] + pad, spacing_gen)
    z = _axis_samples(bb_min[2] - pad, bb_max[2] + pad, spacing_gen)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    k = 2.0 * np.pi / cell_size
    g = TPMS[tpms_type]["g"](k * X, k * Y, k * Z).astype(np.float32)
    del X, Y, Z

    # clip to the component: keep f=0 only inside sdf>0
    if target_tris is not None:
        occ = occupancy_from_tris(target_P, target_tris, x, y, z,
                                spacing_gen, log)
        sdf = sdf_coarse(occ, spacing_gen, 4 * spacing_gen)
        del occ
        # combined field: mid-surface (g=0) intersected with the solid.
        # |g| is used as an "ultra-thin wall" cut by the sdf:
        # extract the isosurface of g, then drop outside triangles.
        field = g.copy()
        field[sdf < 0] = np.nan   # marcatore: fuori dal pezzo
    else:
        field = g

    # marching tetrahedra on the g=0 isosurface (NaN handled as outside)
    safe = np.where(np.isfinite(field), field, np.float32(1.0e3))
    safe += np.float32(1.0e-4) * spacing_gen
    tris = marching_tetrahedra(x, y, z, safe)

    # drop triangles whose barycenter falls outside the part
    if target_tris is not None:
        cen = tris.mean(axis=1)
        ix = np.clip(np.searchsorted(x, cen[:, 0]) - 1, 0, len(x) - 2)
        iy = np.clip(np.searchsorted(y, cen[:, 1]) - 1, 0, len(y) - 2)
        iz = np.clip(np.searchsorted(z, cen[:, 2]) - 1, 0, len(z) - 2)
        dentro = sdf[ix, iy, iz] > 0
        tris = tris[dentro]

    V, T = weld_vertices(tris, spacing_gen * 1e-3)
    return V, T


# SCOPE NOTE: automatic FEM-grade remeshing of TPMS (guaranteed
# near-equilateral triangles) is a research problem: it needs surface
# fitting or robust reparameterization (Gmsh stalls on TPMS,
# HyperMesh-like fitters are specialized software). Out of the MVP.
# Here we export a CLEAN mid-surface; analysis-grade quality, if
# needed, comes from the solver's interactive remesh (PrePoMax,
# Gmsh standalone).


def export_fem(V, T, elem_fem, fmt, path, log):
    """Exports the mid-surface (V,T) to .inp (Abaqus/CalculiX) or
    .msh (Gmsh 2.2), plus a helper STL. Internal writers in pure
    Python: no dependencies. Analysis-grade quality (near-equilateral
    triangles) is obtained with a remesh in the solver, if needed.
    Returns indicative quality metrics."""
    qual = _quality_numpy(V, T)
    if fmt == "inp":
        _write_inp(V, T, path)
    else:
        _write_msh(V, T, path)
    log("FEM mesh saved: {}\n".format(path))

    stl = path.rsplit(".", 1)[0] + ".stl"
    Mesh.Mesh([[tuple(p) for p in V[t]] for t in T.tolist()]).write(stl)
    log("Helper STL: {}\n".format(stl))
    return qual


def _quality_numpy(V, T):
    """Angular metrics of the (V,T) mesh in pure numpy."""
    P = V[T]
    a = P[:, 1] - P[:, 0]
    b = P[:, 2] - P[:, 1]
    c = P[:, 0] - P[:, 2]

    def ang(u, w):
        cs = ((u * w).sum(1) /
              (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-12))
        return np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))

    A = ang(-c, a)
    B = ang(-a, b)
    C = 180.0 - A - B
    mn = np.minimum(np.minimum(A, B), C)
    mx = np.maximum(np.maximum(A, B), C)
    return {"n_tri": int(len(T)),
            "ang_min": float(mn.min()),
            "ang_min_medio": float(mn.mean()),
            "ang_max": float(mx.max())}


def _write_inp(V, T, path):
    """Minimal Abaqus/CalculiX writer: nodes + S3 shell elements."""
    with open(path, "w") as f:
        f.write("*NODE\n")
        for i, p in enumerate(V, 1):
            f.write("{}, {:.6f}, {:.6f}, {:.6f}\n".format(
                i, p[0], p[1], p[2]))
        f.write("*ELEMENT, TYPE=S3, ELSET=GIROIDE\n")
        for e, t in enumerate(T, 1):
            f.write("{}, {}, {}, {}\n".format(
                e, t[0] + 1, t[1] + 1, t[2] + 1))


def _write_msh(V, T, path):
    """Minimal Gmsh MSH 2.2 writer: nodes + triangles."""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n{}\n".format(len(V)))
        for i, p in enumerate(V, 1):
            f.write("{} {:.6f} {:.6f} {:.6f}\n".format(
                i, p[0], p[1], p[2]))
        f.write("$EndNodes\n$Elements\n{}\n".format(len(T)))
        for e, t in enumerate(T, 1):
            f.write("{} 2 2 0 1 {} {} {}\n".format(
                e, t[0] + 1, t[1] + 1, t[2] + 1))
        f.write("$EndElements\n")


# ======================================================================
# GENERATION
# ======================================================================

def _axis_samples(lo, hi, spacing):
    n = max(8, int(np.ceil((hi - lo) / spacing)) + 1)
    return np.linspace(lo, hi, n, dtype=np.float32)


def estimate_points(bb_min, bb_max, spacing):
    pad = 2 * spacing
    tot = 1
    for a in range(3):
        n = max(8, int(np.ceil(
            (bb_max[a] - bb_min[a] + 2 * pad) / spacing)) + 1)
        tot *= n
    return tot


# ======================================================================
# MESH CLEANUP — slicer-ready geometry (consistent normals, no
# degenerate/duplicate faces). Pure computation on numpy (V, T).
# ======================================================================

def clean_mesh(V, T, eps_rel=1e-3, log=None):
    """Cleans a (V,T) mesh:
      - welds nearby vertices and removes duplicates;
      - drops degenerate (area ~0) and duplicate triangles;
      - makes normals consistent by orienting faces via adjacency
        and pointing them outwards (signed volume > 0).
    Returns clean (V, T). FreeCAD-independent."""
    V = np.asarray(V, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    if len(T) == 0:
        return V, T

    # The mesh arrives ALREADY welded and watertight from weld_vertices(),
    # and Taubin only moves vertices without changing topology (in the log
    # the open edges stay 0 up to here). So we do NOT re-weld and do NOT
    # delete anything:
    #  - re-welding after Taubin merges thin-wall vertices that ended up
    #    within tolerance, pinching the mesh and opening holes (the regular
    #    grid pattern on low-density panels);
    #  - deleting triangles by area/degeneracy removes faces that seal
    #    edges, opening other holes.
    # The only useful, safe thing here is ensuring the overall normal
    # direction; fine orientation is then left to FreeCAD's
    # harmonizeNormals(). Everything else is identity.
    Vn = np.asarray(V, dtype=np.float64)

    vol = _signed_volume(Vn, T)
    if vol < 0:
        T = T[:, ::-1].copy()

    if log:
        log("Mesh cleanup: {} vertices, {} triangles, "
            "consistent normals (no re-welding, mesh preserved).\n"
            .format(len(Vn), len(T)))
    return Vn, T


def _signed_volume(V, T):
    """Signed volume (sum of tetrahedra at the origin)."""
    a = V[T[:, 0]]
    b = V[T[:, 1]]
    c = V[T[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
