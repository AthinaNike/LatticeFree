# -*- coding: utf-8 -*-
"""
FreeLattice — motore di calcolo (engine)
=========================================
Pipeline implicita per infill TPMS (giroide, Schwarz P, Diamond):
campi SDF, campo TPMS a due falde, shell, raccordo, grading di
densita', marching tetrahedra, pulizia mesh. Separato
dall'interfaccia (RNF6): nessuna dipendenza da Qt.
Richiede solo numpy; usa scipy se disponibile.
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
    # indice GLOBALE del nodo di griglia per ogni corner del cubo: serve
    # a interpolare ogni spigolo sempre nello stesso verso (vedi sotto).
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
            # Direzione "verso il fuori" per questa configurazione: dal
            # baricentro dei corner DENTRO (campo > 0) a quelli FUORI. La
            # normale di ogni triangolo deve concordare con questa direzione,
            # altrimenti scambio due vertici. Cosi' l'orientamento e' coerente
            # e corretto a prescindere dalla tabella _CASES.
            in_idx = [q for q in range(4) if (c >> q) & 1]
            out_idx = [q for q in range(4) if not (c >> q) & 1]
            outward = p[:, out_idx, :].mean(axis=1) - p[:, in_idx, :].mean(axis=1)
            for tri in triangles:
                corners = []
                for (a, b) in tri:
                    # Interpola SEMPRE dal nodo di griglia con id globale
                    # minore verso il maggiore: lo stesso spigolo, condiviso
                    # da tetraedri adiacenti, produce un punto bit-identico in
                    # entrambi -> la saldatura lo fonde sempre (niente cricche).
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
        raise RuntimeError("Nessuna superficie trovata: controlla i parametri.")
    return np.concatenate(tris, axis=0)


# ======================================================================
# CAMPI IMPLICITI
# ======================================================================

# ----------------------------------------------------------------------
# Funzioni nodali dei TPMS supportati. Argomenti gia' scalati (u = k*x).
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
# Registro TPMS.
# Per ogni tipo: funzione nodale g, tabelle di calibrazione densita' <->
# isovalore (frazione di volume {|g| < t}, MISURATA numericamente su
# griglia 220^3 cell-centered, adimensionale: vale per qualunque cella)
# e C_FORMA = media superficiale di |grad g| su g=0 (stesso metodo per
# tutti i tipi; sul giroide riproduce la tabella V0 entro lo 0.2%).
# Aggiungere un TPMS = aggiungere una voce qui (+ un comando in
# commands.py): tutto il resto della pipeline e' generico.
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

# range consigliato di densita' relativa (parete stampabile, pori aperti)
RHO_MIN = 0.10
RHO_MAX = 0.70


def campo_tpms(X, Y, Z, period, thickness, tipo=TPMS_DEFAULT):
    """Campo implicito del TPMS a DUE FALDE (positivo dentro la parete).

    La lamina {|g| < t} e' codificata come PRODOTTO delle due falde:

        f = (t - g)(t + g) / (2 t k) = (t^2 - g^2) / (2 t k)

    Stesso identico solido di (t - |g|)/k, ma f e' LISCIO ovunque
    (polinomiale in g): niente kink del valore assoluto sulla
    superficie media g=0. Le due superfici g=+t e g=-t vengono cosi'
    estratte dalla marching come falde separate e regolari — spariscono
    i pizzicamenti/non-manifold che il kink produceva sulle pareti
    sottili a bassa densita'. Vicino alla superficie f ha lo stesso
    gradiente (|grad g|/k) della vecchia lamina, quindi calibrazioni,
    spessori di parete e raccordo smooth-max restano invariati.
    `thickness` puo' essere scalare o campo (grading).
    """
    k = 2.0 * np.pi / period
    g = TPMS[tipo]["g"](k * X, k * Y, k * Z)
    t = np.maximum(thickness, 1.0e-6)
    return (t * t - g * g) / (2.0 * t * k)


def gyroid_mm(X, Y, Z, period, thickness):
    """Compatibilita' V0: campo del giroide (ora a due falde)."""
    return campo_tpms(X, Y, Z, period, thickness, "gyroid")


def densita_to_isovalore(rho, tipo=TPMS_DEFAULT):
    """Densita' relativa (0-1) -> isovalore di spessore del TPMS."""
    d = TPMS[tipo]
    return float(np.interp(rho, d["cal_rho"], d["cal_t"]))


def isovalore_to_densita(t, tipo=TPMS_DEFAULT):
    """Isovalore -> densita' relativa stimata (0-1)."""
    d = TPMS[tipo]
    return float(np.interp(t, d["cal_t"], d["cal_rho"]))


# Relazione isovalore <-> spessore FISICO di parete (mm).
# La parete e' la fascia |g| < t; spessore w = 2t/|grad g|, e
# |grad g| = (2pi/L) * C_FORMA(tipo), media misurata su g=0.
C_FORMA = TPMS["gyroid"]["c_forma"]   # compatibilita' V0


def spessore_parete_mm(t, cell_size, tipo=TPMS_DEFAULT):
    """Isovalore -> spessore fisico di parete in mm, per data cella."""
    cf = TPMS[tipo]["c_forma"]
    return 2.0 * t * cell_size / (2.0 * np.pi * cf)


def isovalore_da_spessore(w_mm, cell_size, tipo=TPMS_DEFAULT):
    """Spessore fisico (mm) -> isovalore, per data cella."""
    cf = TPMS[tipo]["c_forma"]
    return w_mm * 2.0 * np.pi * cf / (2.0 * cell_size)


def densita_min_stampabile(cell_size, nozzle_mm, tipo=TPMS_DEFAULT):
    """Densita' relativa minima per avere parete >= nozzle, data cella."""
    t_min = isovalore_da_spessore(nozzle_mm, cell_size, tipo)
    return isovalore_to_densita(t_min, tipo)


def spacing_consigliato(cell_size, rho_min_usata, tipo=TPMS_DEFAULT,
                        min_el_per_parete=3.0):
    """Elemento di griglia consigliato: il piu' fine tra cella/15 e
    parete_minima/3. E' la cura dei 'buchi esagonali' alle basse
    densita': con l'auto cella/15 una parete al 10% e' ~mezzo elemento
    (sotto-campionata per costruzione); qui la risoluzione insegue la
    parete piu' sottile davvero presente (grading incluso)."""
    t_min = densita_to_isovalore(rho_min_usata, tipo)
    w_min = spessore_parete_mm(t_min, cell_size, tipo)
    return max(0.05, min(cell_size / 15.0, w_min / min_el_per_parete))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def occupancy_da_tris(P, F, x, y, z, spacing, log=None):
    """Occupanza per parita' di attraversamenti in Z."""
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


def sdf_grezzo(occ, spacing, cap_mm):
    """Distanza con segno (mm), grezza (precisione ~mezzo voxel)."""
    if HAVE_SCIPY:
        d_in = _ndi.distance_transform_edt(occ, sampling=spacing)
        d_out = _ndi.distance_transform_edt(~occ, sampling=spacing)
        return (d_in - d_out).astype(np.float32)
    K = int(np.ceil(cap_mm / spacing)) + 2
    reached = ~occ
    d = np.zeros(occ.shape, dtype=np.float32)
    remaining = occ.copy()

    def dilata(m):
        r = m.copy()
        r[1:, :, :] |= m[:-1, :, :]; r[:-1, :, :] |= m[1:, :, :]
        r[:, 1:, :] |= m[:, :-1, :]; r[:, :-1, :] |= m[:, 1:, :]
        r[:, :, 1:] |= m[:, :, :-1]; r[:, :, :-1] |= m[:, :, 1:]
        return r

    for k in range(1, K + 1):
        reached = dilata(reached)
        new = reached & remaining
        d[new] = k * spacing
        remaining &= ~new
        if not remaining.any():
            break
    d[remaining] = (K + 1) * spacing
    return np.where(occ, d - 0.5 * spacing,
                    -0.5 * spacing).astype(np.float32)


def _dist_punto_triangolo(Q, a, b, c):
    """Distanza esatta punti->triangolo (Ericson), vettorizzata sui punti."""
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


def raffina_banda(P, F, x, y, z, sdf, band, log=None):
    """Sostituisce |sdf| con la distanza esatta nei voxel con
    |sdf| <= band. Il segno resta quello dell'occupanza."""
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
        d = _dist_punto_triangolo(Q, a, b, c3)
        sub_d = d_exact[i0:i1, j0:j1, k0:k1]
        cur = sub_d[li, lj, lk]
        upd = d < cur
        if upd.any():
            sub_d[li[upd], lj[upd], lk[upd]] = d[upd]

    valid = in_band & np.isfinite(d_exact)
    sdf[valid] = np.sign(sdf[valid]) * d_exact[valid].astype(np.float32)
    # i voxel di banda mai toccati (non dovrebbero esistere) restano grezzi
    return sdf


def sdf_box_demo(X, Y, Z, lato):
    h = lato / 2.0
    return np.minimum.reduce([h - np.abs(X), h - np.abs(Y),
                              h - np.abs(Z)]).astype(np.float32)


def smooth_max(a, b, k):
    """Unione morbida (R-function): raccordo di raggio ~k tra i campi."""
    if k <= 0.0:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + 0.25 * k * h * h


def campo_finale(sdf, gy, t_shell, r_blend=0.0):
    if t_shell > 0.0:
        skin = np.minimum(sdf, t_shell - sdf)
        core = sdf - t_shell
        infill = np.minimum(core, gy)
        # unione morbida: la giunzione shell-giroide nasce raccordata
        return smooth_max(skin, infill, r_blend)
    return np.minimum(sdf, gy)


# ======================================================================
# POST-PROCESSING MESH
# ======================================================================

def salda_vertici(tris, eps):
    """Triangle soup -> (V, T) con vertici unificati."""
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
    """Smoothing Taubin (leviga senza ritirare il volume)."""
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
# EXPORT FEM — superficie media (parete singola) + remeshing Gmsh
# ======================================================================

def superficie_media(bb_min, bb_max, cell_size, spacing_gen,
                     target_tris=None, target_P=None, log=None,
                     tipo=TPMS_DEFAULT):
    """Estrae la superficie media del TPMS: isosuperficie g=0,
    SENZA spessore (parete singola), ritagliata sul componente.
    Restituisce (V, T) saldati. Lo spessore NON e' nella geometria:
    sara' una proprieta' degli elementi shell nel solutore."""
    pad = 2 * spacing_gen
    x = _axis_samples(bb_min[0] - pad, bb_max[0] + pad, spacing_gen)
    y = _axis_samples(bb_min[1] - pad, bb_max[1] + pad, spacing_gen)
    z = _axis_samples(bb_min[2] - pad, bb_max[2] + pad, spacing_gen)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    k = 2.0 * np.pi / cell_size
    g = TPMS[tipo]["g"](k * X, k * Y, k * Z).astype(np.float32)
    del X, Y, Z

    # ritaglio sul componente: tengo f=0 solo dentro l'sdf>0
    if target_tris is not None:
        occ = occupancy_da_tris(target_P, target_tris, x, y, z,
                                spacing_gen, log)
        sdf = sdf_grezzo(occ, spacing_gen, 4 * spacing_gen)
        del occ
        # campo combinato: superficie media (g=0) intersecata col solido.
        # uso |g| come "parete sottilissima" tagliata dall'sdf:
        # estraggo l'isosuperficie di g, poi scarto i triangoli fuori.
        field = g.copy()
        field[sdf < 0] = np.nan   # marcatore: fuori dal pezzo
    else:
        field = g

    # marching tetrahedra sull'isosuperficie g=0 (gestendo i NaN come fuori)
    safe = np.where(np.isfinite(field), field, np.float32(1.0e3))
    safe += np.float32(1.0e-4) * spacing_gen
    tris = marching_tetrahedra(x, y, z, safe)

    # scarto i triangoli il cui baricentro cade fuori dal pezzo
    if target_tris is not None:
        cen = tris.mean(axis=1)
        ix = np.clip(np.searchsorted(x, cen[:, 0]) - 1, 0, len(x) - 2)
        iy = np.clip(np.searchsorted(y, cen[:, 1]) - 1, 0, len(y) - 2)
        iz = np.clip(np.searchsorted(z, cen[:, 2]) - 1, 0, len(z) - 2)
        dentro = sdf[ix, iy, iz] > 0
        tris = tris[dentro]

    V, T = salda_vertici(tris, spacing_gen * 1e-3)
    return V, T


# NOTA DI SCOPE: il remeshing automatico FEM-grade dei TPMS (triangoli
# quasi-equilateri garantiti) e' un problema di ricerca: richiede fitting
# di superfici o reparametrizzazione robusta (Gmsh stalla sui TPMS,
# HyperMesh-like fitter sono software specializzati). Fuori dall'MVP.
# Qui esportiamo la superficie media PULITA; la qualita' di calcolo,
# se serve, si ottiene col remesh interattivo del solutore (PrePoMax,
# Gmsh standalone).


def export_fem(V, T, elem_fem, formato, path, log):
    """Esporta la superficie media (V,T) in .inp (Abaqus/CalculiX) o
    .msh (Gmsh 2.2), piu' un STL di servizio. Writer interni in puro
    Python: nessuna dipendenza. La qualita' di calcolo (triangoli
    quasi-equilateri) si ottiene con un remesh nel solutore, se serve.
    Ritorna metriche di qualita' indicative."""
    qual = _qualita_numpy(V, T)
    if formato == "inp":
        _scrivi_inp(V, T, path)
    else:
        _scrivi_msh(V, T, path)
    log("FEM mesh saved: {}\n".format(path))

    stl = path.rsplit(".", 1)[0] + ".stl"
    Mesh.Mesh([[tuple(p) for p in V[t]] for t in T.tolist()]).write(stl)
    log("Helper STL: {}\n".format(stl))
    return qual


def _qualita_numpy(V, T):
    """Metriche angolari della mesh (V,T) in puro numpy."""
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


def _scrivi_inp(V, T, path):
    """Writer Abaqus/CalculiX minimale: nodi + elementi shell S3."""
    with open(path, "w") as f:
        f.write("*NODE\n")
        for i, p in enumerate(V, 1):
            f.write("{}, {:.6f}, {:.6f}, {:.6f}\n".format(
                i, p[0], p[1], p[2]))
        f.write("*ELEMENT, TYPE=S3, ELSET=GIROIDE\n")
        for e, t in enumerate(T, 1):
            f.write("{}, {}, {}, {}\n".format(
                e, t[0] + 1, t[1] + 1, t[2] + 1))


def _scrivi_msh(V, T, path):
    """Writer Gmsh MSH 2.2 minimale: nodi + triangoli."""
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
# GENERAZIONE
# ======================================================================

def _axis_samples(lo, hi, spacing):
    n = max(8, int(np.ceil((hi - lo) / spacing)) + 1)
    return np.linspace(lo, hi, n, dtype=np.float32)


def stima_punti(bb_min, bb_max, spacing):
    pad = 2 * spacing
    tot = 1
    for a in range(3):
        n = max(8, int(np.ceil(
            (bb_max[a] - bb_min[a] + 2 * pad) / spacing)) + 1)
        tot *= n
    return tot


# ======================================================================
# PULIZIA MESH — geometria pronta per lo slicer (normali coerenti,
# niente facce degeneri/duplicate). Calcolo puro su (V, T) numpy.
# ======================================================================

def pulisci_mesh(V, T, eps_rel=1e-3, log=None):
    """Ripulisce una mesh (V,T):
      - salda i vertici vicini e rimuove i duplicati;
      - elimina i triangoli degeneri (area ~0) e i duplicati;
      - rende le normali coerenti orientando le facce per adiacenza
        e rivolgendole verso l'esterno (volume con segno > 0).
    Restituisce (V, T) puliti. Indipendente da FreeCAD."""
    V = np.asarray(V, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    if len(T) == 0:
        return V, T

    # La mesh arriva GIA' saldata e watertight da salda_vertici(), e il
    # Taubin sposta soltanto i vertici senza cambiare la topologia (nel log
    # i bordi restano 0 fino a qui). Quindi NON ri-saldiamo e NON cancelliamo
    # nulla:
    #  - ri-saldare dopo il Taubin fonde i vertici di pareti sottili finiti
    #    entro la tolleranza, pinza la maglia e apre buchi (il reticolo
    #    regolare sui pannelli a bassa densita');
    #  - cancellare triangoli per area/degenerazione toglie facce che
    #    sigillano spigoli, aprendo altri buchi.
    # L'unica cosa utile e sicura qui e' garantire il verso d'insieme delle
    # normali; l'orientamento fine e' poi affidato a harmonizeNormals() di
    # FreeCAD. Tutto il resto e' identita'.
    Vn = np.asarray(V, dtype=np.float64)

    vol = _volume_segno(Vn, T)
    if vol < 0:
        T = T[:, ::-1].copy()

    if log:
        log("Mesh cleanup: {} vertices, {} triangles, "
            "consistent normals (no re-welding, mesh preserved).\n"
            .format(len(Vn), len(T)))
    return Vn, T


def _volume_segno(V, T):
    """Volume con segno (somma dei tetraedri all'origine)."""
    a = V[T[:, 0]]
    b = V[T[:, 1]]
    c = V[T[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
