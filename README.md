# LatticeFree

** Note by Veronica Massara **
This workbench has been created as a follow up of my Master Degree Thesis. The code was made with Claude Opus and Fable5. I'm not a professional programmer, this workbench was created as an engineering need to create parts and components with a parametric TPMS infill


**A free TPMS lattice-infill generator for FreeCAD.** Fill any solid with a
**Gyroid**, **Schwarz P**, or **Diamond** lattice using implicit modelling:
outer shell, smooth fillet at the shell/lattice junction, relative-density
control, density grading (dense skin, light core), and FEM mid-surface export.
No CAD booleans, no mandatory external dependencies.

Under the hood it runs a fully implicit pipeline — the signed distance field of
the part is combined with the TPMS field via `min`/`max`/smooth-max operations,
and the result is extracted as a single watertight mesh. This keeps the geometry
robust even at low densities, where boolean-based approaches typically fall
apart.

> Icon and screenshots go in `Resources/` — drop a render of a graded gyroid
> cube here to make the repo landing page speak for itself.

---

## Why this project

Commercial CAD packages hide TPMS infill behind expensive lattice modules.
LatticeFree does it inside FreeCAD, for free, with the parameters that actually
matter for **3D printing** and **lightweight structural parts** exposed and
explained — not buried. The calculation core (`engine.py`) has no Qt or GUI
dependency, so the same math can be reused in a standalone script or CLI.

---

## The parameters, explained

This is the part that matters most: what each control does, sensible values, and
**why** it changes the result. The dialog is identical for all three TPMS types.

### TPMS type — Gyroid / Schwarz P / Diamond
The surface family the infill is built on. All three are triply-periodic minimal
surfaces (smooth, self-supporting, no flat internal ceilings to bridge), but they
differ in channel topology and stiffness-to-weight behaviour. Each type ships
with its **own measured density↔isovalue calibration** and wall-thickness
constant, so a "30%" gyroid and a "30%" Schwarz P really do land at the same
relative density. Pick the button for the surface you want; everything else is
shared.

### Cell size (`Dimensione cella`, mm)
The size of one repeating unit. This is **independent of density**: it sets how
*coarse or fine* the pattern is, not how much material there is. A large cell
gives few big channels (lighter feel, faster to print, weaker in thin sections);
a small cell gives a fine, dense-looking texture with more walls per millimetre.
Rule of thumb: use enough cells across the part (≥ 3–4) for the density
calibration to hold.

### Relative density, outer (`Densità relativa esterna`, %)
The headline structural knob: how much of the volume is wall, guided **10–70%**.
It maps to an isovalue through a *measured* calibration table (the periodic
average over the infinite lattice), and the tool reports the resulting physical
wall thickness in mm. Caveat worth knowing: on parts only 1–2 cells thick the
realized density can drift a few percent from target (cell-clipping statistics,
most noticeable on Schwarz P); with ≥ 3–4 cells across the part it converges.

### Grading (dense skin → light core)
Optional. Instead of a uniform density, the wall thickness varies **smoothly**
from a denser outer skin to a lighter core, driven by the distance field. The
transition is gradual over the whole volume *on purpose*: an abrupt jump between
a stiff zone and a soft zone concentrates stress right at the interface and
becomes a crack initiation site — the same reason the shell/lattice junction is
filleted rather than left sharp. Smooth stiffness gradient, no discontinuity, no
crack nucleation.

### Nozzle diameter (`Diametro nozzle`, mm, default 0.2)
The printability guard for grading. From the cell size and nozzle diameter the
tool computes the **minimum printable wall**, and clamps the interior density so
it never asks for walls thinner than the nozzle can lay down. If your requested
core density would fall below that floor, it's raised automatically and you get a
warning — no more sub-resolution "ghost" walls that vanish at slicing time.

### Fillet radius (`Raggio raccordo`, mm)
Rounds the junction where the lattice meets the outer shell. A sharp junction is
a stress raiser; the fillet blends the two so load transfers smoothly. Small
values are usually enough; it mainly affects the shell/lattice transition, not
the bulk of the infill.

### Shell thickness (`Spessore shell`, mm)
The solid outer skin wrapping the lattice. `0` leaves the lattice exposed (open
cells at the surface — useful for flow, filtration, or a visible pattern); a
non-zero value gives a closed, load-spreading outer wall. The shell follows
holes and cavities parametrically.

### Grid element size (`Dimensione elemento griglia`, mm) — resolution
The sampling resolution of the implicit field. **Leave it on `auto`** unless you
have a reason not to: auto picks the finer of `cell/15` and `min-wall/3`, so thin
walls at low density are always resolved. This is the single most common source
of confusion: if the grid is too coarse relative to the thinnest wall, you get
**hexagonal holes** — walls under-sampled by construction, *not* a design flaw.
The generation report prints the cell/element ratio and warns whenever the
thinnest wall drops below 2 grid elements. Grid cost grows with the **cube** of
resolution, so very fine grids eat RAM fast; the dialog blocks configurations
above ~40 M points.

### Output — mesh vs solid CAD
`mesh` is fast and is what you want for printing. `solid CAD` converts the mesh to
a BRep solid (slow) and is only worth it when a true CAD body is needed
downstream.

### Smoothing / Decimation
Mesh post-processing. Taubin smoothing rounds off the faceting without shrinking
the part; decimation reduces triangle count for lighter files. Both are optional.

### FEM mid-surface export
Exports the lattice mid-surface for analysis, to `.inp` (CalculiX/Abaqus) or
`.msh` (Gmsh). The export is clean and watertight but **not FEM-grade** on its
own — run a remesh in your solver (PrePoMax, standalone Gmsh) for
analysis-quality elements.

---

## Requirements

- **FreeCAD 1.0+** (uses the bundled Python, PySide, and `Mesh`/`Part` modules).
- **numpy** — bundled with FreeCAD.
- **scipy** — *optional*; if present, the signed distance field is computed
  exactly (Euclidean), otherwise a built-in fallback is used.
- No other dependencies. FEM export uses internal `.inp`/`.msh` writers.

## Installation (manual)

LatticeFree is a standard FreeCAD external workbench. Copy the `LatticeFree`
folder into your FreeCAD `Mod` directory:

- **macOS:** `~/Library/Application Support/FreeCAD/v1-1/Mod/`
- **Windows:** `%APPDATA%\FreeCAD\Mod\`
- **Linux:** `~/.local/share/FreeCAD/Mod/`

The final path must look like `.../Mod/LatticeFree/InitGui.py`. Fully quit and
restart FreeCAD, then pick **LatticeFree** in the workbench selector.

## Usage

1. (Optional) Select a solid in the document — the lattice is confined to its
   volume. With no selection, a demo cube is produced.
2. Click the button for the TPMS you want: **Genera infill Giroide**,
   **Schwarz P**, or **Diamond**.
3. Set the parameters in the dialog (see above) and generate. A parameter report
   and progress are printed in the report view.

---

## Scope — what works and what doesn't (honest)

**Works (validated):** Gyroid / Schwarz P / Diamond infill (watertight and
pinch-free even at low densities), parametric shell following holes and cavities,
exact-band SDF (no terracing on curved surfaces), smooth fillet, relative-density
control with a measured per-TPMS calibration, density grading with printability
clamp, wall-aware auto resolution, watertight mesh cleanup, FEM mid-surface
export.

**Out of scope (research-grade):**
- Automatic FEM-grade remeshing of the TPMS (remesh in your solver instead).
- Grading from an external optimization density map (e.g. topology-optimization
  output).
- Further TPMS (double gyroid, I-WP, Neovius…) — the engine has a TPMS registry,
  so adding one is a dictionary entry plus a command; planned.

---

## License

GNU General Public License v3.0 — see the [`LICENSE`](LICENSE) file for the full
text. FreeCAD itself is LGPL, which imposes no obstacle to distributing a GPLv3
workbench that runs on top of it.

## Maintainer

Veronica Massara.




**New in V1**
- **Three TPMS types**, one toolbar button each (same dialog, same
  parameters): Gyroid, Schwarz P, Diamond. Each type has its own
  measured density↔isovalue calibration and wall-thickness constant.
- **Two-sheet implicit field**: the slab `|g| < t` is now encoded as
  the smooth product `(t−g)(t+g)/(2tk)` instead of `t−|g|`. Same
  solid, but the field has no kink on the mid-surface, so the two
  walls are extracted as separate smooth sheets — the pinch/non-manifold
  artifacts on thin low-density walls are gone at the source.
- **Wall-aware auto resolution**: the auto grid element is now
  `min(cell/15, min_wall/3)` (grading-aware). This is the root fix for
  the "hexagonal holes" at low densities: with the old `cell/15` a 10%
  wall was ~half an element thick, i.e. under-sampled by construction.
  The dialog and the generation report warn whenever the thinnest wall
  falls below 2 grid elements.

---

## Requirements

- **FreeCAD 1.0+** (uses the bundled Python, PySide, and `Mesh`/`Part`
  modules).
- **numpy** — bundled with FreeCAD.
- **scipy** — *optional*. If present, the signed distance field is
  computed exactly (Euclidean); otherwise a built-in fallback is used.
- No other dependencies. The FEM export uses internal `.inp`/`.msh`
  writers — no `gmsh` module required.



> Tip: if the workbench does not show up, open the Python console
> (View > Panels > Python console) and check the report view for a
> line `LatticeFree workbench caricato.` at startup. If there is an
> import error, it will be printed there.

### Performance notes

- Grid size grows with the cube of resolution. On modest laptops, stay
  in **mesh** mode with **auto** element size; very fine grids and
  heavy smoothing can exhaust RAM.
- The dialog shows a live estimate of grid points and blocks
  configurations above ~40 million points.
- "Solid CAD" mode is slow (mesh→BRep conversion); use only when a
  true CAD solid is needed downstream.


Note on relative density: the calibration is the periodic average over
the infinite lattice. On parts only 1–2 cells thick the realized
density can deviate by a few percent (cell clipping statistics),
especially for Schwarz P — with ≥ 3–4 cells across the part it
converges to the target.


## Project structure

```
LatticeFree/
  InitGui.py            # workbench registration (loaded by FreeCAD)
  package.xml.bak       # Addon Manager metadata (disattivato: vedi nota avvio)
  README.md
  LICENSE
  Resources/icons/
    freelattice.svg     # workbench icon
    tpms_gyroid.svg     # per-TPMS toolbar icons
    tpms_schwarz_p.svg
    tpms_diamond.svg
  freelattice/
    __init__.py
    engine.py           # calculation core (no Qt, no FreeCAD GUI deps)
    commands.py         # GUI dialog + command (uses engine)
```

The calculation core (`engine.py`) is deliberately decoupled from the
interface, so it can be reused in a standalone app or CLI.

## Startup loading note (important)

This workbench installs the classic way: `Mod/LatticeFree/InitGui.py`.
On FreeCAD 1.1 an active `package.xml` makes the loader treat the addon as a
"package-format" module and look for a namespaced `freecad/<name>/init_gui.py`
layout — which this addon does not use — so the classic `InitGui.py` scan gets
bypassed and the workbench silently fails to appear at startup. For that reason
the metadata file ships here as `package.xml.bak` (inactive). Do not rename it
back to `package.xml` unless you also convert the addon to the namespaced layout.

On FreeCAD 1.x the user `Mod` directory is versioned, e.g. on macOS:
`~/Library/Application Support/FreeCAD/v1-1/Mod/`.
