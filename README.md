# LatticeFree V2

**A free TPMS lattice-infill generator for FreeCAD.** Fill any solid with a
**Gyroid**, **Schwarz P**, or **Diamond** lattice using implicit modelling:
outer shell, smooth fillet at the shell/lattice junction, relative-density
control, density grading (dense skin, light core), and FEM mid-surface export.
No CAD booleans, no mandatory external dependencies.

**New in V2:** the lattice density can come from a **topology optimization**
driven by the loads and constraints of the FEM workbench, with a **material
library**, a **strength check** (safety factor to yield and ultimate, interlayer
check) and **printability limits** built in.

Under the hood it runs a fully implicit pipeline — the signed distance field of
the part is combined with the TPMS field via `min`/`max`/smooth-max operations,
and the result is extracted as a single watertight mesh. This keeps the geometry
robust even at low densities, where boolean-based approaches typically fall
apart.

> Icon and screenshots go in `Resources/` — drop a render of a graded gyroid
> cube here to make the repo landing page speak for itself.

---

## What's new in V2

### Topology optimization — variable lattice density
A new toolbar button, **Topology optimization**, computes the optimal relative
density map of the lattice (maximum stiffness for a given mass) from loads and
constraints defined in the **FEM workbench** (Analysis + `Fixed`, `Force`,
`Pressure` on faces, edges or vertices), then hands it to the generator, which
builds the infill with a variable wall thickness following the map.

- **Lattice-aware material model.** Stiffness follows the homogenized
  Gibson–Ashby law of the TPMS, `E*/Es = C·ρ^n` (gyroid: C = 0.414, n = 1.386,
  from A. Mastrofini, *Gyroid mechanics*, 2022 — shell FEA of a cubic cell,
  fitted on 6–32 %). Intermediate densities *are* the lattice, so there is no
  SIMP penalization towards solid/void.
- **Density filter** with radius = TPMS cell: the density cannot vary faster
  than one cell. The outer shell is counted as solid material.
- **Pure numpy**: matrix-free PCG on 8-node hexahedra, Optimality Criteria
  update. No new dependency; typical runs take seconds to a few minutes.
- **Output:** 3D preview of the map in density bands, `.npz` map and `.vtk`
  file (ParaView) next to the document, and direct hand-off to the generator
  (checkbox **Density from topology optimization**). The last map survives a
  FreeCAD restart.

### Two objectives
- **Max stiffness for a given mass** — you set the mean density (the mass), the
  solver distributes it where it stiffens the part most.
- **Min mass for the target safety factor** — you set the safety factor (e.g.
  2.0), the solver finds the **lightest** lattice that still meets it. It
  bisects on the mean density, computing the stiffness-optimal map at every
  step and checking its strength, and also finds the uniform lattice that
  meets the same safety factor, so the report shows the real mass saved (e.g.
  "uniform lattice with the same FS: 101 g → mass saved 41 %"). If even the
  maximum density cannot reach the target, it says so and suggests a thicker
  shell, a higher maximum density or a stronger material.

### Materials and strength check
- **Material library** — PLA, PETG, ABS, ASA, PA12 and *Custom*: Young modulus,
  Poisson ratio, yield and ultimate strength, density. Values are typical for
  printed parts (XY): enter your filament datasheet.
- **Strength verification** after the optimization, for both the optimized and
  a uniform lattice of the same mass:
  - lattice strength from `σ*/σs = C2·ρ^m` (default C2 = 0.30, m = 1.5,
    Gibson–Ashby, indicative and editable);
  - solid shell checked with the full material strength;
  - in-plane check with von Mises, **interlayer check** with the tensile stress
    along the build axis against `factor × σs` (FDM: 0.3–0.7);
  - safety factor to yield and to ultimate, critical point, yield-load
    estimate, mass in grams;
  - safety-factor map in the `TopOpt_safety` group (red = yields) and in the
    `.vtk` file.

### Printability limits
- **Minimum wall** = extrusion width × number of lines (default 2 × 0.45 mm)
  → minimum density for the chosen cell.
- **Minimum pore diameter** (default 1.5 mm) → maximum density, from measured
  pore tables (largest inscribed sphere in the void) for each TPMS.
- **Generation grid check**: the thinnest wall must be resolvable within the
  point budget, otherwise the optimization does not start.
- Light lattices need larger cells: **Fit cell to target** picks the smallest
  cell that leaves room to optimize. With 0.9 mm walls the gyroid's minimum
  density is about 28 % at a 10 mm cell, 19 % at 15 mm, 14 % at 20 mm.
- The generator's *auto* grid element now stays within a 25 M point budget.

### Topology optimization — quick guide
1. Open your part. In the **FEM** workbench create an **Analysis**, add a
   **Fixed** constraint on the clamped faces and a **Force** and/or
   **Pressure** constraint on the loaded faces. No mesh or solver is needed.
2. Back in LatticeFree, press **Topology optimization**. Choose component,
   TPMS, cell (or **Fit cell to target**), printability limits, shell,
   objective (mean density = mass, or minimum mass for a target safety factor)
   and material. Without FEM constraints you can try the
   built-in demo cantilever.
3. Read the report view: iterations, stiffness gain and the strength check
   (optimized vs uniform). The density preview appears in the 3D view; the
   safety-factor map is in the hidden `TopOpt_safety` group.
4. Answer **Yes** to open the generator with the map already enabled, or later
   tick **Density from topology optimization** in any TPMS dialog. Keep cell
   size and shell the same as in the optimization (the report warns if they
   differ).

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
resolution, so very fine grids eat RAM fast: *auto* stays within a 25 M point
budget and the dialog blocks configurations above ~40 M points.

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
   **Schwarz P**, or **Diamond** — or start from **Topology optimization**
   (see *What's new in V2*) to get a variable-density map first.
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
- Stress-driven sizing: the optimizer maximizes stiffness and the strength is
  checked afterwards, not imposed. The check is linear and homogenized: wall
  buckling, fatigue and local stress concentrations inside a cell are not
  covered. Verify critical parts in a solver and by testing.
- Stiffness laws for Schwarz P and Diamond are provisional (gyroid values):
  enter measured C, n in the dialog when available.
- Further TPMS (double gyroid, I-WP, Neovius…) — the engine has a TPMS registry,
  so adding one is a dictionary entry plus a command; planned.

---

## License

GNU General Public License v3.0 — see the [`LICENSE`](LICENSE) file for the full
text (GPL-3.0-or-later, Copyright (C) 2026 Veronica Massara). FreeCAD itself is
LGPL, which imposes no obstacle to distributing a GPLv3 workbench that runs on
top of it.

The software is provided **without any warranty** (GPL sections 15–16): safety
factors and strength checks are engineering estimates and do not replace the
verification of the final part by testing.

## Maintainer

Veronica Massara.

## Author's note

** Note by Veronica Massara **
This workbench has been created as a follow up of my Master Degree Thesis. The code was made with Claude Opus and Fable5. I'm not a professional programmer, this workbench was created as an engineering need to create parts and components with a parametric TPMS infill




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
