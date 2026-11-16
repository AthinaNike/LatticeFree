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
"""LatticeFree workbench — FreeCAD registration.

Uses the 'Workbench' and 'Gui' names that FreeCAD injects into the
InitGui.py namespace during the startup scan (same pattern as the
workbenches that load correctly on this installation). Everything that
can fail (icon, command imports) is isolated in try/except, so an
error can never prevent the workbench registration.

External name: LatticeFree (Mod/LatticeFree folder, class
LatticeFreeWorkbench, LatticeFree labels). The internal Python package
stays 'freelattice' and the legacy command 'FreeLattice_Genera': these
are internal names invisible to the user.
"""

import os
import FreeCAD as App


class LatticeFreeWorkbench(Workbench):  # noqa: F821  (Workbench iniettato da FreeCAD)
    """Workbench for TPMS lattice infill generation."""

    MenuText = "LatticeFree V2"
    ToolTip = ("TPMS lattice infill generator (Gyroid, Schwarz P, Diamond) "
               "with topology optimization and strength check")

    def __init__(self):
        # Defensive icon path: an error here must never prevent the
        # workbench registration at startup.
        try:
            icon = os.path.join(App.getUserAppDataDir(), "Mod",
                                "LatticeFree", "Resources", "icons",
                                "freelattice.svg")
            if os.path.isfile(icon):
                self.__class__.Icon = icon
        except Exception:
            pass

    def Initialize(self):
        # Called the first time the workbench is activated.
        try:
            from freelattice import commands  # noqa: F401  (registra i comandi)
            cmds = list(commands.TPMS_COMMANDS)
            try:
                from freelattice import topopt_gui  # noqa: F401
                cmds += list(topopt_gui.TOPOPT_COMMANDS)
            except Exception:
                import traceback
                App.Console.PrintError(
                    "LatticeFree: topology optimization not available:\n"
                    + traceback.format_exc() + "\n")
            self.appendToolbar("LatticeFree", cmds)
            self.appendMenu("LatticeFree", cmds)
            App.Console.PrintMessage("LatticeFree V2 workbench caricato.\n")
        except Exception:
            import traceback
            App.Console.PrintError(
                "LatticeFree: errore nel caricamento dei comandi:\n"
                + traceback.format_exc() + "\n")

    def Activated(self):
        App.Console.PrintMessage("LatticeFree V2 attivo.\n")

    def Deactivated(self):
        pass

    def GetClassName(self):
        return "Gui::PythonWorkbench"


Gui.addWorkbench(LatticeFreeWorkbench())  # noqa: F821  (Gui iniettato da FreeCAD)
