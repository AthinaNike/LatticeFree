# -*- coding: utf-8 -*-
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

    MenuText = "LatticeFree"
    ToolTip = "TPMS lattice infill generator (Gyroid, Schwarz P, Diamond)"

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
            self.appendToolbar("LatticeFree", cmds)
            self.appendMenu("LatticeFree", cmds)
            App.Console.PrintMessage("LatticeFree workbench caricato.\n")
        except Exception:
            import traceback
            App.Console.PrintError(
                "LatticeFree: errore nel caricamento dei comandi:\n"
                + traceback.format_exc() + "\n")

    def Activated(self):
        App.Console.PrintMessage("LatticeFree attivo.\n")

    def Deactivated(self):
        pass

    def GetClassName(self):
        return "Gui::PythonWorkbench"


Gui.addWorkbench(LatticeFreeWorkbench())  # noqa: F821  (Gui iniettato da FreeCAD)
