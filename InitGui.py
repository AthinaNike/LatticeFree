# -*- coding: utf-8 -*-
"""LatticeFree workbench — registrazione in FreeCAD.

Usa i nomi 'Workbench' e 'Gui' che FreeCAD inietta nel namespace di
InitGui.py durante lo scan di avvio (stesso pattern dei workbench che
caricano correttamente su questa installazione). Tutto cio' che puo'
fallire (icona, import dei comandi) e' isolato in try/except, cosi' un
errore non puo' impedire la registrazione del workbench.

Nome esterno: LatticeFree (cartella Mod/LatticeFree, classe
LatticeFreeWorkbench, etichette LatticeFree). Il pacchetto Python interno
resta 'freelattice' e il comando 'FreeLattice_Genera': sono nomi interni
invisibili all'utente.
"""

import os
import FreeCAD as App


class LatticeFreeWorkbench(Workbench):  # noqa: F821  (Workbench iniettato da FreeCAD)
    """Workbench per la generazione di infill lattice TPMS (giroide)."""

    MenuText = "LatticeFree"
    ToolTip = "TPMS lattice infill generator (Gyroid, Schwarz P, Diamond)"

    def __init__(self):
        # Percorso icona difensivo: un errore qui non deve mai impedire
        # la registrazione del workbench all'avvio.
        try:
            icon = os.path.join(App.getUserAppDataDir(), "Mod",
                                "LatticeFree", "Resources", "icons",
                                "freelattice.svg")
            if os.path.isfile(icon):
                self.__class__.Icon = icon
        except Exception:
            pass

    def Initialize(self):
        # Chiamato la prima volta che si attiva il workbench.
        try:
            from freelattice import commands  # noqa: F401  (registra i comandi)
            cmds = list(commands.COMANDI_TPMS)
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
