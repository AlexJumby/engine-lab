from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from engine_lab.gui.main_window import MainWindow


def run_app() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
