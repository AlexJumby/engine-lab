import sys
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def run_app() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
