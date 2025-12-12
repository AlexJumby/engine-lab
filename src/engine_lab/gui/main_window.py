from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Engine Lab")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(QLabel("Hello from Engine Lab GUI"))

        self.setCentralWidget(central)
