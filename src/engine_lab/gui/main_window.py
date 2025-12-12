from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from engine_lab.controllers.pid.pid_afr import PIDAFRController, PIDGains
from engine_lab.models.hirth3203.mvem import Hirth3203Engine

# --- опционально подключаем DDPG, если он есть в проекте ---
try:  # noqa: SIM105
    from engine_lab.controllers.ddpg.ddpg_afr import DDPGLambdaController  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    DDPGLambdaController = None  # type: ignore[assignment]


# ===== простой open-loop контроллер (feedforward по m_air и lambda_ref) =====
class OpenLoopAFRController:
    """
    Очень простой контроллер:
      - берёт текущий расход воздуха m_air
      - задаёт lambda_ref как функция дросселя a
      - считает требуемый mfi_cmd = m_air / (lambda_ref * L_st)
      - режет по лимитам
    Без интегратора и обратной связи по ошибке.
    """

    def __init__(self, params) -> None:
        self.p = params

    def reset(self) -> None:  # для единообразия с PID/DDPG
        pass

    def _lambda_ref(self, a: float) -> float:
        # тот же профиль, что и в модели
        if a <= 0.30:
            return 0.95
        elif a >= 0.50:
            return 1.10
        else:
            return 0.95 + (a - 0.30) * (1.10 - 0.95) / 0.20

    def compute(self, t: float, state: dict, a: float, engine: Hirth3203Engine) -> tuple[float, dict]:
        lam_ref = self._lambda_ref(a)
        m_air = float(state["m_air"])
        Lth = self.p.phys.Lth
        eps = 1e-12

        mfi_cmd = m_air / (lam_ref * Lth + eps)
        mfi_cmd = float(np.clip(mfi_cmd, self.p.lim.mfi_min, self.p.lim.mfi_max))

        info = {
            "lambda_ref": lam_ref,
        }
        return mfi_cmd, info


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Engine Lab – Hirth3203")
        self.resize(1100, 700)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # ===== левая панель – настройки =====
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 0)

        left_panel.addWidget(self._create_engine_group())
        left_panel.addWidget(self._create_controller_group())
        left_panel.addWidget(self._create_scenario_group())

        self.run_button = QPushButton("Run simulation")
        self.run_button.clicked.connect(self.on_run_clicked)
        left_panel.addWidget(self.run_button)

        self.status_label = QLabel("Ready.")
        left_panel.addWidget(self.status_label)
        left_panel.addStretch(1)

        # ===== правая панель – графики =====
        plots_layout = QVBoxLayout()
        main_layout.addLayout(plots_layout, 1)

        pg.setConfigOptions(antialias=True)

        # λ plot
        self.plot_lambda = pg.PlotWidget(title="AFR tracking")
        self.plot_lambda.showGrid(x=True, y=True, alpha=0.3)
        self.plot_lambda.setLabel("bottom", "t", units="s")
        self.plot_lambda.setLabel("left", "λ")

        self.curve_lambda = self.plot_lambda.plot(pen="c", name="λ")
        self.curve_lambda_ref = self.plot_lambda.plot(
            pen=pg.mkPen("w", style=QtCore.Qt.DashLine),
            name="λ_ref",
        )

        self.plot_lambda.addLegend()

        # n plot
        self.plot_n = pg.PlotWidget(title="Engine speed / fuel")
        self.plot_n.showGrid(x=True, y=True, alpha=0.3)
        self.plot_n.setLabel("bottom", "t", units="s")
        self.plot_n.setLabel("left", "n", units="rpm")

        self.curve_n = self.plot_n.plot(pen="y", name="n, rpm")
        self.curve_mfi = self.plot_n.plot(pen="m", name="mfi_cmd · 1e4")

        self.plot_n.addLegend()

        plots_layout.addWidget(self.plot_lambda, 1)
        plots_layout.addWidget(self.plot_n, 1)

    # ------------------------------------------------------------------ группы контролов

    def _create_engine_group(self) -> QGroupBox:
        box = QGroupBox("Engine")
        layout = QFormLayout(box)

        self.engine_combo = QComboBox()
        # пока один двигатель, но интерфейс уже готов под расширение
        self.engine_combo.addItem("Hirth3203 (2T)", userData="hirth3203")

        layout.addRow("Engine:", self.engine_combo)
        return box

    def _create_controller_group(self) -> QGroupBox:
        box = QGroupBox("Controller")
        layout = QFormLayout(box)

        # тип контроллера
        self.ctrl_combo = QComboBox()
        self.ctrl_combo.addItem("PID AFR", userData="pid")
        self.ctrl_combo.addItem("DDPG (RL)", userData="ddpg")
        self.ctrl_combo.addItem("Open loop (feedforward)", userData="open")
        self.ctrl_combo.currentIndexChanged.connect(self.on_ctrl_changed)

        # PID-гейны
        self.spin_kp = QDoubleSpinBox()
        self.spin_kp.setDecimals(5)
        self.spin_kp.setRange(0.0, 1.0)
        self.spin_kp.setSingleStep(0.0005)
        self.spin_kp.setValue(0.004)  # 4e-3

        self.spin_ki = QDoubleSpinBox()
        self.spin_ki.setDecimals(5)
        self.spin_ki.setRange(0.0, 1.0)
        self.spin_ki.setSingleStep(0.001)
        self.spin_ki.setValue(0.02)

        self.spin_kd = QDoubleSpinBox()
        self.spin_kd.setDecimals(5)
        self.spin_kd.setRange(0.0, 1.0)
        self.spin_kd.setSingleStep(0.0005)
        self.spin_kd.setValue(0.0)

        layout.addRow("Type:", self.ctrl_combo)
        layout.addRow("Kp:", self.spin_kp)
        layout.addRow("Ki:", self.spin_ki)
        layout.addRow("Kd:", self.spin_kd)

        return box

    def _create_scenario_group(self) -> QGroupBox:
        box = QGroupBox("Scenario: throttle step")
        layout = QFormLayout(box)

        self.spin_a1 = QDoubleSpinBox()
        self.spin_a1.setDecimals(3)
        self.spin_a1.setRange(0.0, 1.0)
        self.spin_a1.setSingleStep(0.01)
        self.spin_a1.setValue(0.34)

        self.spin_a2 = QDoubleSpinBox()
        self.spin_a2.setDecimals(3)
        self.spin_a2.setRange(0.0, 1.0)
        self.spin_a2.setSingleStep(0.01)
        self.spin_a2.setValue(0.46)

        self.spin_t_step = QDoubleSpinBox()
        self.spin_t_step.setDecimals(2)
        self.spin_t_step.setRange(0.0, 30.0)
        self.spin_t_step.setSingleStep(0.5)
        self.spin_t_step.setValue(5.0)

        self.spin_t_final = QDoubleSpinBox()
        self.spin_t_final.setDecimals(1)
        self.spin_t_final.setRange(0.1, 60.0)
        self.spin_t_final.setSingleStep(1.0)
        self.spin_t_final.setValue(10.0)

        layout.addRow("a1:", self.spin_a1)
        layout.addRow("a2:", self.spin_a2)
        layout.addRow("t_step, s:", self.spin_t_step)
        layout.addRow("t_final, s:", self.spin_t_final)

        return box

    # ------------------------------------------------------------------ slots

    @Slot(int)
    def on_ctrl_changed(self, index: int) -> None:  # noqa: ARG002
        kind = self.ctrl_combo.currentData()
        is_pid = kind == "pid"
        for w in (self.spin_kp, self.spin_ki, self.spin_kd):
            w.setEnabled(is_pid)

    @Slot()
    def on_run_clicked(self) -> None:
        self.status_label.setText("Running simulation...")
        self.run_button.setEnabled(False)
        try:
            self._run_simulation()
            self.status_label.setText("Done.")
        except Exception as exc:  # noqa: BLE001
            # покажем ошибку в статус-баре и пробросим дальше, чтобы было видно в консоли
            self.status_label.setText(f"Error: {exc!r}")
            raise
        finally:
            self.run_button.setEnabled(True)

    # ------------------------------------------------------------------ логика симуляции

    def _create_controller(self, eng: Hirth3203Engine):
        kind = self.ctrl_combo.currentData()

        if kind == "pid":
            gains = PIDGains(
                kp=self.spin_kp.value(),
                ki=self.spin_ki.value(),
                kd=self.spin_kd.value(),
                i_min=-1.0e-3,
                i_max=+1.0e-3,
            )
            ctrl = PIDAFRController(gains, params=eng.p)
            ctrl.reset()
            return ctrl

        if kind == "open":
            ctrl = OpenLoopAFRController(eng.p)
            ctrl.reset()
            return ctrl

        if kind == "ddpg":
            if DDPGLambdaController is None:
                raise RuntimeError(
                    "DDPGAFRController не найден. "
                    "Подключи engine_lab.controllers.ddpg_afr.DDPGAFRController "
                    "или выбери другой тип контроллера."
                )
            # здесь предполагается, что у тебя есть класс с таким интерфейсом:
            ctrl = DDPGLambdaController(
                params=eng.p,
                device="cpu",
            )
            return ctrl

        raise RuntimeError(f"Unknown controller type: {kind!r}")

    def _run_simulation(self) -> None:
        # --- создаём движок и контроллер ---
        eng = Hirth3203Engine()
        ctrl = self._create_controller(eng)

        state = eng.reset()

        a1 = self.spin_a1.value()
        a2 = self.spin_a2.value()
        t_step = self.spin_t_step.value()
        t_final = self.spin_t_final.value()

        dt = eng.dt
        steps = int(t_final / dt)

        t_list: List[float] = []
        lam_list: List[float] = []
        lam_ref_list: List[float] = []
        n_rpm_list: List[float] = []
        mfi_list: List[float] = []

        for k in range(steps):
            t = k * dt
            a = a1 if t < t_step else a2

            mfi_cmd, info = ctrl.compute(t=t, state=state, a=a, engine=eng)
            state = eng.step(mfi_cmd, a)

            t_list.append(t)
            lam_list.append(state["lambda"])
            lam_ref_list.append(info.get("lambda_ref", 1.0))
            n_rpm_list.append(state["n_rps"] * 60.0)
            mfi_list.append(mfi_cmd * 1e4)  # масштаб для графика

        # --- обновляем графики ---
        t_arr = np.asarray(t_list)

        self.curve_lambda.setData(t_arr, np.asarray(lam_list))
        self.curve_lambda_ref.setData(t_arr, np.asarray(lam_ref_list))

        self.curve_n.setData(t_arr, np.asarray(n_rpm_list))
        self.curve_mfi.setData(t_arr, np.asarray(mfi_list))
