from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from engine_lab.models.hirth3203.mvem import Hirth3203Engine
from engine_lab.models.hirth3203.params import MVEMParams


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    # ограничение интеграла (в единицах ДЕЛЬТЫ mfi, кг/с)
    i_min: float = -1.0e-3
    i_max: float = 1.0e-3


class PIDAFRController:
    """
    Простой ПИД-контроллер AFR (λ) с:
    - ошибкой по λ,
    - feed-forward по дросселю (больше газ → больше топлива),
    - анти-windup через ограничение интеграла.
    """

    def __init__(self, gains: PIDGains, params: MVEMParams | None = None):
        self.gains = gains
        self.p = params or MVEMParams()
        self.dt = self.p.time.dt

        self._i_term = 0.0
        self._prev_err = 0.0
        self._prev_t: float | None = None

    # --- служебные методы ---

    def reset(self) -> None:
        """Сброс внутреннего состояния контроллера."""
        self._i_term = 0.0
        self._prev_err = 0.0
        self._prev_t = None

    def _lambda_ref(self, a: float) -> float:
        """Та же логика, что и в env/движке: целевая λ от угла дросселя."""
        if a <= 0.30:
            return 0.95
        elif a >= 0.50:
            return 1.10
        else:
            return 0.95 + (a - 0.30) * (1.10 - 0.95) / 0.20

    # --- основной метод ---

    def compute(
        self,
        *,
        t: float,
        state: Dict,
        a: float,
        engine: Hirth3203Engine,
    ) -> Tuple[float, Dict]:
        """
        Считаем команду по топливу.

        Parameters
        ----------
        t : float
            Текущее время (сек).
        state : dict
            Состояние от Hirth3203Engine.get_state().
        a : float
            Текущее положение дросселя [рад].
        engine : Hirth3203Engine
            Экземпляр двигателя (для dt и λ_ref, если надо).

        Returns
        -------
        mfi_cmd : float
            Команда по топливу, кг/с.
        info : dict
            Отладочная информация (ошибка, члены ПИД и т.п.).
        """
        lam = state["lambda"]
        lam_ref = self._lambda_ref(a)
        # ошибка: положительная → смесь бедная (надо ДОБАВИТЬ топлива)
        err = lam - lam_ref

        # шаг интегрирования
        if self._prev_t is None:
            dt = engine.dt
        else:
            dt = max(t - self._prev_t, 1e-6)

        # --- ПИД ---
        # P
        p_term = self.gains.kp * err

        # I (с ограничением)
        self._i_term += self.gains.ki * err * dt
        self._i_term = float(np.clip(self._i_term, self.gains.i_min, self.gains.i_max))

        # D
        if self._prev_t is None:
            d_term = 0.0
        else:
            derr = (err - self._prev_err) / dt
            d_term = self.gains.kd * derr

        self._prev_err = err
        self._prev_t = t

        delta_mfi = p_term + self._i_term + d_term
        # ограничим поправку, чтобы ПИД не мог разнести систему
        delta_mfi = float(np.clip(delta_mfi, -1.0e-3, 1.0e-3))

        # --- feed-forward по дросселю ---
        p = self.p
        a_min, a_max = p.lim.a_min, p.lim.a_max
        alpha_ff = (a - a_min) / (a_max - a_min)
        alpha_ff = float(np.clip(alpha_ff, 0.0, 1.0))

        mfi_ff = p.lim.mfi_idle + alpha_ff * (p.lim.mfi_max - p.lim.mfi_idle)

        # итоговая команда
        mfi_cmd = mfi_ff + delta_mfi
        mfi_cmd = float(np.clip(mfi_cmd, p.lim.mfi_min, p.lim.mfi_max))

        info = {
            "lambda_ref": lam_ref,
            "err": err,
            "p_term": p_term,
            "i_term": self._i_term,
            "d_term": d_term,
            "delta_mfi": delta_mfi,
            "mfi_ff": mfi_ff,
            "mfi_cmd": mfi_cmd,
        }
        return mfi_cmd, info
