from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine_lab.models.hirth3203.params import MVEMParams


class MVEM2TEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params: MVEMParams | None = None, episode_seconds: float | None = None):
        super().__init__()
        self.p = params or MVEMParams()
        self.steps_per_ep = int((episode_seconds or self.p.time.episode_seconds) / self.p.time.dt)

        # --- spaces ---
        self.mfi_max = self.p.lim.mfi_max
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        high = np.array(
            [
                self.p.lim.n_max_rpm / 60.0,
                3.0,
                1.0,
                1.5,
                2.0,
                self.p.lim.lambda_max,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                0.0,
                0.2,
                0.0,
                0.2,
                -2.0,
                self.p.lim.lambda_min,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)

        # --- state ---
        self._t = 0
        self._rng = np.random.default_rng()
        self._n = self.p.op.n0_rps
        self._a = self.p.op.a0
        self._lambda = self.p.op.lambda0
        self._pm = self.p.phys.p0
        self._ps = self.p.phys.p0
        self._Ts = self.p.phys.T0
        self._theta = 0.0
        self._mass_bal_prev = 0.0
        self._delay = [0.0] * self.p.time.inj_delay_steps
        self._mff = 0.0
        self._mf_delay_steps = max(1, int(round(self.p.shaft.td / self.p.time.dt)))
        self._mf_delay = [0.0] * self._mf_delay_steps
        self._lam_ref_filt = self._lambda_ref(self._a)
        self._u_prev = 0.0
        self._lambda_prev = self._lambda
        self._stall_steps = 0

        self._a_schedule: list[float] = [self._a] * self.steps_per_ep
        self._prev_err: float = 0.0

        self._last_m_air = 0.0
        self._last_mf = 0.0
        self._last_mfi_cmd = 0.0
        self._last_P_in = 0.0
        self._last_P_f = 0.0
        self._last_P_prop = 0.0

    # ---------- Fuel film (Aquino, eq. (13)–(17)) ----------
    def _fuel_film_step(self, mfi: float) -> float:
        p = self.p
        dt = p.time.dt

        n_rpm = p.rps_to_rpm(self._n)
        n_krpm = n_rpm / 1000.0
        pm_bar = p.pa_to_bar(self._pm)

        if p.film.use_regressions:
            # χ(n, pm) по (17)
            chi = -0.277 * pm_bar - 0.055 * n_krpm + 0.68
            chi = float(np.clip(chi, 0.0, 0.9))

            # τ_ff(n, pm) по (16)
            tau_ff = (
                1.35 * (-0.672 * n_krpm + 1.68) * (pm_bar - 0.825) ** 2
                + (-0.06 * n_krpm + 0.15)
                + 0.56
            )
            tau_ff = float(np.clip(tau_ff, 0.05, 1.5))
        else:
            chi = p.film.x_const
            tau_ff = p.film.tau_ff_const

        tau_ff = float(max(tau_ff, 1e-3))

        self._mff = float(np.clip(self._mff, 0.0, 0.02))

        # (13) d m_ff / dt
        mff_dot = -self._mff / tau_ff + chi * mfi
        self._mff += dt * mff_dot
        self._mff = float(np.clip(self._mff, 0.0, 0.02))

        # (14) и (15)
        mfv = (1.0 - chi) * mfi
        mf = mfv + self._mff / tau_ff

        mf = float(np.clip(mf, 0.0, 0.01))
        return mf

    # ---------- Throttle / intake (eq. (2)–(5)) ----------
    def _b1(self, a: float) -> float:
        # (3)
        c = np.cos(a)
        return 1.0 - self.p.thr.a1 * c + self.p.thr.a2 * (c**2)

    def _b2(self, pr: float) -> float:
        # (4)
        pr = float(np.clip(pr, 1e-6, 1.0))
        if pr <= self.p.thr.pc:
            b2 = ((pr**self.p.thr.p1) - (pr**self.p.thr.p2)) ** 0.5 / self.p.thr.pn
            return float(max(b2, 0.0))
        return 1.0

    def _eta_v(self, n_rps: float, ps_pa: float) -> float:
        p = self.p
        sc = p.scav

        n_rpm = p.rps_to_rpm(n_rps)
        n_krpm = n_rpm / 1000.0
        ps_rel = ps_pa / p.phys.p0  # ps/p0

        eta = sc.eta_v0 + sc.k_n * (n_krpm - sc.n_ref_krpm) + sc.k_ps * (ps_rel - 1.0)
        return float(np.clip(eta, sc.eta_min, sc.eta_max))

    def _m_at(self, a: float, pm_pa: float) -> float:
        p0 = self.p.phys.p0
        T0 = self.p.phys.T0
        pr = max(pm_pa / p0, 1e-6)
        mat1 = self.p.thr.mat1

        m_at = self.p.thr.scale * mat1 * (p0 / (T0**0.5)) * self._b1(a) * self._b2(pr)
        return float(max(m_at, 0.0))

    # ---------- Port flow mas (Hendricks eq. (6)) ----------
    def _m_as(self, n_rps: float, pm_pa: float) -> float:
        p = self.p

        if p.port.use_hendricks:
            n, pm_bar = p.hendricks_inputs(n_rps, pm_pa)
            c0, c1, c2, c3 = p.port.c0, p.port.c1, p.port.c2, p.port.c3
            x = n * pm_bar
            mas = c0 + c1 * x + c2 * (n * (pm_bar**2)) + c3 * (n**2) * pm_bar
            mas *= p.port.scale
            return float(max(mas, 0.0))

        n_rpm = p.rps_to_rpm(n_rps)
        pm_bar = p.pa_to_bar(pm_pa)

        mas = p.port.k_as * max(n_rpm, 0.0) * max(pm_bar, 0.0)
        return float(max(mas, 1e-8))

    # ---------- Port flow to cylinder (Model-PA eq. (8) + (9)) ----------
    def _m_ac(self, n_rps: float, ps_pa: float) -> float:
        p = self.p
        n_rpm = p.rps_to_rpm(n_rps)
        ps_bar = p.pa_to_bar(ps_pa)  # в статье ps, бар

        # ηv = ηs = 1 - exp(-l0)
        eta_v = self._eta_v(n_rps, ps_pa)

        Vd = p.geo.Vd
        i_cyl = p.geo.i
        t_strokes = p.geo.t
        R = p.phys.R
        Ts = max(self._Ts, 200.0)

        mac = eta_v * n_rpm * Vd * p.bar_to_pa(ps_bar) * i_cyl / (60.0 * t_strokes * R * Ts)
        return float(max(mac, 0.0))

    # ---------- Manifold ODE (1) ----------
    def _dp_m(self, m_at: float, m_as: float) -> float:
        Tm = self.p.phys.T0
        Vm = self.p.geo.Vm
        return (self.p.phys.R * Tm / Vm) * (m_at - m_as)

        # ---------- Crankcase ODE (упрощённое eq. (7)) ----------

    def _dp_s(
        self,
        ps_pa: float,
        Ts: float,
        Vs: float,
        dVs_dt: float,
        m_as: float,
        mf: float,
        m_ac: float,
        dTs_dt: float,
    ) -> float:
        R = self.p.phys.R
        Ts = max(Ts, 200.0)
        Vs = max(Vs, 1e-6)
        mass_bal = m_as + mf - m_ac
        term_mass = (R * Ts / Vs) * mass_bal
        term_T = (ps_pa / Ts) * dTs_dt
        term_V = (ps_pa / Vs) * dVs_dt
        if Ts < 1e-3:
            return term_mass - term_V
        return term_mass + term_T - term_V

    def _piston_disp(self, theta: float) -> float:
        # X(theta) ≈ stroke/2 * (1 - cos θ)
        return 0.5 * self.p.geo.stroke * (1.0 - math.cos(theta))

    def _Vs(self, theta: float) -> float:
        X = self._piston_disp(theta)
        return self.p.geo.Vz - (math.pi / 4.0) * (self.p.geo.D**2) * X

    def _dVs_dt(self, theta: float, n_rps: float) -> float:
        # dθ/dt = 2π n (рад/с)
        dtheta_dt = 2.0 * math.pi * n_rps
        # dX/dθ = stroke/2 * sin θ
        dX_dtheta = 0.5 * self.p.geo.stroke * math.sin(theta)
        dX_dt = dX_dtheta * dtheta_dt
        return -(math.pi / 4.0) * (self.p.geo.D**2) * dX_dt

    def _dTs(
        self, m_as: float, mf: float, m_ac: float, Tm: float, Ts: float, d_mass_bal_dt: float
    ) -> float:
        k = self.p.phys.k
        mass_bal = m_as + mf - m_ac
        if abs(mass_bal) < 1e-9:
            return 0.0
        term_in = m_as * Tm
        term_out = (m_ac + (1.0 / k) * d_mass_bal_dt) * Ts
        return k / mass_bal * (term_in - term_out)

    # ---------- Friction loss Pf (19) ----------
    def _shaft_losses_power(self, n_rps: float) -> float:
        p = self.p
        n_rpm = p.rps_to_rpm(n_rps)
        n_krpm = n_rpm / 1000.0

        Vd_L = p.geo.Vd * p.geo.i * 1000.0
        a0, a1, a2 = p.loss.a0, p.loss.a1, p.loss.a2

        Pf_kw = n_krpm * (a0 + a1 * n_krpm + a2 * n_krpm**2) * Vd_L / (1.27 * p.loss.fvp)
        Pf_w = max(Pf_kw * 1000.0, 0.0)
        return Pf_w

    # ---------- Thermal efficiency hi (21)–(27) ----------
    def _thermal_efficiency(self, n_rps: float, lam: float, pm_pa: float) -> float:
        p = self.p
        n_rpm = p.rps_to_rpm(n_rps)
        n_krpm = max(n_rpm / 1000.0, 0.5)
        ps = float(pm_pa / p.phys.p0)

        # (22)
        hin = 0.558 * (1.0 - 0.392 * (n_krpm ** (-0.36)))

        # (23)
        hip = 0.9301 + 0.2154 * ps - 0.1657 * (ps**2)

        # (24)
        if lam <= 1.0:
            hil = -1.299 + 3.599 * lam - 1.332 * (lam**2)
        else:
            hil = -0.0205 + 1.741 * lam - 0.745 * (lam**2)

        theta = 25.0

        # (26)–(27)
        if n_krpm < 4.8:
            u1 = 47.31 * ps + 2.6214 + 4.7 * n_krpm
            theta_mbt = min(u1, 45.0)
        else:
            u2 = -56.55 * ps + 79.9
            theta_mbt = min(u2, 45.0)

        dtheta = theta - theta_mbt

        # (25)
        hiu = 0.7 + 0.024 * dtheta - 0.00048 * (dtheta**2)

        hi = hin * hip * hil * hiu * self.p.shaft.eta_scale
        hi = float(np.clip(hi, 0.02, 0.6))
        return hi

    # ---------- Propeller power ----------
    def _propeller_power(self, n_rps: float) -> float:
        omega = n_rps * 2.0 * math.pi
        return self.p.loss.k_prop * (omega**3)

    def _a_profile(self, k: int) -> float:
        if not self._a_schedule:
            return self._a
        idx = min(max(k, 0), self.steps_per_ep - 1)
        return self._a_schedule[idx]

    def _lambda_ref(self, a: float) -> float:
        if a <= 0.30:
            return 0.95
        elif a >= 0.50:
            return 1.10
        else:
            return 0.95 + (a - 0.30) * (1.10 - 0.95) / 0.20

    def _lambda_ref_filtered(self, a: float) -> float:
        lam_target = self._lambda_ref(a)
        alpha = 0.1
        self._lam_ref_filt = (1.0 - alpha) * self._lam_ref_filt + alpha * lam_target
        return self._lam_ref_filt

    def _crankshaft_step(self, mf: float):
        dt = self.p.time.dt
        p = self.p

        self._mf_delay.append(mf)
        mf_delayed = self._mf_delay.pop(0)

        mf_eff = (1.0 - p.shaft.kf) * mf_delayed

        hi = self._thermal_efficiency(self._n, self._lambda, self._pm)

        P_in = p.shaft.Hu * hi * mf_eff

        P_f = self._shaft_losses_power(self._n)

        P_b = self._propeller_power(self._n)

        omega = max(self._n * 2.0 * math.pi, 1.0)

        T_eng = (P_in - P_f) / omega
        T_load = P_b / omega

        domega = (P_in - P_f - P_b) / (p.shaft.J * omega)

        self._n += dt * domega / (2.0 * math.pi)
        self._n = float(np.clip(self._n, 0.0, p.lim.n_max_rpm / 60.0))

        self._last_P_in = float(P_in)
        self._last_P_f = float(P_f)
        self._last_P_prop = float(P_b)
        self._last_T_eng = float(T_eng)
        self._last_T_load = float(T_load)

    def _step_dynamics(self, mfi_cmd: float):
        dt = self.p.time.dt

        self._delay.append(mfi_cmd)
        mfi = self._delay.pop(0)
        mf = self._fuel_film_step(mfi)

        m_at = self._m_at(self._a, self._pm)
        m_as = self._m_as(self._n, self._pm)
        m_ac = self._m_ac(self._n, self._ps)

        self._pm += dt * self._dp_m(m_at, m_as)
        self._pm = float(np.clip(self._pm, 0.2 * self.p.phys.p0, 1.5 * self.p.phys.p0))
        # --- crank angle / Vs ---
        self._theta += 2.0 * math.pi * self._n * dt
        self._theta = self._theta % (2.0 * math.pi)
        Vs = self._Vs(self._theta)
        dVs_dt = self._dVs_dt(self._theta, self._n)
        # --- Ts ---
        Tm = self.p.phys.T0
        mass_bal = m_as + mf - m_ac
        d_mass_bal_dt = (mass_bal - self._mass_bal_prev) / dt
        self._mass_bal_prev = mass_bal

        dTs_dt = self._dTs(m_as, mf, m_ac, Tm, self._Ts, d_mass_bal_dt)
        self._Ts += dt * dTs_dt
        self._Ts = float(np.clip(self._Ts, 200.0, 800.0))

        # --- ps ---
        dps_dt = self._dp_s(self._ps, self._Ts, Vs, dVs_dt, m_as, mf, m_ac, dTs_dt)
        self._ps += dt * dps_dt
        self._ps = float(np.clip(self._ps, 0.2 * self.p.phys.p0, 1.4 * self.p.phys.p0))

        m_air = m_as
        Lth = self.p.phys.Lth
        eps = 1e-12
        F_EPS = 1e-5

        if mf < F_EPS:
            self._raw_lambda = self._lambda
        else:
            self._raw_lambda = m_air / (mf * Lth + eps)
        if self._raw_lambda > self.p.lim.lambda_misfire and self._n < 2500 / 60.0:
            mf = 0.0
            self._crankshaft_step(mf)
            self._lambda = self.p.lim.lambda_misfire
        else:
            self._lambda = float(np.clip(self._raw_lambda, 0.2, 20.0))

        self._crankshaft_step(mf)

        self._last_m_air = m_air
        self._last_mf = mf
        self._last_mfi_cmd = mfi_cmd
        self._last_m_at = m_at
        self._last_m_as = m_as
        self._last_m_ac = m_ac

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        self._rng = np.random.default_rng(seed)
        self._t = 0

        if self.p.drand.enable:

            def scale(x, pct):
                return x * float(self._rng.uniform(1.0 - pct, 1.0 + pct))

            self.p.shaft.J = scale(self.p.shaft.J, self.p.drand.J_pct)
            self.p.shaft.Hu = scale(self.p.shaft.Hu, self.p.drand.Hu_pct)
            self.p.geo.Vm = scale(self.p.geo.Vm, self.p.drand.Vm_pct)
            self.p.shaft.kf = scale(self.p.shaft.kf, self.p.drand.kf_pct)

        self._n = self.p.op.n0_rps * float(self._rng.uniform(0.9, 1.1))
        self._a_schedule = []
        if self._n <= self.p.lim.n_min_rpm / 60.0:
            self._stall_steps += 1
        else:
            self._stall_steps = 0

        # a = float(self._rng.uniform(0.35, 0.46))
        # steps_per_change = max(1, int(0.5 / self.p.time.dt))

        # for k in range(self.steps_per_ep):
        #    if k > 0 and k % steps_per_change == 0:
        #        a += float(self._rng.uniform(-0.04, 0.04))
        #        a = float(np.clip(a, 0.34, 0.46))
        #    self._a_schedule.append(a)

        # в reset, вместо random walk
        dt = self.p.time.dt
        self._a_schedule = []

        mode = self._rng.uniform()

        if mode < 0.25:
            a1 = 0.34
            a2 = 0.46
            mid = self.steps_per_ep // 2
            self._a_schedule = [a1] * mid + [a2] * (self.steps_per_ep - mid)

        elif mode < 0.5:
            a1 = 0.46
            a2 = 0.34
            mid = self.steps_per_ep // 2
            self._a_schedule = [a1] * mid + [a2] * (self.steps_per_ep - mid)

        elif mode < 0.75:
            t1 = int(3.0 / dt)
            t2 = int(7.0 / dt)
            t1 = max(1, min(t1, self.steps_per_ep - 2))
            t2 = max(t1 + 1, min(t2, self.steps_per_ep - 1))

            self._a_schedule = [0.34] * t1 + [0.46] * (t2 - t1) + [0.40] * (self.steps_per_ep - t2)

        else:
            t_steps = 0
            while t_steps < self.steps_per_ep:
                dur_steps = int(
                    self._rng.integers(low=max(1, int(0.5 / dt)), high=max(2, int(3.0 / dt)))
                )
                a_level = float(self._rng.uniform(0.34, 0.46))
                for _ in range(dur_steps):
                    if t_steps >= self.steps_per_ep:
                        break
                    self._a_schedule.append(a_level)
                    t_steps += 1

        if len(self._a_schedule) < self.steps_per_ep:
            self._a_schedule += [self._a_schedule[-1]] * (self.steps_per_ep - len(self._a_schedule))

        self._a = self._a_schedule[0]

        self._lam_ref_filt = self._lambda_ref(self._a)
        self._lambda = float(self._rng.uniform(0.9, 1.1))
        self._pm = self.p.phys.p0 * float(self._rng.uniform(0.95, 1.05))
        self._ps = self.p.phys.p0 * float(self._rng.uniform(0.95, 1.05))
        self._Ts = self.p.phys.T0
        self._theta = float(self._rng.uniform(0.0, 2.0 * math.pi))
        self._mass_bal_prev = 0.0
        self._delay = [0.0] * self.p.time.inj_delay_steps
        self._mff = 0.0
        self._mf_delay = [0.0] * self._mf_delay_steps
        self._u_prev = 0.0
        self._stall_steps = 0

        lam_ref0 = self._lambda_ref(self._a)
        self._lam_ref_filt = lam_ref0
        err0 = self._lambda - lam_ref0

        self._prev_err = float(err0)
        self._lambda_prev = float(self._lambda)

        obs = np.array(
            [
                self._n,
                self._lambda,
                self._a,
                self.p.pa_to_bar(self._pm),
                err0,
                lam_ref0,
            ],
            dtype=np.float32,
        )
        return obs, {}

    def step(self, action: np.ndarray):
        self._t += 1
        self._a = self._a_profile(self._t - 1)

        alpha_u = 0.01

        u = float(np.clip(action[0], -1.0, 1.0))
        u_prev = self._u_prev

        u_filt = (1 - alpha_u) * u_prev + alpha_u * u
        du = u_filt - u_prev

        self._u_prev = u_filt

        u_used = u_filt
        alpha = 0.5 * (u_used + 1.0)

        mfi_cmd = self.p.lim.mfi_idle + alpha * (self.p.lim.mfi_max - self.p.lim.mfi_idle)

        self._step_dynamics(mfi_cmd)

        if self._n <= self.p.lim.n_min_rpm / 60.0:
            self._stall_steps += 1
        else:
            self._stall_steps = 0

        lam_ref = self._lambda_ref_filtered(self._a)
        err = self._lambda - lam_ref

        obs = np.array(
            [
                self._n,
                self._lambda,
                self._a,
                self.p.pa_to_bar(self._pm),
                err,
                lam_ref,
            ],
            dtype=np.float32,
        )

        dt = self.p.time.dt

        err_scale = 0.015
        deadband = 0.008

        err_abs = abs(err)
        err_db = max(err_abs - deadband, 0.0)
        err_n = err_db / err_scale
        cost = err_n**2

        if err > 0.0:
            cost *= 1.6
        else:
            cost *= 1.2

        # prev_err = float(self._prev_err)
        # prev_err_abs = abs(prev_err)
        # err_change = err_abs - prev_err_abs
        # if err_change > 0.0 and err_abs > deadband:
        #    cost += 0.5 * (err_change / err_scale)

        # self._prev_err = float(err)

        lam_over = max(self._raw_lambda - self.p.lim.lambda_misfire, 0.0)
        # if lam_over > 0.0:
        #    cost += 3.0 * (lam_over ** 2)
        cost += 15.0 * (lam_over**2)

        dlam = self._lambda - self._lambda_prev
        self._lambda_prev = float(self._lambda)

        dlam_abs = abs(dlam)
        dlam_db = max(dlam_abs - 0.002, 0.0)
        dlam_n = dlam_db / 0.04

        # du_abs = abs(du)

        w_dlam = 0.0
        w_u = 0.0
        w_du = 0.002

        if err_abs > 0.03:
            w_dlam = 0.0
            w_du = 0.0
            w_u = 0.001

        elif err_abs > 0.015:
            w_dlam = 0.08
            w_du = 0.08
            w_u = 0.003

        else:
            w_dlam = 0.25
            w_du = 0.25
            w_u = 0.008

        cost += w_dlam * (dlam_n**2)
        cost += w_u * (u_used**2)
        cost += w_du * (du**2)

        cost = float(min(cost, 10.0))

        terminated = self._stall_steps >= 10
        if terminated:
            cost += 50.0

        r = -cost * dt

        # terminated = False
        truncated = self._t >= self.steps_per_ep
        info = {
            "err": float(err),
            "u": float(u_used),
            "m_air": float(self._last_m_air),
            "mf": float(self._last_mf),
            "mfi_cmd": float(self._last_mfi_cmd),
            "n_rps": float(self._n),
            "lambda": float(self._lambda),
            "p_m_bar": float(self.p.pa_to_bar(self._pm)),
            "p_s_bar": float(self.p.pa_to_bar(self._ps)),
            "T_eng": float(self._last_T_eng),
            "T_load": float(self._last_T_load),
            "P_in": float(self._last_P_in),
            "P_f": float(self._last_P_f),
            "P_prop": float(self._last_P_prop),
            "lambda_raw": float(self._raw_lambda),
            "m_at": float(self._last_m_at),
            "m_as": float(self._last_m_as),
            "m_ac": float(self._last_m_ac),
            "a": float(self._a),
            "lambda_ref": float(lam_ref),
        }
        return obs, float(r), terminated, truncated, info
