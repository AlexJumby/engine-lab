import math

import numpy as np

from engine_lab.models.hirth3203.params import MVEMParams

class Hirth3203Engine:
    def __init__(self, params: MVEMParams | None = None):
        self.p = params or MVEMParams()
        self.dt = self.p.time.dt

        self._mf_delay_steps = max(1, int(round(self.p.shaft.td / self.p.time.dt)))
 
        self._lam_ref_filt = 0.0
        self._last_T_eng = 0.0
        self._last_T_load = 0.0
        self._last_m_at = 0.0
        self._last_m_as = 0.0
        self._last_m_ac = 0.0

        self.reset()

    def _fuel_film_step(self, mfi: float) -> float:
        p = self.p
        dt = p.time.dt

        n_rpm = p.rps_to_rpm(self._n)
        n_krpm = n_rpm / 1000.0
        pm_bar = p.pa_to_bar(self._pm)

        if p.film.use_regressions:
            chi = -0.277 * pm_bar - 0.055 * n_krpm + 0.68
            chi = float(np.clip(chi, 0.0, 0.9))

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

        mff_dot = -self._mff / tau_ff + chi * mfi
        self._mff += dt * mff_dot
        self._mff = float(np.clip(self._mff, 0.0, 0.02))

        mfv = (1.0 - chi) * mfi
        mf = mfv + self._mff / tau_ff

        mf = float(np.clip(mf, 0.0, 0.01))
        return mf

    def _b1(self, a: float) -> float:
        c = np.cos(a)
        return 1.0 - self.p.thr.a1 * c + self.p.thr.a2 * (c**2)

    def _b2(self, pr: float) -> float:
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
        ps_rel = ps_pa / p.phys.p0  

        eta = sc.eta_v0 + sc.k_n * (n_krpm - sc.n_ref_krpm) + sc.k_ps * (ps_rel - 1.0)
        return float(np.clip(eta, sc.eta_min, sc.eta_max))

    def _m_at(self, a: float, pm_pa: float) -> float:
        p0 = self.p.phys.p0
        T0 = self.p.phys.T0
        pr = max(pm_pa / p0, 1e-6)
        mat1 = self.p.thr.mat1

        m_at = self.p.thr.scale * mat1 * (p0 / (T0**0.5)) * self._b1(a) * self._b2(pr)
        return float(max(m_at, 0.0))

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

    def _m_ac(self, n_rps: float, ps_pa: float) -> float:
        p = self.p
        n_rpm = p.rps_to_rpm(n_rps)
        ps_bar = p.pa_to_bar(ps_pa) 

        eta_v = self._eta_v(n_rps, ps_pa)

        Vd = p.geo.Vd
        i_cyl = p.geo.i
        t_strokes = p.geo.t
        R = p.phys.R
        Ts = max(self._Ts, 200.0)

        mac = eta_v * n_rpm * Vd * p.bar_to_pa(ps_bar) * i_cyl / (60.0 * t_strokes * R * Ts)
        return float(max(mac, 0.0))

    def _dp_m(self, m_at: float, m_as: float) -> float:
        Tm = self.p.phys.T0
        Vm = self.p.geo.Vm
        return (self.p.phys.R * Tm / Vm) * (m_at - m_as)
    
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
        return 0.5 * self.p.geo.stroke * (1.0 - math.cos(theta))

    def _Vs(self, theta: float) -> float:
        X = self._piston_disp(theta)
        return self.p.geo.Vz - (math.pi / 4.0) * (self.p.geo.D**2) * X

    def _dVs_dt(self, theta: float, n_rps: float) -> float:
        dtheta_dt = 2.0 * math.pi * n_rps
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

    def _shaft_losses_power(self, n_rps: float) -> float:
        p = self.p
        n_rpm = p.rps_to_rpm(n_rps)
        n_krpm = n_rpm / 1000.0

        Vd_L = p.geo.Vd * p.geo.i * 1000.0
        a0, a1, a2 = p.loss.a0, p.loss.a1, p.loss.a2

        Pf_kw = n_krpm * (a0 + a1 * n_krpm + a2 * n_krpm**2) * Vd_L / (1.27 * p.loss.fvp)
        Pf_w = max(Pf_kw * 1000.0, 0.0)
        return Pf_w

    def _thermal_efficiency(self, n_rps: float, lam: float, pm_pa: float) -> float:
        p = self.p
        n_rpm = p.rps_to_rpm(n_rps)
        n_krpm = max(n_rpm / 1000.0, 0.5)
        ps = float(pm_pa / p.phys.p0)

        hin = 0.558 * (1.0 - 0.392 * (n_krpm ** (-0.36)))

        hip = 0.9301 + 0.2154 * ps - 0.1657 * (ps**2)

        if lam <= 1.0:
            hil = -1.299 + 3.599 * lam - 1.332 * (lam**2)
        else:
            hil = -0.0205 + 1.741 * lam - 0.745 * (lam**2)

        theta = 25.0

        if n_krpm < 4.8:
            u1 = 47.31 * ps + 2.6214 + 4.7 * n_krpm
            theta_mbt = min(u1, 45.0)
        else:
            u2 = -56.55 * ps + 79.9
            theta_mbt = min(u2, 45.0)

        dtheta = theta - theta_mbt

        hiu = 0.7 + 0.024 * dtheta - 0.00048 * (dtheta**2)

        hi = hin * hip * hil * hiu * self.p.shaft.eta_scale
        hi = float(np.clip(hi, 0.02, 0.6))
        return hi

    def _propeller_power(self, n_rps: float) -> float:
        omega = n_rps * 2.0 * math.pi
        return self.p.loss.k_prop * (omega**3)

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
        self._theta += 2.0 * math.pi * self._n * dt
        self._theta = self._theta % (2.0 * math.pi)
        Vs = self._Vs(self._theta)
        dVs_dt = self._dVs_dt(self._theta, self._n)
        Tm = self.p.phys.T0
        mass_bal = m_as + mf - m_ac
        d_mass_bal_dt = (mass_bal - self._mass_bal_prev) / dt
        self._mass_bal_prev = mass_bal

        dTs_dt = self._dTs(m_as, mf, m_ac, Tm, self._Ts, d_mass_bal_dt)
        self._Ts += dt * dTs_dt
        self._Ts = float(np.clip(self._Ts, 200.0, 800.0))

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


    def reset(self, *, n_rps: float | None = None, a: float | None = None) -> dict:
        self._t = 0

        self._n = float(n_rps) if n_rps is not None else self.p.op.n0_rps
        self._a = float(a) if a is not None else self.p.op.a0

        self._lambda = self.p.op.lambda0
        self._raw_lambda = self._lambda

        self._pm = self.p.phys.p0
        self._ps = self.p.phys.p0
        self._Ts = self.p.phys.T0
        self._theta = 0.0

        self._mass_bal_prev = 0.0
        self._delay = [0.0] * self.p.time.inj_delay_steps
        self._mff = 0.0
        self._mf_delay = [0.0] * self._mf_delay_steps
        self._stall_steps = 0

        self._lam_ref_filt = self._lambda_ref(self._a)

        self._last_m_air = 0.0
        self._last_mf = 0.0
        self._last_mfi_cmd = 0.0
        self._last_P_in = 0.0
        self._last_P_f = 0.0
        self._last_P_prop = 0.0
        self._last_T_eng = 0.0
        self._last_T_load = 0.0
        self._last_m_at = 0.0
        self._last_m_as = 0.0
        self._last_m_ac = 0.0

        return self.get_state()

    def step(self, mfi_cmd: float, a: float) -> dict:
        self._a = float(a)
        self._step_dynamics(mfi_cmd)
        self._t += 1
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "t": self._t * self.dt,
            "n_rps": self._n,
            "lambda": self._lambda,
            "pm_bar": self.p.pa_to_bar(self._pm),
            "ps_bar": self.p.pa_to_bar(self._ps),
            "Ts": self._Ts,
            "m_air": self._last_m_air,
            "mf": self._last_mf,
            "P_in": self._last_P_in,
            "P_f": self._last_P_f,
            "P_prop": self._last_P_prop,
        }
