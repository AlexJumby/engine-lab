from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

@dataclass
class Physical:
    R: float = 287.05  # J/(kg·K)
    k: float = 1.4  # Cp/Cv
    Lth: float = 14.7
    p0: float = 101325.0  # Pa
    T0: float = 298.15  # K
    g: float = 9.80665  # m/s²


@dataclass
class Timing:
    dt: float = 0.01
    inj_delay_steps: int = 2
    episode_seconds: float = 10.0


@dataclass
class Geometry:
    Vm: float = 1.0e-3

    Vz: float = 6.0e-4

    D: float = 0.076
    stroke: float = 0.069

    @property
    def Vd_cyl(self) -> float:
        return math.pi * (self.D**2) * self.stroke / 4.0

    @property
    def Vd(self) -> float:
        return self.Vd_cyl

    i: int = 2
    t: int = 2


@dataclass
class Throttle:
    # alpha_t * A_th * sqrt(2*k/(R*T0)) и т.п. – её будем подбирать.
    mat1: float = 1.0
    a1: float = 1.4073
    a2: float = 0.4087
    p1: float = 0.4404
    p2: float = 2.3143
    pn: float = 0.7404
    pc: float = 0.4125
    scale: float = 1.5e-4


@dataclass
class PortFlow:
    use_hendricks: bool = False
    k_as: float = 1.5e-5

    n_unit: str = "krpm"
    pm_unit: str = "bar"
    c0: float = -0.366
    c1: float = 0.08979
    c2: float = -0.0337
    c3: float = 0.0001
    scale: float = 1.0


@dataclass
class FuelFilm:
    use_regressions: bool = True
    tau_ff_const: float = 0.6
    x_const: float = 0.4


@dataclass
class Shaft:
    J: float = 0.015
    Hu: float = 43e6
    kf: float = 0.05
    eta_scale: float = 0.55
    td: float = 0.02


@dataclass
class ShaftLoss:
    # Pf(kW) = n_krpm * (a0 + a1 n + a2 n^2) * Vd_L / (1.27 * fvp)
    a0: float = 1.673
    a1: float = 0.272
    a2: float = 0.0135
    fvp: float = 1.4

    k_prop: float = 5.5e-4


@dataclass
class Limits:
    n_min_rpm: float = 100.0
    n_max_rpm: float = 9000.0
    a_min: float = 0.0
    a_max: float = 1.0

    mfi_min: float = 0.6e-3
    mfi_idle: float = 0.8e-3
    mfi_max: float = 4.0e-3

    lambda_min: float = 0.6
    lambda_max: float = 1.6
    lambda_misfire: float = 1.7


@dataclass
class DomainRand:
    enable: bool = False
    J_pct: float = 0.2
    Hu_pct: float = 0.1
    Vm_pct: float = 0.2
    Vz_pct: float = 0.2
    kf_pct: float = 0.2


@dataclass
class Scavenging:
    eta_v0: float = 0.55
    n_ref_krpm: float = 3.0
    k_n: float = -0.05
    k_ps: float = 0.10
    eta_min: float = 0.3
    eta_max: float = 0.8


@dataclass
class Operating:
    n0_rps: float = 5000.0 / 60.0
    a0: float = 0.4
    lambda0: float = 1.0


@dataclass
class MVEMParams:
    phys: Physical = field(default_factory=Physical)
    time: Timing = field(default_factory=Timing)
    geo: Geometry = field(default_factory=Geometry)
    thr: Throttle = field(default_factory=Throttle)
    port: PortFlow = field(default_factory=PortFlow)
    film: FuelFilm = field(default_factory=FuelFilm)
    shaft: Shaft = field(default_factory=Shaft)
    loss: ShaftLoss = field(default_factory=ShaftLoss)
    lim: Limits = field(default_factory=Limits)
    drand: DomainRand = field(default_factory=DomainRand)
    op: Operating = field(default_factory=Operating)
    scav: Scavenging = field(default_factory=Scavenging)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def rpm_to_rps(n_rpm: float) -> float:
        return n_rpm / 60.0

    @staticmethod
    def rps_to_rpm(n_rps: float) -> float:
        return n_rps * 60.0

    @staticmethod
    def pa_to_bar(p_pa: float) -> float:
        return p_pa / 1e5

    @staticmethod
    def bar_to_pa(p_bar: float) -> float:
        return p_bar * 1e5

    def hendricks_inputs(self, n_rps: float, pm_pa: float) -> tuple[float, float]:
        n_rpm = self.rps_to_rpm(n_rps)
        pm_bar = self.pa_to_bar(pm_pa)
        if self.port.n_unit == "krpm":
            n = n_rpm / 1000.0
        else:
            n = n_rpm
        return n, pm_bar
