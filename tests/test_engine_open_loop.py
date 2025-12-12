import math

from engine_lab.models.hirth3203.mvem import Hirth3203Engine
from engine_lab.models.hirth3203.params import MVEMParams


def run_to_time(eng: Hirth3203Engine, mfi_cmd: float, a: float, t_final: float):
    steps = int(t_final / eng.dt)
    state = eng.reset(n_rps=eng.p.op.n0_rps, a=a)
    for _ in range(steps):
        state = eng.step(mfi_cmd=mfi_cmd, a=a)
    return state


def test_increase_fuel_increases_speed():
    p = MVEMParams()
    eng = Hirth3203Engine(p)

    a = 0.4
    idle = p.lim.mfi_idle
    rich = p.lim.mfi_idle * 1.5

    state_idle = run_to_time(eng, mfi_cmd=idle, a=a, t_final=5.0)
    state_rich = run_to_time(eng, mfi_cmd=rich, a=a, t_final=5.0)

    assert state_rich["n_rps"] > state_idle["n_rps"]


def test_more_throttle_changes_airflow_and_lambda_sensibly():
    p = MVEMParams()
    eng = Hirth3203Engine(p)

    mfi = p.lim.mfi_idle * 1.2

    state_low = run_to_time(eng, mfi_cmd=mfi, a=0.35, t_final=5.0)
    state_high = run_to_time(eng, mfi_cmd=mfi, a=0.5,  t_final=5.0)

    m_air_low = state_low["m_air"]
    m_air_high = state_high["m_air"]
    lam_low = state_low["lambda"]
    lam_high = state_high["lambda"]
    n_low = state_low["n_rps"]
    n_high = state_high["n_rps"]

    # 1) Воздух не должен отличаться радикально (разница < 2%)
    rel_diff_air = abs(m_air_high - m_air_low) / m_air_low
    assert rel_diff_air < 0.02

    # 2) При том же топливе с большим дросселем смесь должна быть беднее
    assert lam_high > lam_low

    # 3) Обороты не должны сильно просесть (никаких “завалов”)
    rel_drop_n = (n_low - n_high) / n_low
    assert rel_drop_n < 0.02
