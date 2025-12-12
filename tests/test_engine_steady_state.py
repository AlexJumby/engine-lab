from statistics import mean

from engine_lab.models.hirth3203.mvem import Hirth3203Engine
from engine_lab.models.hirth3203.params import MVEMParams


def simulate_and_collect(eng: Hirth3203Engine, mfi_cmd: float, a: float, t_final: float):
    steps = int(t_final / eng.dt)
    t = []
    n = []
    lam = []

    state = eng.reset(n_rps=eng.p.op.n0_rps, a=a)
    for _ in range(steps):
        state = eng.step(mfi_cmd=mfi_cmd, a=a)
        t.append(state["t"])
        n.append(state["n_rps"])
        lam.append(state["lambda"])
    return t, n, lam


def window_mean_span(values, frac_last=0.3):
    k0 = int(len(values) * (1.0 - frac_last))
    tail = values[k0:]
    v_mean = mean(tail)
    v_min = min(tail)
    v_max = max(tail)
    span = v_max - v_min
    return v_mean, span


def test_idle_like_steady_state():
    p = MVEMParams()
    eng = Hirth3203Engine(p)

    a = 0.4
    mfi = p.lim.mfi_idle

    t, n, lam = simulate_and_collect(eng, mfi_cmd=mfi, a=a, t_final=10.0)

    n_mean, n_span = window_mean_span(n, frac_last=0.3)
    lam_mean, lam_span = window_mean_span(lam, frac_last=0.3)

    # требования "почти стационарности" — можно потом подрегулировать
    assert n_mean > 0.0
    assert n_span < 0.05 * n_mean          # ±5% по оборотам
    assert lam_span < 0.05 * max(lam_mean, 1e-3)  # ±5% по λ


def test_richer_mixture_gives_higher_steady_speed():
    p = MVEMParams()
    eng = Hirth3203Engine(p)

    a = 0.4
    mfi1 = p.lim.mfi_idle * 0.9
    mfi2 = p.lim.mfi_idle * 1.3

    _, n1, _ = simulate_and_collect(eng, mfi_cmd=mfi1, a=a, t_final=8.0)
    _, n2, _ = simulate_and_collect(eng, mfi_cmd=mfi2, a=a, t_final=8.0)

    n1_mean, _ = window_mean_span(n1, frac_last=0.3)
    n2_mean, _ = window_mean_span(n2, frac_last=0.3)

    assert n2_mean > n1_mean
