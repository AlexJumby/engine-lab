from engine_lab.models.hirth3203.mvem import Hirth3203Engine
from engine_lab.models.hirth3203.params import MVEMParams


def test_engine_reset_default():
    eng = Hirth3203Engine(MVEMParams())
    state = eng.get_state()

    assert state["t"] == 0.0
    assert state["n_rps"] > 0.0
    assert 0.5 <= state["lambda"] <= 2.0
    assert 0.5 <= state["pm_bar"] <= 1.5
    assert 0.5 <= state["ps_bar"] <= 1.5
    assert 200.0 <= state["Ts"] <= 800.0


def test_engine_reset_custom_operating_point():
    p = MVEMParams()
    eng = Hirth3203Engine(p)

    n_rps = 4000.0 / 60.0
    a = 0.45
    state = eng.reset(n_rps=n_rps, a=a)

    assert abs(state["n_rps"] - n_rps) < 1e-9
    # просто проверяем, что reset принял a,
    # а динамика уже дальше начнёт двигать смесь
    assert eng._a == a
