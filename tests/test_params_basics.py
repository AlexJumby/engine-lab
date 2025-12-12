from engine_lab.models.hirth3203.params import MVEMParams


def test_geometry_volumes_positive():
    p = MVEMParams()
    assert p.geo.Vm > 0.0
    assert p.geo.Vz > 0.0
    assert p.geo.Vd_cyl > 0.0
    assert p.geo.Vd > 0.0


def test_limits_consistent():
    p = MVEMParams()
    assert p.lim.n_min_rpm < p.lim.n_max_rpm
    assert p.lim.a_min <= p.op.a0 <= p.lim.a_max
    assert p.lim.lambda_min < p.lim.lambda_max
    assert p.lim.mfi_min < p.lim.mfi_max
    assert p.lim.mfi_idle >= p.lim.mfi_min


def test_unit_conversions_roundtrip():
    p = MVEMParams()
    p_bar = 1.23
    p_pa = p.bar_to_pa(p_bar)
    assert abs(p.pa_to_bar(p_pa) - p_bar) < 1e-6

    n_rpm = 5234.0
    n_rps = p.rpm_to_rps(n_rpm)
    assert abs(p.rps_to_rpm(n_rps) - n_rpm) < 1e-6
