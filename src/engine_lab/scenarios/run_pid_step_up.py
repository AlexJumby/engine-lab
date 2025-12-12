import matplotlib.pyplot as plt

from engine_lab.models.hirth3203.mvem import Hirth3203Engine
from engine_lab.controllers.pid.pid_afr import PIDAFRController, PIDGains


def main():
    eng = Hirth3203Engine()
    gains = PIDGains(
        kp=4.0e-3,
        ki=2.0e-2,
        kd=0.0,
        i_min=-1.0e-3,
        i_max=1.0e-3,
    )
    ctrl = PIDAFRController(gains, params=eng.p)

    state = eng.reset()

    t_list, lam_list, lam_ref_list = [], [], []
    mfi_list, n_rpm_list, a_list = [], [], []

    t_final = 10.0
    steps = int(t_final / eng.dt)

    for k in range(steps):
        t = k * eng.dt

        # сценарий: ступенька по дросселю
        if t < 5.0:
            a = 0.34
        else:
            a = 0.46

        mfi_cmd, info = ctrl.compute(t=t, state=state, a=a, engine=eng)
        state = eng.step(mfi_cmd, a)

        t_list.append(t)
        lam_list.append(state["lambda"])
        lam_ref_list.append(info["lambda_ref"])
        mfi_list.append(mfi_cmd)
        n_rpm_list.append(state["n_rps"] * 60.0)
        a_list.append(a)

    # --- графики ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    ax1.plot(t_list, lam_list, label="λ")
    ax1.plot(t_list, lam_ref_list, "--", label="λ_ref")
    ax1.set_ylabel("λ")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t_list, mfi_list)
    ax2.set_ylabel("mfi_cmd, kg/s")
    ax2.grid(True)

    ax3.plot(t_list, n_rpm_list)
    ax3.set_ylabel("n, rpm")
    ax3.set_xlabel("t, s")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
