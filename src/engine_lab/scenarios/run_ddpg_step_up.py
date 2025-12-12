import numpy as np
import matplotlib.pyplot as plt

from engine_lab.models.hirth3203.mvem import Hirth3203Engine
from engine_lab.controllers.ddpg.ddpg_afr import DDPGLambdaController


def main():
    # 1. Двигатель
    eng = Hirth3203Engine()
    state = eng.reset()

    # 2. Контроллер DDPG (путь поменяй на свой .zip)
    ctrl = DDPGLambdaController(
        model_path=r"src\engine_lab\controllers\ddpg\ddpg_lambda_finetuned_v1_4.zip",
        vecnorm_path=r"src\engine_lab\controllers\ddpg\vecnorm_lambda_finetuned_v1_4.pkl",
        device="cpu",
    )
    ctrl.reset()

    # 3. Сценарий по дросселю: 0.34 -> 0.46
    t_final = 10.0
    dt = eng.dt
    steps = int(t_final / dt)

    a1 = 0.34
    a2 = 0.46
    step_index = steps // 2

    ts: list[float] = []
    lam_ref_list: list[float] = []
    lam_list: list[float] = []
    n_list: list[float] = []
    u_list: list[float] = []

    for k in range(steps):
        t = k * dt
        a = a1 if k < step_index else a2

        mfi_cmd, info = ctrl.compute(t=t, state=state, a=a, engine=eng)

        state = eng.step(mfi_cmd=mfi_cmd, a=a)

        ts.append(state["t"])
        lam_list.append(state["lambda"])
        n_list.append(state["n_rps"] * 60.0)
        lam_ref_list.append(info["lambda_ref"])
        u_list.append(info["u"])

    # 4. Графики
    ts = np.array(ts)

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(ts, lam_ref_list, "k--", label="λ_ref")
    plt.plot(ts, lam_list, label="λ_DDPG")
    plt.ylabel("λ")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(ts, u_list)
    plt.ylabel("u")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(ts, n_list)
    plt.ylabel("n, rpm")
    plt.xlabel("t, s")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
