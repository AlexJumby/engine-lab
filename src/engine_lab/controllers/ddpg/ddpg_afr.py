import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from engine_lab.models.hirth3203.mvem import Hirth3203Engine
from engine_lab.controllers.base import BaseLambdaController
from engine_lab.models.hirth3203.params import MVEMParams


class DDPGLambdaController(BaseLambdaController):
    def __init__(self, params: MVEMParams, model_path: str | None = None, vecnorm_path: str | None = None, device: str | None = "cpu"):
        self.p = params           

        if model_path is None:
            from importlib.resources import files
            model_path = str(
                files("engine_lab.controllers.ddpg")
                .joinpath("ddpg_lambda_finetuned_v1_4.zip")
            )

        if vecnorm_path is None:
            from importlib.resources import files
            vecnorm_path = str(
                files("engine_lab.controllers.ddpg")
                .joinpath("vecnorm_lambda_finetuned_v1_4.pkl")
            )


        self.model = DDPG.load(model_path, device=device)

        self.vecnorm = None
        if vecnorm_path is not None:
            # создаём фиктивное окружение с тем же пространством obs
            def _make_env():
                # мини-обёртка, которая только ради spaces
                params = MVEMParams()
                from .env.mvem_2t import MVEM2TEnv  # если скопируешь env сюда – откуда нужно импортнёшь
                return MVEM2TEnv(params=params)

            dummy_env = DummyVecEnv([_make_env])
            self.vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False

    def reset(self) -> None:
        # если нужно, можно обнулить внутренние состояния
        pass

    def _obs_from_state(self, engine: Hirth3203Engine, state: dict, a: float):
        lam = state["lambda"]
        n_rps = state["n_rps"]
        pm_bar = state["pm_bar"]

        lam_ref = engine._lambda_ref(a)
        err = lam - lam_ref

        obs = np.array([n_rps, lam, a, pm_bar, err, lam_ref], dtype=np.float32)
        return obs

    def compute(self, *, t: float, state: dict, a: float, engine: Hirth3203Engine):
        obs = self._obs_from_state(engine, state, a)

        if self.vecnorm is not None:
            # VecNormalize ожидает батч, поэтому добавляем ось [None, :]
            obs_in = self.vecnorm.normalize_obs(obs[None, :])[0]
        else:
            obs_in = obs

        action, _ = self.model.predict(obs_in, deterministic=True)
        u = float(np.clip(action[0], -1.0, 1.0))

        p = engine.p
        alpha = 0.5 * (u + 1.0)
        mfi_cmd = p.lim.mfi_idle + alpha * (p.lim.mfi_max - p.lim.mfi_idle)

        lam_ref = engine._lambda_ref(a)
        err = state["lambda"] - lam_ref

        info = {"u": u, "lambda_ref": lam_ref, "err": err}
        return mfi_cmd, info
