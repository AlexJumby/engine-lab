import numpy as np
import matplotlib.pyplot as plt
from engine_lab.models.hirth3203 import Hirth3203Engine

eng = Hirth3203Engine()
state = eng.reset()

ts, n_list, lam_list = [], [], []

for k in range(1000):
    t = k * eng.dt
    if t < 2.0:
        a = 0.35
    else:
        a = 0.45
    mfi_cmd = 0.001  # просто константа для теста

    state = eng.step(mfi_cmd, a)

    ts.append(state["t"])
    n_list.append(state["n_rps"] * 60.0)
    lam_list.append(state["lambda"])

plt.figure()
plt.subplot(2,1,1); plt.plot(ts, n_list); plt.ylabel("n, rpm")
plt.subplot(2,1,2); plt.plot(ts, lam_list); plt.ylabel("lambda"); plt.xlabel("t, s")
plt.show()
