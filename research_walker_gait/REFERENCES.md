# Implementation references checked 2026-09-29

Gymnasium, Walker2d-v5 environment documentation: observation/action definition, reward, health criteria, time step and episode length.
```
https://gymnasium.farama.org/environments/mujoco/walker2d/
```

Stable-Baselines3, PPO documentation; this experiment pins release 2.7.1, rather than the changing master documentation.
```
https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html
```

MuJoCo state specification and integration-state restoration. Numerical replay is also verified empirically in this experiment.
```
https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
```

MG implementation is included unchanged under code/actdim in the result archive. This experiment does not prove a new delay-embedding theorem or transfer smooth-system assumptions to contact dynamics.
