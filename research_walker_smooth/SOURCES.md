# Method attribution

Mysore, S., Mabsout, B., Mancuso, R., and Saenko, K. (2021). **Regularizing Action Policies for Smooth Control with Reinforcement Learning.** ICRA. Author project page: https://ai.bu.edu/caps/

The page states temporal and spatial policy smoothness terms. Our experiment uses only a temporal auxiliary loss on consecutive rollout observations, the squared Euclidean norm averaged over actuators, and clipped action means. It does not implement the complete CAPS algorithm and does not reproduce its published results. The current experiment tests MG as an observer, not novelty of an RL regularizer.

Stable-Baselines3 PPO 2.7.1, installed source `PPO.train`, MIT license in `SB3_LICENSE.txt`. The original train method hash and patched method hash are recorded in `ppo_source.json`. This experiment adds only the declared auxiliary temporal term and pair sampling before rollout flattening. The zero-coefficient regression compares it with the installed unmodified PPO.

Gymnasium Walker2d-v5 and MuJoCo versions are pinned in `requirements.txt`. The implementation and reported numerical results are in this archive; no empirical outcomes are copied from external publications.
