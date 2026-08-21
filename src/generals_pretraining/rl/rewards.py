"""Reward function for PPO training.

Signature matches the rollout call site (extra args unused, kept for that interface):
    win_lose_reward(prior_obs, action, next_obs, winners, truncated=False, gamma=1.0) -> (N,)
"""

import jax.numpy as jnp


def win_lose_reward(prior_obs, action, next_obs, winners, truncated=False, gamma=1.0):
    """+1 if we won (winner == 0), -1 if we lost (winner == 1), else 0."""
    return jnp.where(winners == 0, 1.0, jnp.where(winners == 1, -1.0, 0.0))
