"""Offline trajectory collection with privileged simulator labels."""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from generals.core.action import compute_valid_move_mask
from generals.core.game import get_observation

from generals_pretraining.models import obs_to_array, random_action, reset_done_envs
from generals_pretraining.models.common import augment_obs, init_obs_state
from generals_pretraining.pretraining.targets import make_belief_targets


def collect_belief_dataset(env, pool, *, num_envs: int, num_steps: int, seed: int):
    """Collect random-policy trajectories for both seats in parallel."""
    key = jrandom.PRNGKey(seed)
    key, state_key = jrandom.split(key)
    states = jax.vmap(env.init_state)(jrandom.split(state_key, num_envs))

    single_obs_state = init_obs_state(env.pad_to, env.pad_to)
    obs_state = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (2 * num_envs, *x.shape)),
        single_obs_state,
    )

    def concatenate_players(a, b):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y]), a, b)

    def scan_step(carry, _):
        states, obs_state, key, state_pool = carry
        obs_p0 = jax.vmap(lambda state: get_observation(state, 0))(states)
        obs_p1 = jax.vmap(lambda state: get_observation(state, 1))(states)
        observations = concatenate_players(obs_p0, obs_p1)

        obs_arrays = jax.vmap(obs_to_array)(observations)
        augmented, next_obs_state = jax.vmap(augment_obs)(obs_arrays, obs_state)
        temporal = jnp.stack(
            [next_obs_state.opponent_army_history, next_obs_state.opponent_land_history],
            axis=1,
        )

        targets_p0 = jax.vmap(lambda state: make_belief_targets(state, 0))(states)
        targets_p1 = jax.vmap(lambda state: make_belief_targets(state, 1))(states)
        targets = concatenate_players(targets_p0, targets_p1)

        key, action_key = jrandom.split(key)
        action_keys = jrandom.split(action_key, 2 * num_envs)
        actions = jax.vmap(random_action)(action_keys, observations)
        env_actions = jnp.stack([actions[:num_envs], actions[num_envs:]], axis=1)
        timesteps, new_states = jax.vmap(lambda state, action: env.step(state, action, state_pool))(states, env_actions)
        dones = timesteps.terminated | timesteps.truncated
        next_obs_state = reset_done_envs(next_obs_state, jnp.concatenate([dones, dones]))

        valid_masks = jax.vmap(lambda obs: compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains))(
            observations
        )
        sample = (augmented, temporal, valid_masks, targets)
        return (new_states, next_obs_state, key, state_pool), sample

    collect = jax.jit(lambda carry: jax.lax.scan(scan_step, carry, None, length=num_steps))
    _, samples = collect((states, obs_state, key, pool))
    return samples


def save_belief_dataset(path: str | Path, samples) -> None:
    """Write a compressed, flat sample table consumable by ``train_belief.py``."""
    observations, temporal, valid_masks, targets = samples

    def flatten(value):
        return np.asarray(value).reshape(-1, *value.shape[2:])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=flatten(observations).astype(np.float16),
        temporal=flatten(temporal).astype(np.float16),
        valid_masks=flatten(valid_masks).astype(bool),
        enemy_general=flatten(targets.enemy_general).astype(bool),
        enemy_armies=flatten(targets.enemy_armies).astype(np.int32),
        enemy_ownership=flatten(targets.enemy_ownership).astype(bool),
        hidden_mask=flatten(targets.hidden_mask).astype(bool),
        passable_mask=flatten(targets.passable_mask).astype(bool),
    )
