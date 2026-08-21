"""Privileged-state labels for belief representation pretraining."""

from typing import NamedTuple

import jax.numpy as jnp
from generals.core.game import GameState, get_visibility


class BeliefTargets(NamedTuple):
    """Spatial labels aligned with a player's partial observation."""

    enemy_general: jnp.ndarray
    enemy_armies: jnp.ndarray
    enemy_ownership: jnp.ndarray
    hidden_mask: jnp.ndarray
    passable_mask: jnp.ndarray


def make_belief_targets(state: GameState, player_idx: int) -> BeliefTargets:
    """Build labels from simulator state that is hidden from ``player_idx``.

    The encoder still receives only the legal partial observation. Full state is
    used solely to construct offline supervision.
    """
    opponent_idx = 1 - player_idx
    visible = get_visibility(state.ownership[player_idx])
    enemy_ownership = state.ownership[opponent_idx]

    return BeliefTargets(
        enemy_general=state.generals & enemy_ownership,
        enemy_armies=jnp.where(enemy_ownership, state.armies, 0),
        enemy_ownership=enemy_ownership,
        hidden_mask=~visible,
        passable_mask=state.passable,
    )
