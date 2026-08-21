"""Belief-prediction heads attached to the unchanged AverageJoe torso."""

import equinox as eqx
import jax
import jax.numpy as jnp

from generals_pretraining.models.transformer import HistoryTransformer
from generals_pretraining.pretraining.targets import BeliefTargets


class BeliefPretrainer(eqx.Module):
    """Predict privileged opponent state from legal partial observations."""

    encoder: HistoryTransformer
    general_head: eqx.nn.Linear
    army_head: eqx.nn.Linear
    ownership_head: eqx.nn.Linear

    def __init__(self, encoder: HistoryTransformer, *, key):
        embed_dim = encoder.pos_encoding.shape[-1]
        patch_area = encoder.patch_size**2
        k_general, k_army, k_ownership = jax.random.split(key, 3)
        self.encoder = encoder
        self.general_head = eqx.nn.Linear(embed_dim, patch_area, key=k_general)
        self.army_head = eqx.nn.Linear(embed_dim, patch_area, key=k_army)
        self.ownership_head = eqx.nn.Linear(embed_dim, patch_area, key=k_ownership)

    def _unpatchify(self, values):
        grid = self.encoder.pad_to // self.encoder.patch_size
        patch = self.encoder.patch_size
        return (
            values.reshape(grid, grid, patch, patch)
            .transpose(0, 2, 1, 3)
            .reshape(self.encoder.pad_to, self.encoder.pad_to)
        )

    def __call__(self, obs, temporal):
        patch_tokens = self.encoder.encode(obs, temporal)[3:]
        return {
            "enemy_general_logits": self._unpatchify(jax.vmap(self.general_head)(patch_tokens)),
            "enemy_army": self._unpatchify(jax.vmap(self.army_head)(patch_tokens)),
            "enemy_ownership_logits": self._unpatchify(jax.vmap(self.ownership_head)(patch_tokens)),
        }


def belief_loss(model: BeliefPretrainer, obs, temporal, targets: BeliefTargets):
    """Weighted single-example loss and diagnostics."""
    pred = model(obs, temporal)
    passable = targets.passable_mask.astype(jnp.float32)
    hidden = (targets.hidden_mask & targets.passable_mask).astype(jnp.float32)
    hidden_count = jnp.maximum(hidden.sum(), 1.0)

    general_logits = jnp.where(targets.passable_mask, pred["enemy_general_logits"], -1e9)
    general_loss = -jnp.sum(targets.enemy_general.reshape(-1) * jax.nn.log_softmax(general_logits.reshape(-1)))

    army_target = jnp.log1p(targets.enemy_armies.astype(jnp.float32)) / jnp.log(501.0)
    army_loss = jnp.sum(hidden * (pred["enemy_army"] - army_target) ** 2) / hidden_count

    ownership_loss_map = (
        jnp.maximum(pred["enemy_ownership_logits"], 0)
        - (pred["enemy_ownership_logits"] * targets.enemy_ownership)
        + jnp.log1p(jnp.exp(-jnp.abs(pred["enemy_ownership_logits"])))
    )
    ownership_loss = jnp.sum(hidden * ownership_loss_map) / hidden_count

    total = general_loss + army_loss + ownership_loss
    return total, {
        "loss": total,
        "general_loss": general_loss,
        "army_loss": army_loss,
        "ownership_loss": ownership_loss,
        "hidden_fraction": hidden.sum() / jnp.maximum(passable.sum(), 1.0),
    }
