import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from generals.core.game import create_initial_state, get_observation

from networks import load_pretrained_encoder, obs_to_array
from networks.common import augment_obs, init_obs_state
from networks.transformer import HistoryTransformer
from pretraining.model import BeliefPretrainer, belief_loss
from pretraining.targets import make_belief_targets


def example_state():
    grid = jnp.array(
        [
            [1, 0, -2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
        ],
        dtype=jnp.int32,
    )
    return create_initial_state(grid)


def test_belief_targets_use_privileged_enemy_state():
    targets = make_belief_targets(example_state(), 0)

    assert targets.enemy_general[3, 3]
    assert targets.enemy_ownership[3, 3]
    assert targets.enemy_armies[3, 3] == 1
    assert targets.hidden_mask[3, 3]
    assert not targets.passable_mask[0, 2]


def test_belief_model_matches_board_shape_and_has_finite_loss():
    state = example_state()
    observation = get_observation(state, 0)
    augmented, obs_state = augment_obs(obs_to_array(observation), init_obs_state(4))
    temporal = jnp.stack([obs_state.opponent_army_history, obs_state.opponent_land_history])
    encoder = HistoryTransformer(
        grid_size=4,
        pad_to=4,
        patch_size=2,
        depth=1,
        embed_dim=16,
        n_head=4,
        ff_factor=2,
        key=jrandom.PRNGKey(0),
    )
    model = BeliefPretrainer(encoder, key=jrandom.PRNGKey(1))

    predictions = model(augmented, temporal)
    loss, metrics = belief_loss(model, augmented, temporal, make_belief_targets(state, 0))

    assert predictions["enemy_general_logits"].shape == (4, 4)
    assert predictions["enemy_army"].shape == (4, 4)
    assert predictions["enemy_ownership_logits"].shape == (4, 4)
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(value) for value in metrics.values())


def test_encoder_transfer_preserves_fresh_ppo_heads(tmp_path):
    kwargs = dict(
        grid_size=4,
        pad_to=4,
        patch_size=2,
        depth=1,
        embed_dim=16,
        n_head=4,
        ff_factor=2,
    )
    pretrained = HistoryTransformer(**kwargs, key=jrandom.PRNGKey(2))
    fresh = HistoryTransformer(**kwargs, key=jrandom.PRNGKey(3))
    checkpoint = tmp_path / "encoder.eqx"
    eqx.tree_serialise_leaves(checkpoint, pretrained)

    transferred = load_pretrained_encoder(checkpoint, fresh)

    assert jnp.array_equal(transferred.embedder.weight, pretrained.embedder.weight)
    assert jnp.array_equal(transferred.policy_head.weight, fresh.policy_head.weight)
    assert jnp.array_equal(transferred.value_head.weight, fresh.value_head.weight)
