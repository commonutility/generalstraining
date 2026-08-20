import jax
import jax.numpy as jnp
from generals.core.env import GeneralsEnv

from config import Config
from train.ppo import device_put_replicated, get_city_range, set_city_range


def test_device_put_replicated_adds_pmap_axis():
    tree = {"weights": jnp.arange(3), "bias": jnp.array(2.0)}
    devices = jax.devices()

    replicated = device_put_replicated(tree, devices)

    assert replicated["weights"].shape == (len(devices), 3)
    assert replicated["bias"].shape == (len(devices),)
    assert jnp.array_equal(replicated["weights"][0], tree["weights"])
    assert replicated["bias"][0] == tree["bias"]


def test_city_range_compatibility_uses_current_simulator_field():
    env = GeneralsEnv(grid_dims=(4, 4), num_cities_range=(1, 2))

    assert get_city_range(env) == (1, 2)
    set_city_range(env, (2, 3))
    assert env.num_castles_range == (2, 3)


def test_cpu_smoke_config_keeps_gae90_ppo_invariants():
    released = Config.from_yaml("configs/custom/L_7d_gae90.yaml")
    smoke = Config.from_yaml("configs/smoke/L_7d_gae90_cpu.yaml")

    unchanged_fields = [
        "network",
        "depth",
        "embed_dim",
        "n_head",
        "ff_factor",
        "patch_size",
        "conv_dim",
        "use_bf16",
        "gamma",
        "gae_lambda",
        "clip_eps",
        "max_grad_norm",
        "vf_coef",
        "target_kl",
        "adv_top_frac",
        "ent_schedule",
        "ent_coef_start",
        "ent_power",
        "ent_coef_min",
        "ent_coef_end",
        "ent_coef_decay_iters",
        "lr",
        "lr_schedule",
        "lr_power_law_numerator",
        "lr_power_law_exponent",
        "lr_power_law_min",
        "lr_power_law_max",
        "value_loss",
        "num_bins",
        "v_min",
        "v_max",
        "hl_sigma",
        "num_epochs",
    ]
    for field in unchanged_fields:
        assert getattr(smoke, field) == getattr(released, field)

    per_device_total = smoke.num_steps * 2 * smoke.num_envs
    n_keep = int(per_device_total * smoke.adv_top_frac)
    n_keep = (n_keep // smoke.minibatch_size) * smoke.minibatch_size

    assert smoke.init_checkpoint == ""
    assert smoke.init_encoder_checkpoint == ""
    assert smoke.ref_eval_every == 0
    assert smoke.num_iters == 2
    assert smoke.eval_games == 2
    assert n_keep >= smoke.minibatch_size
    map_size_combinations = (smoke.max_grid_size - smoke.min_grid_size + 1) ** 2
    assert smoke.pool_size >= map_size_combinations
