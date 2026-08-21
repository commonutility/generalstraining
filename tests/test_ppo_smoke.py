import jax
import jax.numpy as jnp
from generals_pretraining.envs import GeneralsEnv
from generals_pretraining.config import Config
from generals_pretraining.rl.ppo import device_put_replicated, get_city_range, set_city_range, should_save_checkpoint


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


def test_averagejoe_8k_uses_exact_checkpoint_milestones():
    cfg = Config.from_yaml("configs/experiments/L_7d_gae90_8k.yaml")

    assert cfg.num_iters == 8000
    assert cfg.ckpt_every == 0
    assert cfg.save_every == 0
    assert cfg.save_at == [1000, 2000, 4000, 8000]

    saved = [
        iteration
        for iteration in range(1, cfg.num_iters + 1)
        if should_save_checkpoint(iteration, cfg.save_every, cfg.save_at)
    ]
    assert saved == cfg.save_at

    env_interactions = cfg.num_iters * cfg.num_envs * cfg.num_steps
    assert env_interactions == 2_097_152_000
    assert 2 * env_interactions == 4_194_304_000


def test_periodic_and_exact_checkpoint_schedules_are_additive():
    saved = [
        iteration
        for iteration in range(1, 9)
        if should_save_checkpoint(iteration, every=3, save_at=[2, 8])
    ]

    assert saved == [2, 3, 6, 8]
