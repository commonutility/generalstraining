"""Round-robin tournament between training milestone checkpoints.

Answers "is the policy still improving with more iterations?" by playing EMA
milestones from each seed head-to-head on the final-curriculum-stage evaluation
distribution. The default set is iterations 1,000, 2,000, and 4,000.

Usage:
    python analysis/run_milestone_tournament.py \
        --ckpt-dir artifacts/tournament \
        --num-games 64 \
        --out analysis/metrics/milestone_tournament.json
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import jax.random as jrandom

from generals.core.env import GeneralsEnv

from generals_pretraining.evaluation.agent import Agent
from generals_pretraining.evaluation.matchup import play_match, compute_elo

# S3 experiment + run names holding each EMA milestone. Iterations 1,000 and
# 2,000 came from the original jobs; iteration 4,000 came from the resumed jobs.
S3_MILESTONES = {
    (44, 1000): (
        "exp3seeds_scratch_seed44-20260821T174537Z-12b60918",
        "exp3seeds_scratch_seed44",
    ),
    (44, 2000): (
        "exp3seeds_scratch_seed44-20260821T174537Z-12b60918",
        "exp3seeds_scratch_seed44",
    ),
    (44, 4000): (
        "exp8k_resume2k_v2_scratch_seed44-20260822T184203Z-d0e992f8",
        "exp8k_resume2k_v2_scratch_seed44",
    ),
    (45, 1000): (
        "exp3seeds_scratch_seed45-20260821T174538Z-6a4e2473",
        "exp3seeds_scratch_seed45",
    ),
    (45, 2000): (
        "exp3seeds_scratch_seed45-20260821T174538Z-6a4e2473",
        "exp3seeds_scratch_seed45",
    ),
    (45, 4000): (
        "exp8k_resume2k_v2_scratch_seed45-20260822T184205Z-63679248",
        "exp8k_resume2k_v2_scratch_seed45",
    ),
    (46, 1000): (
        "exp3seeds_scratch_seed46-20260821T181509Z-16c2bb72",
        "exp3seeds_scratch_seed46",
    ),
    (46, 2000): (
        "exp3seeds_scratch_seed46-20260821T181509Z-16c2bb72",
        "exp3seeds_scratch_seed46",
    ),
    (46, 4000): (
        "exp8k_resume2k_v2_scratch_seed46-20260822T184207Z-028f922a",
        "exp8k_resume2k_v2_scratch_seed46",
    ),
}


def fetch_from_s3(bucket, seeds, iters, ckpt_dir):
    import boto3

    s3 = boto3.client("s3")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        config_experiment, config_run = S3_MILESTONES[(seed, iters[0])]
        targets = [
            (
                f"generals/experiments/{config_experiment}/checkpoints/"
                f"{config_run}/config.yaml",
                f"seed{seed}_config.yaml",
            )
        ]
        for it in iters:
            try:
                experiment, run = S3_MILESTONES[(seed, it)]
            except KeyError as error:
                raise ValueError(f"no S3 checkpoint configured for seed {seed} at {it}") from error
            targets.append(
                (
                    f"generals/experiments/{experiment}/checkpoints/{run}/"
                    f"{run}_ema_{it}.eqx",
                    f"seed{seed}_ema_{it}.eqx",
                )
            )
        for remote, local in targets:
            dest = ckpt_dir / local
            if dest.exists():
                continue
            print(f"downloading s3://{bucket}/{remote}", flush=True)
            s3.download_file(bucket, remote, str(dest))


def play_match_tiebreak(agent_a, agent_b, env, pool, num_games, truncation, key):
    """play_match variant that scores truncation draws by land, then army.

    Returns (wins_a, wins_b, draws, real_wins_a, real_wins_b) where real_*
    counts general captures only and wins_* additionally includes tiebreaks.
    """
    import jax
    import jax.numpy as jnp

    from generals.core.game import get_observation
    from generals.core.action import compute_valid_move_mask
    from generals_pretraining.models import obs_to_array, reset_done_envs

    assert agent_a.pad_to == agent_b.pad_to
    pad_to = agent_a.pad_to

    def keyless(fn):
        return lambda net, obs, mask, temporal, key: fn(net, obs, mask, temporal)

    action_fn_a = keyless(agent_a.greedy_fn)
    action_fn_b = keyless(agent_b.greedy_fn)

    single_state = agent_a.init_obs_state_fn(pad_to, pad_to)
    batched_state = jax.tree.map(
        lambda x: jnp.repeat(x[None], num_games, axis=0), single_state
    )
    step_fn = (lambda s, a: env.step(s, a, pool)) if pool is not None else env.step

    @jax.jit
    def _play(net_a, net_b, key):
        key, *init_keys = jrandom.split(key, num_games + 1)
        states = jax.vmap(env.init_state)(jnp.stack(init_keys))

        counters = jnp.zeros(5, dtype=jnp.int32)  # wins_a, wins_b, draws, real_a, real_b
        finished = jnp.zeros(num_games, dtype=jnp.bool_)

        def scan_body(carry, _):
            states, key, finished, counters, obs_state_a, obs_state_b = carry

            key, key_a, key_b = jrandom.split(key, 3)
            keys_a = jrandom.split(key_a, num_games)
            keys_b = jrandom.split(key_b, num_games)

            obs_p0 = jax.vmap(lambda s: get_observation(s, 0))(states)
            obs_p1 = jax.vmap(lambda s: get_observation(s, 1))(states)

            obs_arr_a = jax.vmap(obs_to_array)(obs_p0)
            masks_a = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0)
            obs_aug_a, obs_state_a = jax.vmap(agent_a.augment_fn)(obs_arr_a, obs_state_a)
            temporal_a = jnp.stack([obs_state_a.opponent_army_history, obs_state_a.opponent_land_history], axis=1)
            action_a = jax.vmap(action_fn_a, in_axes=(None, 0, 0, 0, 0))(net_a, obs_aug_a, masks_a, temporal_a, keys_a)

            obs_arr_b = jax.vmap(obs_to_array)(obs_p1)
            masks_b = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p1)
            obs_aug_b, obs_state_b = jax.vmap(agent_b.augment_fn)(obs_arr_b, obs_state_b)
            temporal_b = jnp.stack([obs_state_b.opponent_army_history, obs_state_b.opponent_land_history], axis=1)
            action_b = jax.vmap(action_fn_b, in_axes=(None, 0, 0, 0, 0))(net_b, obs_aug_b, masks_b, temporal_b, keys_b)

            actions = jnp.stack([action_a, action_b], axis=1)
            timesteps, new_states = jax.vmap(step_fn)(states, actions)

            dones = timesteps.terminated | timesteps.truncated
            new_done = dones & ~finished
            real_a = timesteps.info.winner == 0
            real_b = timesteps.info.winner == 1
            trunc = timesteps.truncated & ~timesteps.terminated

            land = timesteps.info.land
            army = timesteps.info.army
            a_ahead = (land[:, 0] > land[:, 1]) | ((land[:, 0] == land[:, 1]) & (army[:, 0] > army[:, 1]))
            b_ahead = (land[:, 1] > land[:, 0]) | ((land[:, 0] == land[:, 1]) & (army[:, 1] > army[:, 0]))

            counters = counters + jnp.array([
                jnp.sum(new_done & (real_a | (trunc & a_ahead))),
                jnp.sum(new_done & (real_b | (trunc & b_ahead))),
                jnp.sum(new_done & trunc & ~a_ahead & ~b_ahead),
                jnp.sum(new_done & real_a),
                jnp.sum(new_done & real_b),
            ], dtype=jnp.int32)
            finished = finished | dones

            obs_state_a = reset_done_envs(obs_state_a, dones)
            obs_state_b = reset_done_envs(obs_state_b, dones)
            return (new_states, key, finished, counters, obs_state_a, obs_state_b), None

        (_, _, _, counters, _, _), _ = jax.lax.scan(
            scan_body,
            (states, key, finished, counters, batched_state, batched_state),
            None,
            length=truncation,
        )
        return counters

    counters = _play(agent_a.network, agent_b.network, key)
    return tuple(int(c) for c in counters)


def make_eval_env(cfg, truncation, stage_index=-1):
    """Selected curriculum-stage env, mirroring scripts/eval_selfplay.py."""
    stages = cfg.curriculum_stages
    if not stages:
        raise ValueError("tournament config has no curriculum stages")
    try:
        stage = stages[stage_index]
    except IndexError as error:
        raise ValueError(
            f"stage index {stage_index} is outside the {len(stages)}-stage curriculum"
        ) from error
    cities = (
        (stage.num_cities_min, stage.num_cities_max)
        if stage.num_cities_min is not None
        else (cfg.num_cities_min, cfg.num_cities_max)
    )
    castle = (
        (stage.castle_val_min, stage.castle_val_max)
        if stage.castle_val_min is not None
        else (cfg.castle_val_min, cfg.castle_val_max)
    )
    gs = cfg.max_grid_size
    return GeneralsEnv(
        grid_dims=(gs, gs),
        pad_to=cfg.pad_to,
        min_generals_distance=stage.min_generals_distance,
        max_generals_distance=stage.max_generals_distance,
        truncation=truncation,
        num_cities_range=cities,
        castle_val_range=castle,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default="artifacts/tournament")
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46])
    parser.add_argument("--iters", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--num-games", type=int, default=64,
                        help="games per pairing per side (total per pair = 2x)")
    parser.add_argument("--truncation", type=int, default=None,
                        help="defaults to the training config truncation")
    parser.add_argument("--stage-index", type=int, default=-1,
                        help="curriculum stage to evaluate (default: -1, final stage)")
    parser.add_argument("--key", type=int, default=7)
    parser.add_argument("--out", default="analysis/metrics/milestone_tournament.json")
    parser.add_argument("--fetch-from-s3", default=None, metavar="BUCKET",
                        help="download checkpoints from this S3 bucket first")
    parser.add_argument("--tiebreak", action="store_true",
                        help="score truncation draws by final land, then army")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if args.fetch_from_s3:
        fetch_from_s3(args.fetch_from_s3, args.seeds, args.iters, ckpt_dir)

    agents = []
    for seed in args.seeds:
        cfg_path = ckpt_dir / f"seed{seed}_config.yaml"
        for it in args.iters:
            eqx_path = ckpt_dir / f"seed{seed}_ema_{it}.eqx"
            agent = Agent.load(str(eqx_path), str(cfg_path))
            agent.name = f"s{seed}@{it}"
            agents.append(agent)
            print(f"loaded {agent.name}: {agent.param_count():,} params")

    cfg = agents[0].config
    truncation = args.truncation or cfg.truncation
    env = make_eval_env(cfg, truncation, args.stage_index)
    stage = cfg.curriculum_stages[args.stage_index]
    key = jrandom.PRNGKey(args.key)
    key, pool_key = jrandom.split(key)
    pool, _ = env.reset(pool_key)
    print(f"eval env: grid={cfg.max_grid_size} pad={cfg.pad_to} "
          f"stage={args.stage_index} dist={stage.min_generals_distance}-"
          f"{stage.max_generals_distance} truncation={truncation}")

    agent_map = {a.name: a for a in agents}
    names = list(agent_map)
    h2h = {n: {o: {"wins": 0, "losses": 0, "draws": 0, "real_wins": 0, "real_losses": 0}
               for o in names if o != n} for n in names}

    pairs = list(combinations(names, 2))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    for a, b in pairs:
        for p0, p1 in [(a, b), (b, a)]:
            key, mk = jrandom.split(key)
            t0 = time.time()
            if args.tiebreak:
                w0, w1, d, r0, r1 = play_match_tiebreak(
                    agent_map[p0], agent_map[p1], env, pool,
                    args.num_games, truncation, mk)
            else:
                w0, w1, d = play_match(agent_map[p0], agent_map[p1], env, pool,
                                       args.num_games, truncation, mk)
                r0, r1 = w0, w1
            h2h[p0][p1]["wins"] += w0
            h2h[p0][p1]["losses"] += w1
            h2h[p0][p1]["draws"] += d
            h2h[p0][p1]["real_wins"] += r0
            h2h[p0][p1]["real_losses"] += r1
            h2h[p1][p0]["wins"] += w1
            h2h[p1][p0]["losses"] += w0
            h2h[p1][p0]["draws"] += d
            h2h[p1][p0]["real_wins"] += r1
            h2h[p1][p0]["real_losses"] += r0
            done += 1
            print(f"[{done}/{2 * len(pairs)}] {p0} (P0) vs {p1} (P1): "
                  f"{w0}W/{w1}L/{d}D (captures {r0}/{r1})  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        # Checkpoint results after every completed pair so partial runs are usable.
        out_path.write_text(json.dumps({"h2h": h2h, "num_games_per_side": args.num_games,
                                        "truncation": truncation}, indent=2))

    elo = compute_elo(names, h2h)

    print("\n=== Overall (all opponents) ===")
    for n in sorted(names, key=lambda n: -elo[n]):
        w = sum(r["wins"] for r in h2h[n].values())
        l = sum(r["losses"] for r in h2h[n].values())
        d = sum(r["draws"] for r in h2h[n].values())
        dec = 100 * w / (w + l) if w + l else 0.0
        print(f"{n:10s} elo={elo[n]:7.1f}  {w}W/{l}L/{d}D  decisive_win={dec:.0f}%")

    pooled_comparisons = {}
    for earlier, later in zip(args.iters, args.iters[1:]):
        print(f"\n=== iter-{later} vs iter-{earlier} head-to-head ===")
        tw = tl = td = 0
        for seed_a in args.seeds:
            for seed_b in args.seeds:
                r = h2h[f"s{seed_a}@{later}"][f"s{seed_b}@{earlier}"]
                tw += r["wins"]; tl += r["losses"]; td += r["draws"]
                label = "same seed " if seed_a == seed_b else "cross seed"
                print(f"  {label} s{seed_a}@{later} vs s{seed_b}@{earlier}: "
                      f"{r['wins']}W/{r['losses']}L/{r['draws']}D")
        dec = 100 * tw / (tw + tl) if tw + tl else 0.0
        print(f"  pooled: {tw}W/{tl}L/{td}D  decisive_win={dec:.1f}%")
        pooled_comparisons[f"{later}_vs_{earlier}"] = {
            "wins": tw, "losses": tl, "draws": td,
        }

    output = {
        "h2h": h2h,
        "elo": elo,
        "num_games_per_side": args.num_games,
        "truncation": truncation,
        "stage_index": args.stage_index,
        "generals_distance": [
            stage.min_generals_distance,
            stage.max_generals_distance,
        ],
        "pooled_milestone_comparisons": pooled_comparisons,
    }
    if "2000_vs_1000" in pooled_comparisons:
        output["pooled_2000_vs_1000"] = pooled_comparisons["2000_vs_1000"]
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
