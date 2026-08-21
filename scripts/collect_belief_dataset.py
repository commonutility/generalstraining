"""Generate an offline belief-pretraining dataset from simulator self-play."""

import argparse

import jax.random as jrandom

from generals_pretraining.envs import GeneralsEnv
from generals_pretraining.pretraining.data import collect_belief_dataset, save_belief_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/belief/random-small.npz")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--min-generals-distance", type=int, default=6)
    parser.add_argument("--max-generals-distance", type=int, default=17)
    parser.add_argument("--truncation", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = GeneralsEnv(
        grid_dims=(args.grid_size, args.grid_size),
        pad_to=args.grid_size,
        min_generals_distance=args.min_generals_distance,
        max_generals_distance=args.max_generals_distance,
        truncation=args.truncation,
        pool_size=args.pool_size,
    )
    pool, _ = env.reset(jrandom.PRNGKey(args.seed))
    samples = collect_belief_dataset(
        env,
        pool,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed + 1,
    )
    save_belief_dataset(args.output, samples)
    count = args.num_envs * args.num_steps * 2
    print(f"Saved {count:,} player-state samples to {args.output}")


if __name__ == "__main__":
    main()
