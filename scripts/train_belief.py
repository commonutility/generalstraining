"""Pretrain the AverageJoe representation on privileged belief targets."""

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from generals_pretraining.config import Config
from generals_pretraining.models import build_network
from generals_pretraining.pretraining.model import BeliefPretrainer, belief_loss
from generals_pretraining.pretraining.targets import BeliefTargets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-config", default="configs/experiments/s_budget.yaml")
    parser.add_argument("--output", default="artifacts/pretraining/belief_s.eqx")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_dataset(path):
    dataset = np.load(path)
    observations = jnp.asarray(dataset["observations"], dtype=jnp.float32)
    temporal = jnp.asarray(dataset["temporal"], dtype=jnp.float32)
    targets = BeliefTargets(
        enemy_general=jnp.asarray(dataset["enemy_general"]),
        enemy_armies=jnp.asarray(dataset["enemy_armies"]),
        enemy_ownership=jnp.asarray(dataset["enemy_ownership"]),
        hidden_mask=jnp.asarray(dataset["hidden_mask"]),
        passable_mask=jnp.asarray(dataset["passable_mask"]),
    )
    return observations, temporal, targets


@eqx.filter_jit
def update(model, opt_state, optimizer, observations, temporal, targets):
    def batch_loss(candidate):
        losses, metrics = jax.vmap(belief_loss, in_axes=(None, 0, 0, 0))(candidate, observations, temporal, targets)
        return losses.mean(), jax.tree.map(jnp.mean, metrics)

    (_, metrics), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, metrics


def main():
    args = parse_args()
    observations, temporal, targets = load_dataset(args.dataset)
    config = Config.from_yaml(args.model_config)
    key = jrandom.PRNGKey(args.seed)
    key, encoder_key, head_key = jrandom.split(key, 3)
    model = BeliefPretrainer(build_network(config, encoder_key), key=head_key)

    optimizer = optax.adamw(args.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    sample_count = observations.shape[0]
    batches_per_epoch = sample_count // args.batch_size
    if batches_per_epoch == 0:
        raise ValueError(f"Dataset has {sample_count} samples, fewer than batch size {args.batch_size}")

    for epoch in range(args.epochs):
        key, permutation_key = jrandom.split(key)
        order = np.asarray(jrandom.permutation(permutation_key, sample_count))
        epoch_metrics = []
        for batch_index in range(batches_per_epoch):
            indices = order[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
            batch_targets = jax.tree.map(lambda value: value[indices], targets)
            model, opt_state, metrics = update(
                model,
                opt_state,
                optimizer,
                observations[indices],
                temporal[indices],
                batch_targets,
            )
            epoch_metrics.append(metrics)
        means = jax.tree.map(lambda *values: float(jnp.mean(jnp.stack(values))), *epoch_metrics)
        print(
            f"epoch {epoch + 1:03d}/{args.epochs} "
            f"loss={means['loss']:.4f} general={means['general_loss']:.4f} "
            f"army={means['army_loss']:.4f} ownership={means['ownership_loss']:.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(output, model.encoder)
    eqx.tree_serialise_leaves(output.with_suffix(".pretrainer.eqx"), model)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "model_config": args.model_config,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "samples": sample_count,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved PPO-compatible encoder checkpoint to {output}")


if __name__ == "__main__":
    main()
