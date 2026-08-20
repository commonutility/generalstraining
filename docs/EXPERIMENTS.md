# Workshop experiment contract

## Question

Does a representation trained to infer strategically relevant latent state
reduce the online interaction budget required by self-play PPO?

This repository starts from the released AverageJoe code rather than attempting
to reproduce it independently. The simulator is pinned as a git submodule. The
primary intervention is the initialization of `HistoryTransformer`; PPO,
architecture, curriculum, evaluation maps, and online interaction budgets stay
fixed.

## Conditions

1. **Scratch**: random `HistoryTransformer` initialization.
2. **Behavior cloning**: the same transformer pretrained to predict actions
   from filtered expert replays, with fresh policy and value heads for PPO.
3. **Belief**: the same transformer pretrained from partial observations to
   predict the enemy general, hidden enemy ownership, and hidden enemy armies,
   with fresh policy and value heads for PPO.

The repository currently implements the scratch and belief paths. Behavior
cloning is intentionally not faked with simulator actions: it requires a
versioned human-replay dataset and legality-checked replay importer. That is the
next data dependency.

## Leakage boundary

`pretraining.data` constructs the network input from `get_observation`, exactly
as PPO does. Full `GameState` is used only by `make_belief_targets`. Privileged
state must never be added to the input arrays, temporal features, policy
rollouts, or evaluation.

For a held-out game, every timestep belongs to the same split. Do not randomly
split individual states across train and validation, because adjacent states
are near duplicates.

## Primary endpoint

Plot evaluation Elo or fixed-opponent win rate against cumulative PPO player
steps. Report:

- steps to each predeclared strength threshold;
- area under the learning curve through the common interaction budget;
- final strength at the common budget;
- bootstrap confidence intervals over evaluation games and variation over at
  least three training seeds.

Wall-clock time is secondary because pretraining has an offline cost. Report it
separately as pretraining accelerator-hours plus PPO accelerator-hours.

## Mechanism tests

Freeze checkpoints before PPO and at matched PPO interaction counts. Fit
linear probes on game-disjoint data for enemy-general location, hidden enemy
ownership, hidden enemy army, and eventual winner. The probe dataset and
regularization search must be identical across conditions.

The key mediation claim should only be made if representation quality and
online learning speed covary across seeds or objectives. A single attention
visualization is qualitative evidence, not a mechanism test.

## Commands

Initialize dependencies:

```bash
git submodule update --init --recursive
python -m pip install -e vendor/generals-bots
python -m pip install -e ".[dev]"
```

Collect a small random-policy dataset:

```bash
python scripts/collect_belief_dataset.py \
  --output data/belief/random-s.npz \
  --grid-size 12 --num-envs 64 --num-steps 256
```

Pretrain:

```bash
python scripts/train_belief.py \
  --dataset data/belief/random-s.npz \
  --model-config configs/experiments/s_budget.yaml \
  --output artifacts/pretraining/belief_s.eqx
```

Run the controlled PPO pair:

```bash
python main.py --config configs/experiments/s_budget.yaml \
  --run_name scratch_seed44 --seed 44

python main.py --config configs/experiments/s_budget.yaml \
  --run_name belief_seed44 --seed 44 \
  --init_encoder_checkpoint artifacts/pretraining/belief_s.eqx
```

Repeat with predeclared seeds. Use one offline dataset and one pretrained
checkpoint per pretraining seed; do not select checkpoints using downstream
PPO results.

## Planned extensions

- Add a generals.io replay importer and behavior-cloning objective.
- Add game-disjoint dataset manifests and sharded storage for large corpora.
- Add frozen linear probes and matched-step checkpoint evaluation.
- Add an out-of-distribution map suite after the in-distribution result is
  stable.
- Test policy-head transfer only as a separate ablation; it is not
  representation-only pretraining.
