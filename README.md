# Generals Training

Self-play PPO and representation pretraining for [generals.io](https://generals.io). The game simulator is the `generals-bots` environment in `vendor/generals-bots`.

Requires Python 3.11+ and a JAX build for your machine (CPU or GPU).

## Layout

```
vendor/generals-bots/   JAX game environment (submodule)
src/generals_pretraining/
  config.py             Training config dataclass
  envs/                 Environment wrapper
  models/               Policy–value transformer
  pretraining/          Belief-state dataset, targets, and encoder
  rl/                   PPO, rollouts, rewards, evaluations
  evaluation/           Matchups and agent helpers
  utils/                Logging
scripts/                Train, collect data, evaluate
configs/                YAML presets (map sizes, experiments, CPU smoke)
jobs/                   AWS Batch submit / status / logs / cancel
infra/aws/              CDK stacks for GPU jobs
analysis/               Metrics plots and tournament scripts
tests/                  Smoke tests
docs/                   Experiment contract and AWS job docs
```

## Setup

```bash
git submodule update --init --recursive
pip install -e vendor/generals-bots
pip install -e ".[dev]"
```

## Train

Belief pretraining (partial observations in, privileged simulator state as labels only):

```bash
python scripts/collect_belief_dataset.py --output data/belief/random-s.npz
python scripts/train_belief.py \
  --dataset data/belief/random-s.npz \
  --output artifacts/pretraining/belief_s.eqx
```

PPO from scratch, or with a pretrained encoder:

```bash
python scripts/train_ppo.py --config configs/experiments/s_budget.yaml \
  --run_name scratch_seed44

python scripts/train_ppo.py --config configs/experiments/s_budget.yaml \
  --run_name belief_seed44 \
  --init_encoder_checkpoint artifacts/pretraining/belief_s.eqx
```

CPU smoke test:

```bash
python scripts/train_ppo.py --config configs/smoke/L_7d_gae90_cpu.yaml
```

GPU jobs go through the Batch CLI. See `docs/aws-jobs.md`.

```bash
python jobs/cli.py --profile generals-jobs submit \
  --name training-run \
  --command "python scripts/train_ppo.py --config configs/experiments/s_budget.yaml"
```

## Evaluate

```bash
python scripts/evaluate.py
python scripts/eval_selfplay.py
```

## Tests

```bash
pytest
```

Training logs to Weights & Biases when `.secrets/wandb_token.txt` is present; otherwise console only.

Experiment design and leakage rules: `docs/EXPERIMENTS.md`.
