<div align="center">

# Average Joe

**The first superhuman [generals.io](https://generals.io) bot, trained from scratch with self-play reinforcement learning.**

*“Its ability to flow army in complex situations is phenomenal.”*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white&style=flat-square)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-5e35b1?style=flat-square)](https://github.com/jax-ml/jax)
[![Equinox](https://img.shields.io/badge/Equinox-d8973c?style=flat-square)](https://github.com/patrick-kidger/equinox)
[![generals.io rank #1](https://img.shields.io/badge/generals.io-%231%20on%20ladder-e23a3a?style=flat-square)](https://generals.io/profiles/Average%20Joe)

<p align="center">
  <img src="assets/game1.webp" width="250" alt="Self-play game 1" />
  <img src="assets/game2.webp" width="250" alt="Self-play game 2" />
  <img src="assets/game3.webp" width="250" alt="Self-play game 3" />
</p>

</div>

---

> [!IMPORTANT]
> This fork extends Average Joe for the workshop project **When Does
> Pretraining Help Self-Play?** It keeps the released PPO baseline intact and
> adds belief-state representation pretraining from privileged simulator
> labels. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the controlled
> comparison and exact commands.

## Workshop setup

Clone with the pinned simulator and install both editable packages:

```bash
git clone --recurse-submodules https://github.com/commonutility/generalstraining.git
cd generalstraining
pip install -e vendor/generals-bots
pip install -e ".[dev]"
```

The first end-to-end path is:

```bash
# Partial observations are inputs; full simulator state supplies labels only.
python scripts/collect_belief_dataset.py \
  --output data/belief/random-s.npz

python scripts/train_belief.py \
  --dataset data/belief/random-s.npz \
  --output artifacts/pretraining/belief_s.eqx

# Controlled online-budget comparison.
python main.py --config configs/experiments/s_budget.yaml \
  --run_name scratch_seed44
python main.py --config configs/experiments/s_budget.yaml \
  --run_name belief_seed44 \
  --init_encoder_checkpoint artifacts/pretraining/belief_s.eqx
```

The encoder checkpoint is loadable by `main.py` through the
`init_encoder_checkpoint` option. It transfers torso parameters while preserving
fresh PPO policy and value heads, keeping the intervention at representation
initialization.

---

Average Joe is a bot for [generals.io](https://generals.io) — a real-time, fog-of-war
strategy game — that taught itself to play at a **superhuman** level, from zero, through
millions of games against itself.

- 🏆 **Superhuman, from scratch** — trained purely by self-play; it never sees a human game.
- 🔥 **Blazing-fast simulator** — runs on [**generals-bots**](https://github.com/strakam/generals-bots), a fully-vectorized JAX environment.
- 🔁 **Fully reproducible** — one config and one command reproduce the released agent end to end.
- 🛠️ **Powered by [JAX](https://github.com/jax-ml/jax) + [Equinox](https://github.com/patrick-kidger/equinox)** — a small, pure-functional, JIT-compiled training loop.

## 📊 Results

In its first **1,000 ranked games** on the [generals.io](https://generals.io) 1v1 ladder, Average Joe won **81.5%** and finished as the **#1-rated player** — ahead of the strongest human and well clear of the prior AI state of the art.

<p align="center">
  <img src="assets/leaderboard.png" width="780" alt="generals.io 1v1 leaderboard: Average Joe leads on OpenSkill rating" />
</p>

## 🎮 Watch it play

Average Joe competes on the [generals.io](https://generals.io) 1v1 ladder — watch its live games and replays:

- [Average Joe](https://generals.io/profiles/Average%20Joe)
- [L_7d_gae90_30k_ema](https://generals.io/profiles/L_7d_gae90_30k_ema)

## 🧠 Architecture

<p align="center">
  <img src="assets/architecture.png" width="820" alt="Average Joe network architecture" />
</p>

The board — plus a short history of each player's army and land — is encoded as tokens and
run through a small transformer with two heads: one picks the move, the other estimates who
is winning.

- **Policy–value transformer** — pre-norm self-attention over board + temporal tokens; emits per-cell move logits and a distributional (HL-Gauss) value. &nbsp;·&nbsp; `networks/transformer.py`
- **Self-play PPO** — one network plays both sides; GAE, top-k advantage filtering, EMA weights for evaluation. &nbsp;·&nbsp; `train/ppo.py`

## 📦 Install

Requires Python ≥ 3.11 and a [JAX](https://docs.jax.dev/en/latest/installation.html) build
for your accelerator (CPU/GPU/TPU).

```bash
pip install -e .
```

Average Joe runs on the [`generals-bots`](https://github.com/strakam/generals-bots)
environment (the `generals.core.*` package — the vectorized game, observations, and reward
functions), a **separate, non-PyPI** package. Install it from source and make it importable
before running.

## 🚀 Train

```bash
python main.py --config configs/custom/L_7d_gae90.yaml
```

`L_7d_gae90` is the config behind the released agent. Checkpoints (a regular and an EMA copy)
are written to `checkpoints/<run_name>/`, alongside the exact config that produced them. Any
`Config` field can be overridden on the CLI, e.g. `--num_envs 256`. `configs/` also holds
map-size presets (`S` / `M` / `L` / `default`).

For a CPU-only smoke test of the released PPO path from random weights, run:

```bash
python main.py --config configs/smoke/L_7d_gae90_cpu.yaml
```

This config keeps the released `L_7d_gae90` architecture and PPO hyperparameters,
but reduces rollout, pool, minibatch, truncation, and evaluation sizes to validate
the loop on one CPU device. It writes `checkpoints/smoke_L_7d_gae90_cpu/config.yaml`,
an EMA checkpoint at iteration 2, and `smoke_L_7d_gae90_cpu_final.eqx`. The saved
config intentionally leaves `init_checkpoint` and `init_encoder_checkpoint` empty.

When moving to a GPU/accelerator, use the released config directly. Disable
reference Elo until the unpublished reference checkpoints are available:

```bash
python main.py --config configs/custom/L_7d_gae90.yaml --ref_eval_every 0
```

Normal GPU training runs through the managed AWS Batch interface:

```bash
python jobs/cli.py --profile generals-jobs submit \
  --name training-run \
  --command "python main.py --config configs/custom/L_7d_gae90.yaml --ref_eval_every 0"
```

See [`docs/aws-jobs.md`](docs/aws-jobs.md) for workload classes, monitoring,
Spot usage, artifact paths, and the `us-west-2` fallback.

For the 8,000-iteration AverageJoe run with exact full and EMA checkpoints at
iterations 1,000, 2,000, 4,000, and 8,000:

```bash
python main.py --config configs/experiments/L_7d_gae90_8k.yaml
```

The trainer supports exact milestones for any run through `save_at` in YAML or
`--save_at 1000 2000 4000 8000` on the CLI. Set `save_every: 0` and
`ckpt_every: 0` when only those milestone files should be written. Logged
`train/env_interactions` and `train/agent_interactions` account for the active
device count; on one device this 8,000-iteration run collects 2,097,152,000
environment steps and 4,194,304,000 two-player agent transitions.

The upstream repository did not publish a dependency lock, released-agent checkpoint,
or authoritative simulator revision, so this command recreates the released PPO setup
but is not a bitwise reproduction guarantee.

## 🕹️ Evaluate / play

```bash
python evals/eval.py                                       # vs a random opponent (pygame)
python evals/eval_selfplay.py                              # the agent vs itself
```

## 📈 Logging (optional)

Training logs to [Weights & Biases](https://wandb.ai) when a token is present at
`.secrets/wandb_token.txt`; otherwise it runs console-only.
