# Data

Datasets are generated locally and not tracked in git (everything in this
directory except this README is gitignored).

## Belief-pretraining dataset

Generate the offline belief dataset from simulator self-play:

```bash
python scripts/collect_belief_dataset.py --output data/belief/random-s.npz
```

Partial observations are the inputs; the full simulator state supplies
privileged labels only. See `docs/EXPERIMENTS.md` for the full pipeline and
the exact commands used in the controlled comparison.
