"""Pull AWS Batch training logs from CloudWatch and plot metrics over time.

Each training iteration prints a line like:

    [00:38:15] Iter 130/2000 | Loss: 1.6439 | PG: -0.0121 | VF: 3.2035 | ...

This script fetches the full log stream for one or more Batch jobs, parses
those lines (plus the periodic EVAL lines) into per-run CSVs, and renders
comparison plots across runs.

Usage:
    python scripts/plot_cloud_metrics.py \
        --job us-east-1:JOB_ID:seed44 \
        --job us-west-2:JOB_ID:seed45 \
        --out-dir analysis/metrics
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import boto3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_GROUP = "/aws/batch/generals-training"

ITER_RE = re.compile(r"^\[(\d+):(\d+):(\d+)\] Iter\s+(\d+)/(\d+) \| (.*)$")
FIELD_RE = re.compile(r"([A-Za-z/]+): ([^|]+?)(?: \||$)")
WLD_RE = re.compile(r"(\d+)/(\d+)/(\d+) \((\d+)%/(\d+)%/(\d+)%\)")
EVAL_RE = re.compile(r"EVAL: (\d+)W/(\d+)L/(\d+)D \((\d+)%\)")

# Metric label in the log line -> CSV column name.
SCALAR_FIELDS = {
    "Loss": "loss",
    "PG": "pg_loss",
    "VF": "vf_loss",
    "Ent": "entropy",
    "KL": "approx_kl",
    "Clip": "clip_frac",
    "GNorm": "grad_norm",
    "EV": "explained_variance",
    "Reward": "reward",
    "Eps": "episodes",
    "EpLen": "episode_length",
    "Cities": "cities",
    "LR": "lr",
    "SPS": "sps",
}

PLOTS = [
    ("vf_loss", "Value function loss"),
    ("explained_variance", "Explained variance"),
    ("loss", "Total loss"),
    ("entropy", "Policy entropy"),
    ("approx_kl", "Approx KL"),
    ("win_pct", "Rollout win %"),
    ("episode_length", "Mean episode length"),
    ("eval_win_pct", "Eval win % (greedy vs random)"),
]


def fetch_log_lines(region: str, job_id: str) -> list[str]:
    session = boto3.Session(region_name=region)
    batch = session.client("batch")
    jobs = batch.describe_jobs(jobs=[job_id])["jobs"]
    if not jobs:
        raise SystemExit(f"job {job_id} not found in {region}")
    stream = jobs[0].get("container", {}).get("logStreamName")
    if not stream:
        raise SystemExit(f"job {job_id} has no log stream yet")

    logs = session.client("logs")
    lines: list[str] = []
    paginator = logs.get_paginator("filter_log_events")
    for page in paginator.paginate(logGroupName=LOG_GROUP, logStreamNames=[stream]):
        lines.extend(event["message"] for event in page["events"])
    return lines


def parse_lines(lines: list[str]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    last_row: dict[str, float] | None = None
    for line in lines:
        eval_match = EVAL_RE.search(line)
        if eval_match and last_row is not None:
            wins, losses, draws, pct = (int(g) for g in eval_match.groups())
            last_row["eval_wins"] = wins
            last_row["eval_losses"] = losses
            last_row["eval_draws"] = draws
            last_row["eval_win_pct"] = pct
            continue
        iter_match = ITER_RE.match(line)
        if not iter_match:
            continue
        hours, minutes, seconds, iteration, total, rest = iter_match.groups()
        row: dict[str, float] = {
            "iteration": int(iteration),
            "total_iterations": int(total),
            "elapsed_seconds": int(hours) * 3600 + int(minutes) * 60 + int(seconds),
        }
        for label, value in FIELD_RE.findall(rest):
            if label in SCALAR_FIELDS:
                try:
                    row[SCALAR_FIELDS[label]] = float(value.strip().rstrip("s"))
                except ValueError:
                    pass
            elif label == "W/L/D":
                wld = WLD_RE.search(value)
                if wld:
                    row["wins"], row["losses"], row["draws"] = (
                        int(wld.group(1)), int(wld.group(2)), int(wld.group(3)))
                    row["win_pct"], row["loss_pct"], row["draw_pct"] = (
                        int(wld.group(4)), int(wld.group(5)), int(wld.group(6)))
        rows.append(row)
        last_row = row
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_runs(runs: dict[str, list[dict[str, float]]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for ax, (column, title) in zip(axes.flat, PLOTS):
        for label, rows in runs.items():
            points = [(r["iteration"], r[column]) for r in rows if column in r]
            if not points:
                continue
            xs, ys = zip(*sorted(points))
            ax.plot(xs, ys, label=label, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Generals PPO training metrics (parsed from CloudWatch)", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "training_metrics.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        metavar="REGION:JOB_ID:LABEL",
        help="Batch job to include; repeatable",
    )
    parser.add_argument("--out-dir", default="analysis/metrics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[dict[str, float]]] = {}
    for spec in args.job:
        region, job_id, label = spec.split(":", 2)
        rows = parse_lines(fetch_log_lines(region, job_id))
        if not rows:
            print(f"{label}: no iteration lines found yet, skipping")
            continue
        runs[label] = rows
        csv_path = out_dir / f"{label}.csv"
        write_csv(rows, csv_path)
        print(f"{label}: {len(rows)} iterations parsed -> {csv_path}")

    if not runs:
        raise SystemExit("no data parsed from any job")
    plot_path = plot_runs(runs, out_dir)
    print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
