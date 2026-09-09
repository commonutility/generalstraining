"""Plot the iteration-1000 vs iteration-2000 milestone tournament."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS_DIR = Path(__file__).parent / "metrics"
TIEBREAK_PATH = METRICS_DIR / "milestone_tournament_tiebreak.json"
CAPTURE_PATH = METRICS_DIR / "milestone_tournament.json"
OUTPUT_PATH = METRICS_DIR / "milestone_tournament.png"
SEEDS = (44, 45, 46)


def wilson_interval(wins: int, losses: int, draws: int) -> tuple[float, float, float]:
    """Return score rate and Wilson 95% interval, counting draws as half a win."""
    n = wins + losses + draws
    successes = wins + 0.5 * draws
    p = successes / n
    z = 1.96
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return p, center - margin, center + margin


def main() -> None:
    tiebreak = json.loads(TIEBREAK_PATH.read_text())
    capture = json.loads(CAPTURE_PATH.read_text())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    fig.suptitle(
        "PPO milestone tournament — iteration 1,000 vs 2,000",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.952,
        "3 seeds · 512 games per checkpoint pairing · greedy EMA policies · "
        "23×23 final-curriculum maps",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )

    # A: final-state tiebreak Elo.
    ax = axes[0, 0]
    elo = tiebreak["elo"]
    ordered = sorted(elo, key=elo.get)
    colors = ["#4C78A8" if name.endswith("@1000") else "#F58518" for name in ordered]
    bars = ax.barh(ordered, [elo[name] for name in ordered], color=colors)
    ax.set_xlim(1200, 1680)
    ax.set_xlabel("Tiebreak Elo rating")
    ax.set_ylabel("Checkpoint (seed@iteration)")
    ax.set_title("A. Strength using land/army tiebreaks", loc="left", fontweight="bold")
    for bar, name in zip(bars, ordered):
        ax.text(
            bar.get_width() + 6,
            bar.get_y() + bar.get_height() / 2,
            f"{elo[name]:.1f}",
            va="center",
            fontsize=9,
        )
    ax.plot([], [], color="#4C78A8", linewidth=7, label="Iteration 1,000")
    ax.plot([], [], color="#F58518", linewidth=7, label="Iteration 2,000")
    ax.legend(loc="lower right", frameon=True)

    # B: direct same-seed tiebreak comparison with binomial uncertainty.
    ax = axes[0, 1]
    same_seed_rows: list[tuple[str, int, int, int]] = []
    for seed in SEEDS:
        result = tiebreak["h2h"][f"s{seed}@2000"][f"s{seed}@1000"]
        same_seed_rows.append(
            (f"Seed {seed}", result["wins"], result["losses"], result["draws"])
        )
    pooled = (
        "Pooled",
        sum(row[1] for row in same_seed_rows),
        sum(row[2] for row in same_seed_rows),
        sum(row[3] for row in same_seed_rows),
    )
    direct_rows = [*same_seed_rows, pooled]
    intervals = [wilson_interval(wins, losses, draws) for _, wins, losses, draws in direct_rows]
    rates = np.array([item[0] for item in intervals]) * 100
    lower = rates - np.array([item[1] for item in intervals]) * 100
    upper = np.array([item[2] for item in intervals]) * 100 - rates
    x = np.arange(len(direct_rows))
    ax.bar(x, rates, color=["#72B7B2", "#72B7B2", "#72B7B2", "#F58518"])
    ax.errorbar(x, rates, yerr=[lower, upper], fmt="none", ecolor="#333333", capsize=4)
    ax.axhline(50, color="#777777", linestyle="--", linewidth=1.2, label="Even matchup")
    ax.set_xticks(x, [row[0] for row in direct_rows])
    ax.set_ylim(45, 65)
    ax.set_ylabel("Iteration-2,000 score rate (%)")
    ax.set_xlabel("Training seed")
    ax.set_title("B. Iteration 2,000 wins each same-seed matchup", loc="left", fontweight="bold")
    for index, (rate, row) in enumerate(zip(rates, direct_rows)):
        ax.text(index, rate + upper[index] + 0.5, f"{rate:.1f}%", ha="center", fontsize=9)
        ax.text(
            index,
            45.6,
            f"{row[1]}–{row[2]}–{row[3]}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    ax.legend(loc="upper left")

    # C: Elo progression makes between-seed variation visible.
    ax = axes[1, 0]
    seed_colors = {44: "#4C78A8", 45: "#54A24B", 46: "#E45756"}
    for seed in SEEDS:
        values = [elo[f"s{seed}@1000"], elo[f"s{seed}@2000"]]
        ax.plot(
            [1000, 2000],
            values,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=seed_colors[seed],
            label=f"Seed {seed}",
        )
        change = values[1] - values[0]
        ax.annotate(
            f"{change:+.1f}",
            (2000, values[1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=seed_colors[seed],
        )
    ax.set_xticks([1000, 2000], ["1,000", "2,000"])
    ax.set_ylim(1250, 1680)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Tiebreak Elo rating")
    ax.set_title("C. Seed variance exceeds the milestone gain", loc="left", fontweight="bold")
    ax.legend(loc="center left")

    # D: capture-only outcomes expose the truncation problem.
    ax = axes[1, 1]
    capture_rows = []
    for seed in SEEDS:
        result = capture["h2h"][f"s{seed}@2000"][f"s{seed}@1000"]
        total = result["wins"] + result["losses"] + result["draws"]
        capture_rows.append(
            (
                100 * result["wins"] / total,
                100 * result["losses"] / total,
                100 * result["draws"] / total,
            )
        )
    capture_arr = np.array(capture_rows)
    labels = [f"Seed {seed}" for seed in SEEDS]
    ax.bar(labels, capture_arr[:, 0], color="#54A24B", label="2k captures")
    ax.bar(
        labels,
        capture_arr[:, 1],
        bottom=capture_arr[:, 0],
        color="#E45756",
        label="1k captures",
    )
    ax.bar(
        labels,
        capture_arr[:, 2],
        bottom=capture_arr[:, 0] + capture_arr[:, 1],
        color="#BAB0AC",
        label="Truncation draws",
    )
    for index, draw_rate in enumerate(capture_arr[:, 2]):
        ax.text(index, 50, f"{draw_rate:.1f}% draws", ha="center", va="center", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Outcome share (%)")
    ax.set_title("D. Capture-only evaluation is draw-dominated", loc="left", fontweight="bold")
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=True)

    fig.text(
        0.02,
        0.015,
        "Tiebreak: unresolved games scored by final land, then army. "
        "Error bars: Wilson 95% interval. Sources: milestone_tournament.json and "
        "milestone_tournament_tiebreak.json (Aug 22, 2026).",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.935))
    fig.savefig(OUTPUT_PATH, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
