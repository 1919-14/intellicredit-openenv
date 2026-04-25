"""
compare_results.py
==================
Compare baseline_results.json (pre-training) with post_training_results.json
(same format, generated after GRPO).

Usage:
    python compare_results.py
    python compare_results.py --baseline baseline_results.json \
                              --after    post_training_results.json \
                              --out      comparison.png

The post-training JSON is produced by the same evaluation harness that created
baseline_results.json — just run it again after training and save the output.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Metrics to compare  (key-in-breakdown, display-label, higher-is-better)
# ─────────────────────────────────────────────────────────────────────────────
METRICS = [
    ("accuracy",              "Accuracy",          True),
    ("hard_rule_compliance",  "Hard Rule Comply",  True),
    ("forensic_handling",     "Forensic Handling", True),
    ("survival_rate",         "Survival Rate",     True),
    ("npa_rate",              "NPA Rate",          False),   # lower = better
    ("capital_utilization",   "Capital Util.",     True),
]

TOP_METRICS = [
    ("score",        "Task Score",    True),
    ("total_reward", "Total Reward",  True),
]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_table(baseline: dict, after: dict) -> None:
    tasks = sorted(set(baseline) & set(after))
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("  Before vs. After Training — Per-Task Summary")
    print("╠══════════════════════════════════════════════════════════════╣")
    header = f"  {'Task':<8} {'Metric':<22} {'Before':>8} {'After':>8} {'Δ':>8}"
    print(header)
    print("  " + "─" * 58)
    for task in tasks:
        b, a = baseline[task], after[task]
        for key, label, higher_better in TOP_METRICS:
            bv = b.get(key, float("nan"))
            av = a.get(key, float("nan"))
            delta = av - bv
            arrow = ("↑" if delta > 0 else "↓") if not np.isnan(delta) else ""
            sign  = "+" if delta > 0 else ""
            print(f"  {task:<8} {label:<22} {bv:>8.4f} {av:>8.4f} {sign}{delta:>7.4f} {arrow}")
        for key, label, _ in METRICS:
            bv = b["breakdown"].get(key, float("nan"))
            av = a["breakdown"].get(key, float("nan"))
            delta = av - bv
            arrow = ("↑" if delta > 0 else "↓") if not np.isnan(delta) else ""
            sign  = "+" if delta > 0 else ""
            print(f"  {'':8} {label:<22} {bv:>8.4f} {av:>8.4f} {sign}{delta:>7.4f} {arrow}")
        print("  " + "─" * 58)
    print("╚══════════════════════════════════════════════════════════════╝\n")


def plot_comparison(baseline: dict, after: dict, out: str) -> None:
    tasks   = sorted(set(baseline) & set(after))
    n_tasks = len(tasks)

    all_metrics = TOP_METRICS + METRICS
    n_metrics   = len(all_metrics)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    cols = [p["color"] for p in prop_cycle]
    c_before, c_after = cols[0], cols[1]

    fig, axes = plt.subplots(
        n_tasks, n_metrics,
        figsize=(n_metrics * 2.4, n_tasks * 3.0),
        squeeze=False,
    )
    fig.suptitle("IntelliCredit — Baseline vs. Post-Training", fontsize=13,
                 fontweight="bold", y=1.01)

    for row, task in enumerate(tasks):
        b_data = baseline[task]
        a_data = after[task]

        for col, (key, label, higher_better) in enumerate(all_metrics):
            ax = axes[row][col]

            # Pull value from top-level or breakdown
            if key in ("score", "total_reward"):
                bv = b_data.get(key, 0.0)
                av = a_data.get(key, 0.0)
            else:
                bv = b_data["breakdown"].get(key, 0.0)
                av = a_data["breakdown"].get(key, 0.0)

            bars = ax.bar(["Before", "After"], [bv, av],
                          color=[c_before, c_after], alpha=0.82, width=0.5,
                          edgecolor="white", linewidth=0.8)

            # Annotate bar tops
            for bar, val in zip(bars, [bv, av]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * max(abs(bv), abs(av), 0.01),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold")

            # Delta arrow on the "After" bar
            delta = av - bv
            if abs(delta) > 1e-6:
                sign  = "+" if delta > 0 else ""
                is_improvement = (delta > 0) == higher_better
                colour = "#2ca02c" if is_improvement else "#d62728"
                ax.set_title(f"{label}\n{sign}{delta:.3f}", fontsize=8,
                             color=colour, fontweight="bold")
            else:
                ax.set_title(label, fontsize=8)

            if row == 0 and col == 0:
                ax.set_ylabel(task.upper(), fontsize=9, fontweight="bold")
            elif col == 0:
                ax.set_ylabel(task.upper(), fontsize=9, fontweight="bold")

            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  📊 Saved → {out}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare before/after training results")
    parser.add_argument("--baseline", default="baseline_results.json",
                        help="Pre-training results JSON (default: baseline_results.json)")
    parser.add_argument("--after",    default="post_training_results.json",
                        help="Post-training results JSON (default: post_training_results.json)")
    parser.add_argument("--out",      default="comparison.png",
                        help="Output plot path (default: comparison.png)")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    after_path    = Path(args.after)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    if not after_path.exists():
        raise FileNotFoundError(
            f"Post-training results not found: {after_path}\n"
            "Run your evaluation harness after training and save to that path."
        )

    baseline = load(str(baseline_path))
    after    = load(str(after_path))

    print_table(baseline, after)
    plot_comparison(baseline, after, args.out)


if __name__ == "__main__":
    main()
