#!/usr/bin/env python3
"""Plot GavelFGD-Combined experiment results.

Three figures:
1. Absolute metrics (JCT, frag rate, frag total, unalloc%) vs utilization
2. Improvement % over Gavel baseline
3. Improvement % over GavelFGD baseline
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR  = Path(__file__).parent / "results" / "gavelfgd_combined"
FIGURES_DIR  = Path(__file__).parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
POLICY_STYLE = {
    "gavel":    {"color": "#1f77b4", "marker": "o", "ls": "-",  "label": "Gavel"},
    "gavelfgd": {"color": "#ff7f0e", "marker": "s", "ls": "--", "label": "GavelFGD"},
    "gavelfgd_combined": {"color": "#d62728", "marker": "D", "ls": "-", "label": "GavelFGD+"},
}


# ── helpers ────────────────────────────────────────────────────────────────────
def load_results():
    records = []
    for f in sorted(RESULTS_DIR.glob("exp_*.json")):
        with open(f) as fp:
            data = json.load(fp)
        records.extend(data if isinstance(data, list) else [data])
    return records


def classify(name: str) -> str:
    if name.startswith("gavelfgd_combined"):
        return "gavelfgd_combined"
    if name.startswith("gavelfgd"):
        return "gavelfgd"
    if name.startswith("gavel"):
        return "gavel"
    return "other"


def aggregate(records_for_policy):
    """Average over seeds, keyed by lam. Returns list of dicts sorted by utilization."""
    buckets = defaultdict(list)
    for r in records_for_policy:
        buckets[r["lam"]].append(r)
    rows = []
    for lam in sorted(buckets, reverse=True):   # low lam = high load; sort low→high util
        recs = buckets[lam]
        row = {"lam": lam}
        for m in ["avg_jct", "avg_frag_rate", "avg_frag_total",
                  "avg_unalloc_pct", "avg_utilization"]:
            vals = [r[m] for r in recs if m in r]
            row[m]            = np.mean(vals) if vals else np.nan
            row[m + "_std"]   = np.std(vals)  if vals else np.nan
        rows.append(row)
    return sorted(rows, key=lambda r: r["avg_utilization"])


def plot_metric(ax, rows_by_policy, metric, ylabel, title, policies, error_bars=True):
    for pol in policies:
        rows = rows_by_policy.get(pol)
        if not rows:
            continue
        st = POLICY_STYLE[pol]
        xs = [r["avg_utilization"] for r in rows]
        ys = [r[metric] for r in rows]
        es = [r[metric + "_std"] for r in rows]
        if error_bars:
            ax.errorbar(xs, ys, yerr=es,
                        color=st["color"], marker=st["marker"],
                        linestyle=st["ls"], linewidth=1.8, markersize=6,
                        capsize=3, label=st["label"])
        else:
            ax.plot(xs, ys,
                    color=st["color"], marker=st["marker"],
                    linestyle=st["ls"], linewidth=1.8, markersize=6,
                    label=st["label"])
    ax.set_xlabel("GPU Utilization (%)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Absolute metrics
# ══════════════════════════════════════════════════════════════════════════════
def figure_absolute(rows_by_policy):
    POLICIES = ["gavel", "gavelfgd", "gavelfgd_combined"]
    PANELS = [
        ("avg_jct",        "Avg JCT (s)",           "Average Job Completion Time"),
        ("avg_frag_rate",  "Frag Rate (GPUs/node)",  "Fragmentation Rate"),
        ("avg_frag_total", "Frag Total (GPUs/node)", "Total Fragmentation"),
        ("avg_unalloc_pct","Unallocated GPU (%)",    "Unallocated GPU %"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "GavelFGD vs GavelFGD+: Absolute Metrics\n"
        "(Alibaba cluster, steady-state, 7 load levels)",
        fontsize=13, fontweight="bold"
    )

    for ax, (metric, ylabel, title) in zip(axes.flat, PANELS):
        plot_metric(ax, rows_by_policy, metric, ylabel, title, POLICIES)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out = FIGURES_DIR / "gavelfgd_combined_absolute.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – % improvement over Gavel
# ══════════════════════════════════════════════════════════════════════════════
def figure_improvement_over_gavel(rows_by_policy):
    baseline_rows = {r["lam"]: r for r in rows_by_policy.get("gavel", [])}
    if not baseline_rows:
        print("No gavel baseline – skipping Fig 2")
        return

    METRICS = [
        ("avg_jct",        "JCT reduction (%)",          "JCT Improvement over Gavel"),
        ("avg_frag_rate",  "Frag rate reduction (%)",     "Frag Rate Improvement over Gavel"),
        ("avg_frag_total", "Total frag reduction (%)",    "Total Frag Improvement over Gavel"),
    ]
    COMPARE = ["gavelfgd", "gavelfgd_combined"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "GavelFGD and GavelFGD+ Improvement over Gavel Baseline (%)\n"
        "(positive = better than Gavel)",
        fontsize=13, fontweight="bold"
    )

    for ax, (metric, ylabel, title) in zip(axes, METRICS):
        for pol in COMPARE:
            rows = rows_by_policy.get(pol, [])
            xs, ys = [], []
            for row in rows:
                base = baseline_rows.get(row["lam"])
                if base is None or base[metric] == 0:
                    continue
                pct = (base[metric] - row[metric]) / base[metric] * 100
                xs.append(row["avg_utilization"])
                ys.append(pct)
            if xs:
                st = POLICY_STYLE[pol]
                ax.plot(xs, ys,
                        color=st["color"], marker=st["marker"],
                        linestyle=st["ls"], linewidth=1.8, markersize=6,
                        label=st["label"])

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("GPU Utilization (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out = FIGURES_DIR / "gavelfgd_combined_vs_gavel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – % improvement of GavelFGD+ over GavelFGD
# ══════════════════════════════════════════════════════════════════════════════
def figure_improvement_over_gavelfgd(rows_by_policy):
    baseline_rows = {r["lam"]: r for r in rows_by_policy.get("gavelfgd", [])}
    combined_rows = rows_by_policy.get("gavelfgd_combined", [])
    if not baseline_rows or not combined_rows:
        print("Missing gavelfgd or gavelfgd_combined data – skipping Fig 3")
        return

    METRICS = [
        ("avg_jct",        "JCT reduction (%)",          "JCT: GavelFGD+ vs GavelFGD"),
        ("avg_frag_rate",  "Frag rate reduction (%)",     "Frag Rate: GavelFGD+ vs GavelFGD"),
        ("avg_frag_total", "Total frag reduction (%)",    "Total Frag: GavelFGD+ vs GavelFGD"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "GavelFGD+ Improvement over GavelFGD (%)\n"
        "(positive = GavelFGD+ is better)",
        fontsize=13, fontweight="bold"
    )

    st = POLICY_STYLE["gavelfgd_combined"]
    for ax, (metric, ylabel, title) in zip(axes, METRICS):
        xs, ys = [], []
        for row in combined_rows:
            base = baseline_rows.get(row["lam"])
            if base is None or base[metric] == 0:
                continue
            pct = (base[metric] - row[metric]) / base[metric] * 100
            xs.append(row["avg_utilization"])
            ys.append(pct)
        if xs:
            ax.bar(xs, ys, width=3,
                   color=st["color"], alpha=0.75, label="GavelFGD+")
            ax.plot(xs, ys,
                    color=st["color"], marker=st["marker"],
                    linestyle=st["ls"], linewidth=1.8, markersize=6)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("GPU Utilization (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "gavelfgd_combined_vs_gavelfgd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    records = load_results()
    if not records:
        print("No results found in", RESULTS_DIR)
        raise SystemExit(1)

    print(f"Loaded {len(records)} records")

    by_policy = defaultdict(list)
    for r in records:
        by_policy[classify(r["name"])].append(r)

    rows_by_policy = {pol: aggregate(recs) for pol, recs in by_policy.items()}

    # Print quick summary
    for pol in ["gavel", "gavelfgd", "gavelfgd_combined"]:
        rows = rows_by_policy.get(pol, [])
        print(f"\n{POLICY_STYLE[pol]['label']} ({len(rows)} load points):")
        for r in rows:
            print(f"  util={r['avg_utilization']:.1f}%  "
                  f"jct={r['avg_jct']:.0f}s  "
                  f"frag_rate={r['avg_frag_rate']:.2f}%")

    print("\nGenerating figures...")
    figure_absolute(rows_by_policy)
    figure_improvement_over_gavel(rows_by_policy)
    figure_improvement_over_gavelfgd(rows_by_policy)
    print("Done. Figures saved to:", FIGURES_DIR)
