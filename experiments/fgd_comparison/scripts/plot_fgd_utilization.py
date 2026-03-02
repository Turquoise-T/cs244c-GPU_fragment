#!/usr/bin/env python3
"""
Plot utilization for FGD vs Strided from results_fgd.csv.

Usage:
  cd experiments/fgd_comparison
  python scripts/plot_fgd_utilization.py

Outputs:
  figures/fgd_vs_strided_util.png      # avg utilization vs jobs_per_hr
  figures/fgd_util_improvement.png     # relative improvement per jobs_per_hr
"""

import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results" / "results_fgd.csv"
FIG_DIR = ROOT / "figures"


def main():
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_CSV)

    # Group by (jobs_per_hr, placement_strategy) and average over seeds.
    grouped = (
        df.groupby(["jobs_per_hr", "placement_strategy"])
        .agg(
            util_mean=("utilization", "mean"),
            util_std=("utilization", "std"),
            jct_mean=("jct_sec", "mean"),
            jct_std=("jct_sec", "std"),
        )
        .reset_index()
    )

    def get_series(strategy: str):
        sub = grouped[grouped["placement_strategy"] == strategy].sort_values(
            "jobs_per_hr"
        )
        return (
            sub["jobs_per_hr"].values,
            sub["util_mean"].values,
            sub["util_std"].values,
        )

    x_strided, u_strided, u_strided_std = get_series("strided")
    x_fgd, u_fgd, u_fgd_std = get_series("fgd")

    # 1) Utilization vs jobs_per_hr
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x_strided,
        u_strided,
        yerr=u_strided_std,
        marker="s",
        linestyle="--",
        color="tab:red",
        label="Strided",
        capsize=3,
    )
    plt.errorbar(
        x_fgd,
        u_fgd,
        yerr=u_fgd_std,
        marker="o",
        linestyle="-",
        color="tab:green",
        label="FGD",
        capsize=3,
    )
    plt.xlabel("Job arrival rate (jobs/hr)")
    plt.ylabel("Average utilization")
    plt.title("FGD vs Strided: Utilization")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out1 = FIG_DIR / "fgd_vs_strided_util.png"
    plt.tight_layout()
    plt.savefig(out1)
    plt.close()

    # 2) Relative improvement: (u_fgd - u_strided) / u_strided
    # Align on common x values.
    util_improvement = []
    for jph in sorted(df["jobs_per_hr"].unique()):
        u_s = grouped[
            (grouped["jobs_per_hr"] == jph)
            & (grouped["placement_strategy"] == "strided")
        ]["util_mean"].values
        u_f = grouped[
            (grouped["jobs_per_hr"] == jph)
            & (grouped["placement_strategy"] == "fgd")
        ]["util_mean"].values
        if len(u_s) == 1 and len(u_f) == 1 and u_s[0] > 0:
            delta = (u_f[0] - u_s[0]) / u_s[0]
        else:
            delta = 0.0
        util_improvement.append((jph, delta))

    xs = [x for x, _ in util_improvement]
    ys = [y for _, y in util_improvement]

    plt.figure(figsize=(6, 4))
    plt.bar(xs, ys, width=0.5, color="tab:blue")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Job arrival rate (jobs/hr)")
    plt.ylabel("Utilization improvement (FGD vs Strided)")
    plt.title("FGD Utilization Improvement over Strided")
    plt.grid(True, axis="y", alpha=0.3)
    out2 = FIG_DIR / "fgd_util_improvement.png"
    plt.tight_layout()
    plt.savefig(out2)
    plt.close()

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()

