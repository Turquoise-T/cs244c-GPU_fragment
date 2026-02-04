"""
Plot Figure 7(a) from CSV results file.

Usage:
    python plot_results.py <csv_path> [output_path]

Example:
    python plot_results.py figure7a_results.csv figure7a.png
"""

import sys
import os

from experiment import load_results_from_csv, plot_figure7a


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <csv_path> [output_path]")
        print("Example: python plot_results.py figure7a_results.csv figure7a.png")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # Optional output path
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading results from {csv_path}...")
    results = load_results_from_csv(csv_path)

    print(f"Loaded {len(results)} schedulers:")
    for name, result_list in results.items():
        num_points = sum(len(r.fragmentation_curve) for r in result_list)
        print(f"  {name}: {len(result_list)} runs, {num_points} data points")

    print("\nPlotting...")
    plot_figure7a(results, output_path)


if __name__ == "__main__":
    main()
