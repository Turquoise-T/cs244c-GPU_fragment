#!/usr/bin/env python3
"""
Download Alibaba GPU Cluster Trace v2023 CSVs.

Downloads the two required CSV files from the Alibaba clusterdata repository:
- openb_node_list_gpu_node.csv  (GPU node specifications)
- openb_pod_list_default.csv    (pod/job specifications with timestamps)

Usage:
    python download_traces.py [--output-dir data/]
"""

import argparse
import os
import sys
import urllib.request

BASE_URL = (
    "https://raw.githubusercontent.com/alibaba/clusterdata/master/"
    "cluster-trace-gpu-v2023/csv"
)

FILES = [
    "openb_node_list_gpu_node.csv",
    "openb_pod_list_default.csv",
]


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f"  Downloading {os.path.basename(dest_path)}...")
    print(f"    URL: {url}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    size = os.path.getsize(dest_path)
    print(f"\n    Done: {size / (1024 * 1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download Alibaba GPU trace v2023")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "csv"),
        help="Directory to save CSV files (default: data/csv/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Downloading Alibaba cluster-trace-gpu-v2023 CSVs...")
    print(f"Output directory: {args.output_dir}\n")

    for filename in FILES:
        dest = os.path.join(args.output_dir, filename)
        if os.path.exists(dest):
            size = os.path.getsize(dest)
            print(f"  {filename} already exists ({size / (1024 * 1024):.1f} MB), skipping.")
            continue
        url = f"{BASE_URL}/{filename}"
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"\n  ERROR downloading {filename}: {e}", file=sys.stderr)
            print("  You may need to download manually from:", file=sys.stderr)
            print(f"    {url}", file=sys.stderr)
            return 1

    print("\nAll files downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
