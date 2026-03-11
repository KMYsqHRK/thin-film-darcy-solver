#!/usr/bin/env python3
"""
Time vs LiftForce plotter for multiple analysis cases.

Usage:
  python visualize_lift_force.py [root_dir]

Directory layouts supported:
  (A) Multiple cases under one root:
        root_dir/
          case_A/output/output_lift_forces.csv
          case_B/output/output_lift_forces.csv
          ...

  (B) Single output directory:
        root_dir/output_lift_forces.csv

  root_dir defaults to '../output' when not specified.

Output:
  lift_force.png  saved in root_dir
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.unicode_minus"] = False

FILENAME = "output_lift_forces.csv"


def find_csv_files(root: Path) -> list[tuple[str, Path]]:
    """
    Return list of (case_label, csv_path).

    Search order:
      1. root/<case>/<anything>/output_lift_forces.csv  (multi-case layout A)
      2. root/output_lift_forces.csv                    (single-case layout B)
    """
    # Layout A: any CSV two or more levels deep
    candidates = sorted(root.rglob(FILENAME))
    # Exclude files directly in root (those belong to layout B)
    deep = [p for p in candidates if p.parent != root]

    if deep:
        results = []
        for p in deep:
            # Use the first subdirectory under root as the case label
            relative = p.relative_to(root)
            label = relative.parts[0]
            results.append((label, p))
        return results

    # Layout B: single CSV directly in root
    single = root / FILENAME
    if single.exists():
        return [(root.name, single)]

    return []


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # t=0 時点の LiftForce のみ 0 に上書き
    df.loc[df.index[0], "LiftForce"] = 0.0
    return df


def plot(cases: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, df in cases:
        ax.plot(df["Time"], df["LiftForce"], label=label, linewidth=1.5)
    t = np.linspace(0, 20, 2000)  # 時間範囲は適宜調整
    P = np.sqrt(2 * 1.0e-8 * 49000 * t / 0.001) * 49000 * 10 * 0.5
    ax.plot(t, P, color='steelblue', linewidth=2, label="Theoretical")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Lift Force [N]", fontsize=12)
    ax.set_title("Time vs Lift Force", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(-0.1, 20)
    ax.set_ylim(0, 1200000)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Time vs LiftForce for multiple cases.")
    parser.add_argument("root_dir", nargs="?", default="../output",
                        help="Root directory to search (default: ../output)")
    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    entries = find_csv_files(root)
    if not entries:
        print(f"Error: no {FILENAME} found under {root}", file=sys.stderr)
        sys.exit(1)

    cases = []
    for label, csv_path in entries:
        df = load(csv_path)
        cases.append((label, df))
        print(f"  loaded: {csv_path}  ({len(df)} rows)")

    out_path = root / "lift_force.png"
    plot(cases, out_path)


if __name__ == "__main__":
    main()
