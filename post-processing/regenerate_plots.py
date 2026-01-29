#!/usr/bin/env python3
"""Regenerate global/ROI plots from CSVs as PDF and PNG without rerunning video processing."""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

COLOR_RAW = "#000000"
COLOR_FAST = "#1f77b4"
COLOR_DEEPCAD = "#ff7f0e"
COLOR_TED = "#2ca02c"


def plot_global(df, out_pdf, out_png, title):
    plt.figure(figsize=(10, 4))
    plt.plot(df["time_ms"], df["raw"], label="Raw", color=COLOR_RAW, linewidth=1.5)
    plt.plot(df["time_ms"], df["fast"], label="FAST", color=COLOR_FAST, linewidth=1.5)
    plt.plot(df["time_ms"], df["deepcadrt"], label="DeepCAD-RT", color=COLOR_DEEPCAD, linewidth=1.5)
    plt.plot(df["time_ms"], df["ted"], label="TeD", color=COLOR_TED, linewidth=1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("ΔF/F0")
    plt.title(title)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_roi(df, out_pdf, out_png, title):
    plt.figure(figsize=(10, 4))
    for label, color in [("raw", COLOR_RAW), ("fast", COLOR_FAST), ("deepcadrt", COLOR_DEEPCAD), ("ted", COLOR_TED)]:
        grouped = df.groupby(["time_ms"])[label]
        mean = grouped.mean().values
        std = grouped.std().values
        time = grouped.mean().index.values
        plt.plot(time, mean, label=label.upper() if label != "deepcadrt" else "DeepCAD-RT", color=color, linewidth=1.5)
        plt.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)
    plt.xlabel("Time (ms)")
    plt.ylabel("ΔF/F0")
    plt.title(title)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    movie_dirs = [
        os.path.join(args.output_root, d)
        for d in os.listdir(args.output_root)
        if d.startswith("FGF2-G3_")
    ]

    for d in movie_dirs:
        g_csv = os.path.join(d, "global_dff.csv")
        r_csv = os.path.join(d, "roi_dff.csv")
        if not os.path.exists(g_csv) or not os.path.exists(r_csv):
            continue
        g = pd.read_csv(g_csv)
        r = pd.read_csv(r_csv)
        name = os.path.basename(d)
        plot_global(g, os.path.join(d, "global_dff.pdf"), os.path.join(d, "global_dff.png"), f"Global ΔF/F0 - {name}")
        plot_roi(r, os.path.join(d, "roi_dff.pdf"), os.path.join(d, "roi_dff.png"), f"ROI ΔF/F0 (mean ± std) - {name}")


if __name__ == "__main__":
    main()
