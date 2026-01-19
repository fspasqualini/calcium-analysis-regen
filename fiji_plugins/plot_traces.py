#!/usr/bin/env python3
"""
plot_traces.py - Publication-quality calcium trace visualization

Reads CSV files exported by the CalciumAnalysis_Wizard ImageJ plugin
and creates matplotlib figures with:
- 3-panel layout (Magenta | Cyan | Background)
- Individual traces as thin semi-transparent lines
- Mean curve as bold solid line
- Shaded confidence interval (±SEM or ±SD)

Usage:
    python plot_traces.py --input calcium_traces_dff.csv --output figure.png
    python plot_traces.py --input calcium_traces_dff.csv --output figure.pdf --ci sem
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_traces(csv_path):
    """Load traces from CSV and separate by group."""
    df = pd.read_csv(csv_path)
    
    groups = {
        "Magenta": {"traces": [], "names": []},
        "Cyan": {"traces": [], "names": []},
        "BG": {"traces": [], "names": []}
    }
    
    # Extract individual trace columns
    for col in df.columns:
        if col == "Frame":
            continue
        if col.startswith("Magenta") and "_" in col and not any(x in col for x in ["mean", "std", "min", "max"]):
            groups["Magenta"]["traces"].append(df[col].values)
            groups["Magenta"]["names"].append(col)
        elif col.startswith("Cyan") and "_" in col and not any(x in col for x in ["mean", "std", "min", "max"]):
            groups["Cyan"]["traces"].append(df[col].values)
            groups["Cyan"]["names"].append(col)
        elif col.startswith("BG") and "_" in col and not any(x in col for x in ["mean", "std", "min", "max"]):
            groups["BG"]["traces"].append(df[col].values)
            groups["BG"]["names"].append(col)
    
    # Extract pre-computed stats if available
    for group_key in groups.keys():
        mean_col = f"{group_key}_mean"
        std_col = f"{group_key}_std"
        if mean_col in df.columns:
            groups[group_key]["mean"] = df[mean_col].values
        if std_col in df.columns:
            groups[group_key]["std"] = df[std_col].values
    
    return df["Frame"].values, groups


def calculate_stats(traces, ci_type="sem"):
    """Calculate mean and confidence interval from traces."""
    if not traces:
        return None, None, None
    
    traces_array = np.array(traces)
    mean = np.mean(traces_array, axis=0)
    std = np.std(traces_array, axis=0, ddof=1)
    
    if ci_type == "sem":
        ci = std / np.sqrt(len(traces))
    else:  # sd
        ci = std
    
    return mean, ci, std


def plot_group_panel(ax, frames, traces, mean, ci, group_config, ci_type="sem"):
    """Plot a single group panel with traces, mean, and shaded CI."""
    color = group_config["color"]
    light_color = group_config["light_color"]
    title = group_config["title"]
    
    # Plot individual traces (thin, semi-transparent)
    for trace in traces:
        ax.plot(frames, trace, color=light_color, alpha=0.4, linewidth=0.8)
    
    if mean is not None:
        # Plot shaded confidence interval
        lower = mean - ci
        upper = mean + ci
        ax.fill_between(frames, lower, upper, color=color, alpha=0.3, 
                       label=f"±{ci_type.upper()}")
        
        # Plot mean as bold line
        ax.plot(frames, mean, color=color, linewidth=2.5, label="Mean")
    
    ax.set_title(f"{title} (n={len(traces)})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("F/F₀", fontsize=10)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_figure(frames, groups, output_path, ci_type="sem", dpi=300):
    """Create the 3-panel figure."""
    
    group_configs = {
        "Magenta": {
            "title": "S/G2/M Regenerating",
            "color": "#E91E63",  # Pink/Magenta
            "light_color": "#F8BBD9"
        },
        "Cyan": {
            "title": "G0/G1 Quiescent", 
            "color": "#00BCD4",  # Cyan
            "light_color": "#B2EBF2"
        },
        "BG": {
            "title": "Background",
            "color": "#9E9E9E",  # Gray
            "light_color": "#E0E0E0"
        }
    }
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Calcium Transient Analysis", fontsize=14, fontweight='bold', y=1.02)
    
    for idx, (group_key, config) in enumerate(group_configs.items()):
        ax = axes[idx]
        group_data = groups[group_key]
        traces = group_data.get("traces", [])
        
        if not traces:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f"{config['title']} (n=0)", fontsize=12)
            continue
        
        # Use pre-computed stats if available, otherwise calculate
        if "mean" in group_data and "std" in group_data:
            mean = group_data["mean"]
            std = group_data["std"]
            if ci_type == "sem":
                ci = std / np.sqrt(len(traces))
            else:
                ci = std
        else:
            mean, ci, std = calculate_stats(traces, ci_type)
        
        plot_group_panel(ax, frames, traces, mean, ci, config, ci_type)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    # Also show if running interactively
    try:
        plt.show()
    except:
        pass
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Create publication-quality calcium trace figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_traces.py --input calcium_traces_dff.csv --output figure.png
    python plot_traces.py --input calcium_traces_dff.csv --output figure.pdf --ci sd
    python plot_traces.py -i traces.csv -o plot.png --dpi 600
        """
    )
    parser.add_argument("-i", "--input", required=True, 
                       help="Input CSV file (dF/F0 normalized traces)")
    parser.add_argument("-o", "--output", required=True,
                       help="Output figure path (png, pdf, svg, etc.)")
    parser.add_argument("--ci", choices=["sem", "sd"], default="sem",
                       help="Confidence interval type: sem (default) or sd")
    parser.add_argument("--dpi", type=int, default=300,
                       help="Figure DPI for raster formats (default: 300)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"Loading traces from: {args.input}")
    frames, groups = load_traces(args.input)
    
    print(f"Found: {len(groups['Magenta']['traces'])} Magenta, "
          f"{len(groups['Cyan']['traces'])} Cyan, "
          f"{len(groups['BG']['traces'])} Background traces")
    
    create_figure(frames, groups, args.output, ci_type=args.ci, dpi=args.dpi)
    
    return 0


if __name__ == "__main__":
    exit(main())
