"""
Cross-Talk Change Distribution Analysis
========================================

Analyzes the distribution of changes and creates histograms showing
how many LEDs changed at each percentage level (1%, 2%, 3%, etc.)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Base directory
RESULTS_DIR = Path("crosstalk_comparison_results")

# Modules to analyze
MODULES = ["Rear", "Middle", "Front"]


def analyze_change_distribution(module_name):
    """Analyze the distribution of changes for a specific module"""

    comparison_file = (
        RESULTS_DIR / f"{module_name}_comparison" / "detailed_comparison.csv"
    )

    if not comparison_file.exists():
        print(f"❌ File not found: {comparison_file}")
        return None

    # Load data
    df = pd.read_csv(comparison_file)

    print(f"\n{'='*80}")
    print(f"CHANGE DISTRIBUTION ANALYSIS: {module_name}")
    print(f"{'='*80}")
    print(f"Total LEDs analyzed: {len(df)}")

    # Analyze height changes
    height_changes = df["Height_Change_%"].abs()
    peak_changes = df["Peak_Change_%"].abs()

    # Count LEDs at different percentage thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    print(f"\n📊 HEIGHT CHANGE DISTRIBUTION:")
    print(f"{'Threshold':<12} {'Count':<10} {'Percentage':<15} {'Cumulative %'}")
    print("-" * 60)

    height_stats = []
    cumulative_count = 0

    for threshold in thresholds:
        count = len(height_changes[height_changes > threshold])
        percentage = (count / len(df)) * 100
        cumulative_count = count
        height_stats.append(
            {
                "Threshold": f">{threshold}%",
                "Count": count,
                "Percentage": percentage,
                "Cumulative": percentage,
            }
        )
        print(
            f">{threshold:>5.1f}%     {count:<10} {percentage:<14.2f}% {percentage:>6.2f}%"
        )

    print(f"\n📊 PEAK CHANGE DISTRIBUTION:")
    print(f"{'Threshold':<12} {'Count':<10} {'Percentage':<15} {'Cumulative %'}")
    print("-" * 60)

    peak_stats = []

    for threshold in thresholds:
        count = len(peak_changes[peak_changes > threshold])
        percentage = (count / len(df)) * 100
        peak_stats.append(
            {
                "Threshold": f">{threshold}%",
                "Count": count,
                "Percentage": percentage,
                "Cumulative": percentage,
            }
        )
        print(
            f">{threshold:>5.1f}%     {count:<10} {percentage:<14.2f}% {percentage:>6.2f}%"
        )

    # Detailed binning for histogram (0.1% bins up to 5%, then larger bins)
    print(f"\n📊 DETAILED BINNING (Height Changes):")
    print(f"{'Range':<20} {'Count':<10} {'Percentage'}")
    print("-" * 50)

    bins = [
        (0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 1.0),
        (1.0, 1.5),
        (1.5, 2.0),
        (2.0, 2.5),
        (2.5, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
        (5.0, 10.0),
    ]

    for low, high in bins:
        count = len(height_changes[(height_changes >= low) & (height_changes < high)])
        percentage = (count / len(df)) * 100
        print(f"{low:>5.1f}%-{high:<5.1f}%     {count:<10} {percentage:>6.2f}%")

    return {
        "module": module_name,
        "df": df,
        "height_changes": height_changes,
        "peak_changes": peak_changes,
        "height_stats": height_stats,
        "peak_stats": peak_stats,
    }


def create_distribution_visualizations(all_results):
    """Create comprehensive distribution visualizations"""

    print(f"\n{'='*80}")
    print("CREATING DISTRIBUTION VISUALIZATIONS")
    print(f"{'='*80}")

    # Figure 1: Histograms for each module
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "Cross-Talk Change Distribution by Module", fontsize=16, fontweight="bold"
    )

    for idx, result in enumerate(all_results):
        if result is None:
            continue

        module = result["module"]
        height_changes = result["height_changes"]
        peak_changes = result["peak_changes"]

        # Height change histogram
        ax = axes[idx, 0]
        ax.hist(
            height_changes,
            bins=50,
            range=(0, 5),
            edgecolor="black",
            alpha=0.7,
            color="blue",
        )
        ax.axvline(
            x=1.0, color="red", linestyle="--", linewidth=2, label="1% threshold"
        )
        ax.axvline(
            x=2.0, color="orange", linestyle="--", linewidth=2, label="2% threshold"
        )
        ax.axvline(
            x=3.0, color="green", linestyle="--", linewidth=2, label="3% threshold"
        )
        ax.set_xlabel("Absolute Height Change (%)")
        ax.set_ylabel("Number of LEDs")
        ax.set_title(f"{module}: Height Change Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = height_changes.mean()
        median_val = height_changes.median()
        max_val = height_changes.max()
        above_1pct = len(height_changes[height_changes > 1.0])
        above_2pct = len(height_changes[height_changes > 2.0])
        above_3pct = len(height_changes[height_changes > 3.0])

        stats_text = f"Mean: {mean_val:.2f}%\nMedian: {median_val:.2f}%\nMax: {max_val:.2f}%\n>1%: {above_1pct}\n>2%: {above_2pct}\n>3%: {above_3pct}"
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
            fontfamily="monospace",
        )

        # Peak change histogram
        ax = axes[idx, 1]
        ax.hist(
            peak_changes,
            bins=50,
            range=(0, 5),
            edgecolor="black",
            alpha=0.7,
            color="green",
        )
        ax.axvline(
            x=1.0, color="red", linestyle="--", linewidth=2, label="1% threshold"
        )
        ax.axvline(
            x=2.0, color="orange", linestyle="--", linewidth=2, label="2% threshold"
        )
        ax.axvline(
            x=3.0, color="green", linestyle="--", linewidth=2, label="3% threshold"
        )
        ax.set_xlabel("Absolute Peak Change (%)")
        ax.set_ylabel("Number of LEDs")
        ax.set_title(f"{module}: Peak Change Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = peak_changes.mean()
        median_val = peak_changes.median()
        max_val = peak_changes.max()
        above_1pct = len(peak_changes[peak_changes > 1.0])
        above_2pct = len(peak_changes[peak_changes > 2.0])
        above_3pct = len(peak_changes[peak_changes > 3.0])

        stats_text = f"Mean: {mean_val:.2f}%\nMedian: {median_val:.2f}%\nMax: {max_val:.2f}%\n>1%: {above_1pct}\n>2%: {above_2pct}\n>3%: {above_3pct}"
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            fontsize=9,
            fontfamily="monospace",
        )

    plt.tight_layout()
    output_file1 = RESULTS_DIR / "distribution_histograms_by_module.png"
    plt.savefig(output_file1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_file1}")

    # Figure 2: Combined comparison bar chart
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Prepare data for bar charts
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    threshold_labels = [f">{t}%" for t in thresholds]

    # Height changes
    ax = axes[0]
    x = np.arange(len(thresholds))
    width = 0.25

    for idx, result in enumerate(all_results):
        if result is None:
            continue
        module = result["module"]
        height_changes = result["height_changes"]
        counts = [len(height_changes[height_changes > t]) for t in thresholds]
        percentages = [(c / len(result["df"])) * 100 for c in counts]
        ax.bar(x + idx * width, percentages, width, label=module, alpha=0.8)

    ax.set_xlabel("Change Threshold", fontsize=12)
    ax.set_ylabel("Percentage of LEDs (%)", fontsize=12)
    ax.set_title(
        "Height Changes: Percentage of LEDs Exceeding Each Threshold",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(threshold_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Peak changes
    ax = axes[1]
    for idx, result in enumerate(all_results):
        if result is None:
            continue
        module = result["module"]
        peak_changes = result["peak_changes"]
        counts = [len(peak_changes[peak_changes > t]) for t in thresholds]
        percentages = [(c / len(result["df"])) * 100 for c in counts]
        ax.bar(x + idx * width, percentages, width, label=module, alpha=0.8)

    ax.set_xlabel("Change Threshold", fontsize=12)
    ax.set_ylabel("Percentage of LEDs (%)", fontsize=12)
    ax.set_title(
        "Peak Changes: Percentage of LEDs Exceeding Each Threshold",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(threshold_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file2 = RESULTS_DIR / "distribution_comparison_bar_chart.png"
    plt.savefig(output_file2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_file2}")

    # Figure 3: Cumulative distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Height changes cumulative
    ax = axes[0]
    for result in all_results:
        if result is None:
            continue
        module = result["module"]
        height_changes = result["height_changes"]
        sorted_changes = np.sort(height_changes)
        cumulative = np.arange(1, len(sorted_changes) + 1) / len(sorted_changes) * 100
        ax.plot(sorted_changes, cumulative, linewidth=2, label=module, alpha=0.8)

    ax.axvline(
        x=1.0, color="red", linestyle="--", linewidth=2, alpha=0.5, label="1% threshold"
    )
    ax.axvline(
        x=2.0,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="2% threshold",
    )
    ax.axvline(
        x=3.0,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="3% threshold",
    )
    ax.set_xlabel("Absolute Height Change (%)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage of LEDs (%)", fontsize=12)
    ax.set_title(
        "Cumulative Distribution: Height Changes", fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Peak changes cumulative
    ax = axes[1]
    for result in all_results:
        if result is None:
            continue
        module = result["module"]
        peak_changes = result["peak_changes"]
        sorted_changes = np.sort(peak_changes)
        cumulative = np.arange(1, len(sorted_changes) + 1) / len(sorted_changes) * 100
        ax.plot(sorted_changes, cumulative, linewidth=2, label=module, alpha=0.8)

    ax.axvline(
        x=1.0, color="red", linestyle="--", linewidth=2, alpha=0.5, label="1% threshold"
    )
    ax.axvline(
        x=2.0,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="2% threshold",
    )
    ax.axvline(
        x=3.0,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="3% threshold",
    )
    ax.set_xlabel("Absolute Peak Change (%)", fontsize=12)
    ax.set_ylabel("Cumulative Percentage of LEDs (%)", fontsize=12)
    ax.set_title(
        "Cumulative Distribution: Peak Changes", fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file3 = RESULTS_DIR / "cumulative_distribution.png"
    plt.savefig(output_file3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_file3}")


def create_summary_table(all_results):
    """Create a summary table with change statistics"""

    output_file = RESULTS_DIR / "change_distribution_summary.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-TALK CHANGE DISTRIBUTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            if result is None:
                continue

            module = result["module"]
            df = result["df"]
            height_changes = result["height_changes"]
            peak_changes = result["peak_changes"]

            f.write(f"\n{module.upper()} MODULE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total LEDs: {len(df)}\n\n")

            f.write("HEIGHT CHANGES:\n")
            thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
            for threshold in thresholds:
                count = len(height_changes[height_changes > threshold])
                percentage = (count / len(df)) * 100
                f.write(
                    f"  >{threshold:>4.1f}%: {count:>5} LEDs ({percentage:>5.2f}%)\n"
                )

            f.write("\nPEAK CHANGES:\n")
            for threshold in thresholds:
                count = len(peak_changes[peak_changes > threshold])
                percentage = (count / len(df)) * 100
                f.write(
                    f"  >{threshold:>4.1f}%: {count:>5} LEDs ({percentage:>5.2f}%)\n"
                )

            f.write("\nSTATISTICS:\n")
            f.write(
                f"  Height - Mean: {height_changes.mean():.3f}%, Median: {height_changes.median():.3f}%, Max: {height_changes.max():.3f}%\n"
            )
            f.write(
                f"  Peak   - Mean: {peak_changes.mean():.3f}%, Median: {peak_changes.median():.3f}%, Max: {peak_changes.max():.3f}%\n"
            )

        f.write("\n" + "=" * 80 + "\n")

    print(f"✅ Saved summary table: {output_file}")


def main():
    """Main analysis function"""

    print("=" * 80)
    print("CROSS-TALK CHANGE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Analyze each module
    all_results = []
    for module in MODULES:
        result = analyze_change_distribution(module)
        all_results.append(result)

    # Create visualizations
    create_distribution_visualizations(all_results)

    # Create summary table
    create_summary_table(all_results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
