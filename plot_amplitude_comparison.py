"""
Amplitude Comparison Plots - Reference vs Cross-talk
=====================================================

Creates visualizations comparing LED amplitudes between reference
and cross-talk measurements for all three modules.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("crosstalk_comparison_results")
MODULES = ["Rear", "Middle", "Front"]


def create_amplitude_comparison_plots():
    """Create comprehensive amplitude comparison visualizations"""

    print("\n" + "=" * 80)
    print("AMPLITUDE COMPARISON PLOTS - REFERENCE VS CROSS-TALK")
    print("=" * 80)

    # Figure 1: Side-by-side comparison for all modules
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle(
        "LED Amplitude Comparison: Reference vs Cross-talk",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    all_data = []

    for module_idx, module in enumerate(MODULES):
        comparison_file = (
            RESULTS_DIR / f"{module}_comparison" / "detailed_comparison.csv"
        )

        if not comparison_file.exists():
            print(f"⚠️ File not found: {comparison_file}")
            continue

        # Load data
        df = pd.read_csv(comparison_file)
        all_data.append({"module": module, "df": df})

        print(f"\n📊 {module}: {len(df)} LEDs")
        print(
            f"   Reference Height: {df['Ref_Height'].mean():.4f} ± {df['Ref_Height'].std():.4f}"
        )
        print(
            f"   Cross-talk Height: {df['CT_Height'].mean():.4f} ± {df['CT_Height'].std():.4f}"
        )

        # Column 1: Reference vs Cross-talk amplitudes
        ax = axes[module_idx, 0]
        ax.plot(
            df["LED"],
            df["Ref_Height"],
            "b-",
            linewidth=1,
            alpha=0.7,
            label="Reference",
            markersize=1,
        )
        ax.plot(
            df["LED"],
            df["CT_Height"],
            "r-",
            linewidth=1,
            alpha=0.7,
            label="Cross-talk",
            markersize=1,
        )
        ax.set_xlabel("LED Number", fontsize=11)
        ax.set_ylabel("Pulse Height (Amplitude)", fontsize=11)
        ax.set_title(
            f"{module}: Pulse Height Comparison", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        stats_text = f'Ref Mean: {df["Ref_Height"].mean():.4f}\nCT Mean: {df["CT_Height"].mean():.4f}\nDiff: {(df["CT_Height"].mean() - df["Ref_Height"].mean()):.4f}'
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontfamily="monospace",
        )

        # Column 2: Difference plot
        ax = axes[module_idx, 1]
        difference = df["CT_Height"] - df["Ref_Height"]
        ax.plot(df["LED"], difference, "g-", linewidth=1, alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
        ax.fill_between(
            df["LED"],
            0,
            difference,
            where=(difference >= 0),
            alpha=0.3,
            color="green",
            label="Increase",
        )
        ax.fill_between(
            df["LED"],
            0,
            difference,
            where=(difference < 0),
            alpha=0.3,
            color="red",
            label="Decrease",
        )
        ax.set_xlabel("LED Number", fontsize=11)
        ax.set_ylabel("Height Difference (CT - Ref)", fontsize=11)
        ax.set_title(f"{module}: Absolute Difference", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"Mean: {difference.mean():.4f}\nStd: {difference.std():.4f}\nMax: {difference.max():.4f}\nMin: {difference.min():.4f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            fontfamily="monospace",
        )

        # Column 3: Peak values comparison
        ax = axes[module_idx, 2]
        ax.plot(
            df["LED"],
            df["Ref_Peak"],
            "b-",
            linewidth=1,
            alpha=0.7,
            label="Reference Peak",
            markersize=1,
        )
        ax.plot(
            df["LED"],
            df["CT_Peak"],
            "r-",
            linewidth=1,
            alpha=0.7,
            label="Cross-talk Peak",
            markersize=1,
        )
        ax.set_xlabel("LED Number", fontsize=11)
        ax.set_ylabel("Peak Value", fontsize=11)
        ax.set_title(f"{module}: Peak Value Comparison", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Ref Mean: {df["Ref_Peak"].mean():.4f}\nCT Mean: {df["CT_Peak"].mean():.4f}\nDiff: {(df["CT_Peak"].mean() - df["Ref_Peak"].mean()):.4f}'
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            fontfamily="monospace",
        )

    plt.tight_layout()
    output_file1 = RESULTS_DIR / "amplitude_comparison_all_modules.png"
    plt.savefig(output_file1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {output_file1}")

    # Figure 2: Overlay comparison - all modules together
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Combined Amplitude Comparison: All Modules", fontsize=16, fontweight="bold"
    )

    colors = ["blue", "green", "orange"]

    # Plot 1: All Reference heights
    ax = axes[0, 0]
    for idx, data in enumerate(all_data):
        module = data["module"]
        df = data["df"]
        # Subsample for clarity
        step = max(1, len(df) // 1000)
        ax.plot(
            df["LED"][::step],
            df["Ref_Height"][::step],
            color=colors[idx],
            alpha=0.6,
            linewidth=1.5,
            label=f"{module} Ref",
        )
    ax.set_xlabel("LED Number", fontsize=12)
    ax.set_ylabel("Pulse Height", fontsize=12)
    ax.set_title("Reference Measurements - All Modules", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: All Cross-talk heights
    ax = axes[0, 1]
    for idx, data in enumerate(all_data):
        module = data["module"]
        df = data["df"]
        step = max(1, len(df) // 1000)
        ax.plot(
            df["LED"][::step],
            df["CT_Height"][::step],
            color=colors[idx],
            alpha=0.6,
            linewidth=1.5,
            label=f"{module} CT",
        )
    ax.set_xlabel("LED Number", fontsize=12)
    ax.set_ylabel("Pulse Height", fontsize=12)
    ax.set_title(
        "Cross-talk Measurements - All Modules", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Average values comparison
    ax = axes[1, 0]
    modules_names = [d["module"] for d in all_data]
    ref_means = [d["df"]["Ref_Height"].mean() for d in all_data]
    ct_means = [d["df"]["CT_Height"].mean() for d in all_data]

    x = np.arange(len(modules_names))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, ref_means, width, label="Reference", alpha=0.8, color="blue"
    )
    bars2 = ax.bar(
        x + width / 2, ct_means, width, label="Cross-talk", alpha=0.8, color="red"
    )

    ax.set_xlabel("Module", fontsize=12)
    ax.set_ylabel("Mean Pulse Height", fontsize=12)
    ax.set_title("Average Pulse Height by Module", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modules_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Plot 4: Difference statistics
    ax = axes[1, 1]
    diff_means = [
        (d["df"]["CT_Height"] - d["df"]["Ref_Height"]).mean() for d in all_data
    ]
    diff_stds = [(d["df"]["CT_Height"] - d["df"]["Ref_Height"]).std() for d in all_data]

    ax.bar(modules_names, diff_means, alpha=0.8, color="green")
    ax.errorbar(
        modules_names, diff_means, yerr=diff_stds, fmt="none", color="black", capsize=5
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Module", fontsize=12)
    ax.set_ylabel("Mean Height Difference (CT - Ref)", fontsize=12)
    ax.set_title("Average Difference by Module", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (mean, std) in enumerate(zip(diff_means, diff_stds)):
        ax.text(
            i,
            mean + std,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_file2 = RESULTS_DIR / "amplitude_comparison_combined.png"
    plt.savefig(output_file2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_file2}")

    # Figure 3: Detailed zoom views for each module
    for data in all_data:
        module = data["module"]
        df = data["df"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"{module} Module: Detailed Amplitude Analysis",
            fontsize=16,
            fontweight="bold",
        )

        # Select 3 regions for detailed view
        total_leds = len(df)
        regions = [
            (0, min(200, total_leds), "Start"),
            (total_leds // 2 - 100, total_leds // 2 + 100, "Middle"),
            (max(0, total_leds - 200), total_leds, "End"),
        ]

        for idx, (start, end, label) in enumerate(regions[:3]):
            if idx >= 3:
                break

            ax = axes.flat[idx]
            region_df = df.iloc[start:end]

            ax.plot(
                region_df["LED"],
                region_df["Ref_Height"],
                "b-o",
                linewidth=2,
                markersize=4,
                alpha=0.7,
                label="Reference",
            )
            ax.plot(
                region_df["LED"],
                region_df["CT_Height"],
                "r-s",
                linewidth=2,
                markersize=4,
                alpha=0.7,
                label="Cross-talk",
            )

            ax.set_xlabel("LED Number", fontsize=11)
            ax.set_ylabel("Pulse Height", fontsize=11)
            ax.set_title(
                f'{label} Region: LEDs {region_df["LED"].iloc[0]}-{region_df["LED"].iloc[-1]}',
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Statistics box
            ref_mean = region_df["Ref_Height"].mean()
            ct_mean = region_df["CT_Height"].mean()
            diff = ct_mean - ref_mean
            stats_text = f"Ref: {ref_mean:.4f}\nCT: {ct_mean:.4f}\nΔ: {diff:.4f}\n%Δ: {(diff/ref_mean*100):.2f}%"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                fontfamily="monospace",
            )

        # Last plot: Correlation scatter
        ax = axes.flat[3]
        ax.scatter(df["Ref_Height"], df["CT_Height"], alpha=0.3, s=10)

        # Add perfect correlation line
        min_val = min(df["Ref_Height"].min(), df["CT_Height"].min())
        max_val = max(df["Ref_Height"].max(), df["CT_Height"].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect correlation",
        )

        # Calculate and show correlation
        correlation = df["Ref_Height"].corr(df["CT_Height"])

        ax.set_xlabel("Reference Height", fontsize=11)
        ax.set_ylabel("Cross-talk Height", fontsize=11)
        ax.set_title(
            f"Correlation Plot (R={correlation:.4f})", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = RESULTS_DIR / f"{module}_detailed_amplitude_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {output_file}")

    print("\n" + "=" * 80)
    print("AMPLITUDE COMPARISON PLOTS COMPLETE")
    print("=" * 80)


def create_csv_summary():
    """Create a simplified CSV summary for each module"""

    print("\n" + "=" * 80)
    print("CREATING SIMPLIFIED CSV SUMMARIES")
    print("=" * 80)

    for module in MODULES:
        comparison_file = (
            RESULTS_DIR / f"{module}_comparison" / "detailed_comparison.csv"
        )

        if not comparison_file.exists():
            continue

        df = pd.read_csv(comparison_file)

        # Create simplified summary
        summary_df = df[
            [
                "LED",
                "Ref_Height",
                "CT_Height",
                "Height_Diff",
                "Height_Change_%",
                "Ref_Peak",
                "CT_Peak",
                "Peak_Change_%",
                "Ref_SNR",
                "CT_SNR",
            ]
        ].copy()

        # Round for readability
        for col in [
            "Ref_Height",
            "CT_Height",
            "Height_Diff",
            "Ref_Peak",
            "CT_Peak",
            "Ref_SNR",
            "CT_SNR",
        ]:
            summary_df[col] = summary_df[col].round(4)

        for col in ["Height_Change_%", "Peak_Change_%"]:
            summary_df[col] = summary_df[col].round(2)

        output_file = RESULTS_DIR / f"{module}_amplitude_summary.csv"
        summary_df.to_csv(output_file, index=False)

        print(f"✅ {module}: {len(summary_df)} LEDs → {output_file.name}")


if __name__ == "__main__":
    create_amplitude_comparison_plots()
    create_csv_summary()

    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS AND SUMMARIES CREATED")
    print("=" * 80)
    print(f"\nFiles saved in: {RESULTS_DIR}")
