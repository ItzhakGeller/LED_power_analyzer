"""
Simplified Pulse Height Analysis - Focus on Amplitude Only
===========================================================

Creates simplified CSV files and visualizations focusing only on
pulse height (peak-valley), without peak value information.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("crosstalk_comparison_results")
MODULES = ["Rear", "Middle", "Front"]


def create_simplified_pulse_height_csv():
    """Create simplified CSV with only pulse height data"""

    print("\n" + "=" * 80)
    print("CREATING SIMPLIFIED PULSE HEIGHT CSV FILES")
    print("=" * 80)

    for module in MODULES:
        comparison_file = (
            RESULTS_DIR / f"{module}_comparison" / "detailed_comparison.csv"
        )

        if not comparison_file.exists():
            print(f"⚠️ File not found: {comparison_file}")
            continue

        # Load full data
        df = pd.read_csv(comparison_file)

        # Create simplified version - ONLY pulse height data
        simplified_df = pd.DataFrame(
            {
                "LED": df["LED"],
                "Ref_Height": df["Ref_Height"].round(4),
                "CT_Height": df["CT_Height"].round(4),
                "Height_Diff": df["Height_Diff"].round(4),
                "Height_Change_%": df["Height_Change_%"].round(2),
                "Ref_SNR": df["Ref_SNR"].round(2),
                "CT_SNR": df["CT_SNR"].round(2),
            }
        )

        # Save simplified CSV
        output_file = RESULTS_DIR / f"{module}_pulse_height_only.csv"
        simplified_df.to_csv(output_file, index=False)

        print(f"✅ {module}: {len(simplified_df)} LEDs")
        print(f"   Mean Ref Height: {df['Ref_Height'].mean():.4f}")
        print(f"   Mean CT Height: {df['CT_Height'].mean():.4f}")
        print(f"   Mean Change: {df['Height_Change_%'].mean():.2f}%")
        print(f"   Saved: {output_file.name}")

    print("\n" + "=" * 80)


def create_pulse_height_only_plots():
    """Create visualizations focusing only on pulse height"""

    print("\n" + "=" * 80)
    print("CREATING PULSE HEIGHT VISUALIZATIONS")
    print("=" * 80)

    # Figure 1: Main comparison - pulse height only
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle(
        "Pulse Height Analysis: Reference vs Cross-talk\n(Pulse Height = Peak - Valley)",
        fontsize=16,
        fontweight="bold",
    )

    all_data = []

    for module_idx, module in enumerate(MODULES):
        comparison_file = (
            RESULTS_DIR / f"{module}_comparison" / "detailed_comparison.csv"
        )

        if not comparison_file.exists():
            continue

        df = pd.read_csv(comparison_file)
        all_data.append({"module": module, "df": df})

        # Column 1: Pulse height comparison
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
        ax.set_ylabel("Pulse Height (Peak-Valley)", fontsize=11)
        ax.set_title(
            f"{module}: Pulse Height Comparison", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Statistics box
        ref_mean = df["Ref_Height"].mean()
        ct_mean = df["CT_Height"].mean()
        diff_mean = df["Height_Diff"].mean()
        change_pct = df["Height_Change_%"].mean()

        stats_text = (
            f"Ref Mean: {ref_mean:.4f}\n"
            f"CT Mean: {ct_mean:.4f}\n"
            f"Δ Mean: {diff_mean:.4f}\n"
            f"Change: {change_pct:.2f}%"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
            fontfamily="monospace",
        )

        # Column 2: Percentage change
        ax = axes[module_idx, 1]
        ax.plot(df["LED"], df["Height_Change_%"], "g-", linewidth=1, alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
        ax.axhline(
            y=1, color="orange", linestyle=":", linewidth=1, alpha=0.7, label="±1%"
        )
        ax.axhline(y=-1, color="orange", linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(
            y=2, color="yellow", linestyle=":", linewidth=1, alpha=0.5, label="±2%"
        )
        ax.axhline(y=-2, color="yellow", linestyle=":", linewidth=1, alpha=0.5)

        ax.set_xlabel("LED Number", fontsize=11)
        ax.set_ylabel("Height Change (%)", fontsize=11)
        ax.set_title(
            f"{module}: Percentage Change in Pulse Height",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 5)

        # Statistics
        above_1pct = len(df[abs(df["Height_Change_%"]) > 1.0])
        above_2pct = len(df[abs(df["Height_Change_%"]) > 2.0])
        above_3pct = len(df[abs(df["Height_Change_%"]) > 3.0])

        stats_text = (
            f"Mean: {change_pct:.2f}%\n"
            f'Std: {df["Height_Change_%"].std():.2f}%\n'
            f">1%: {above_1pct} ({above_1pct/len(df)*100:.1f}%)\n"
            f">2%: {above_2pct} ({above_2pct/len(df)*100:.1f}%)\n"
            f">3%: {above_3pct} ({above_3pct/len(df)*100:.1f}%)"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.9),
            fontfamily="monospace",
        )

    plt.tight_layout()
    output_file1 = RESULTS_DIR / "pulse_height_comparison_clean.png"
    plt.savefig(output_file1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_file1}")

    # Figure 2: Combined summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Pulse Height Summary - All Modules", fontsize=16, fontweight="bold")

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # Blue, Green, Orange

    # Plot 1: Average pulse heights
    ax = axes[0, 0]
    modules_names = [d["module"] for d in all_data]
    ref_means = [d["df"]["Ref_Height"].mean() for d in all_data]
    ct_means = [d["df"]["CT_Height"].mean() for d in all_data]

    x = np.arange(len(modules_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ref_means, width, label="Reference", alpha=0.8)
    bars2 = ax.bar(x + width / 2, ct_means, width, label="Cross-talk", alpha=0.8)

    ax.set_xlabel("Module", fontsize=12)
    ax.set_ylabel("Mean Pulse Height", fontsize=12)
    ax.set_title("Average Pulse Height by Module", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modules_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Plot 2: Mean change percentage
    ax = axes[0, 1]
    change_means = [d["df"]["Height_Change_%"].mean() for d in all_data]
    change_stds = [d["df"]["Height_Change_%"].std() for d in all_data]

    bars = ax.bar(modules_names, change_means, color=colors, alpha=0.8)
    ax.errorbar(
        modules_names,
        change_means,
        yerr=change_stds,
        fmt="none",
        color="black",
        capsize=8,
        linewidth=2,
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.axhline(y=1, color="orange", linestyle=":", linewidth=1)
    ax.axhline(y=-1, color="orange", linestyle=":", linewidth=1)

    ax.set_xlabel("Module", fontsize=12)
    ax.set_ylabel("Mean Height Change (%)", fontsize=12)
    ax.set_title("Average Pulse Height Change", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-1, 1)

    # Add value labels
    for i, (mean, std) in enumerate(zip(change_means, change_stds)):
        ax.text(
            i,
            mean + std + 0.05,
            f"{mean:.2f}%\n±{std:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 3: Distribution of changes
    ax = axes[1, 0]
    for idx, data in enumerate(all_data):
        module = data["module"]
        df = data["df"]
        ax.hist(
            df["Height_Change_%"],
            bins=50,
            alpha=0.5,
            label=module,
            color=colors[idx],
            range=(-3, 3),
        )

    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.axvline(x=1, color="orange", linestyle=":", linewidth=1)
    ax.axvline(x=-1, color="orange", linestyle=":", linewidth=1)
    ax.set_xlabel("Height Change (%)", fontsize=12)
    ax.set_ylabel("Number of LEDs", fontsize=12)
    ax.set_title("Distribution of Height Changes", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Statistics table
    ax = axes[1, 1]
    ax.axis("off")

    # Create summary table
    table_data = []
    table_data.append(
        [
            "Module",
            "Total\nLEDs",
            "Mean\nChange%",
            "Std\n%",
            ">1%\nLEDs",
            ">2%\nLEDs",
            ">3%\nLEDs",
        ]
    )

    for data in all_data:
        module = data["module"]
        df = data["df"]
        total = len(df)
        mean_change = df["Height_Change_%"].mean()
        std_change = df["Height_Change_%"].std()
        above_1 = len(df[abs(df["Height_Change_%"]) > 1.0])
        above_2 = len(df[abs(df["Height_Change_%"]) > 2.0])
        above_3 = len(df[abs(df["Height_Change_%"]) > 3.0])

        table_data.append(
            [
                module,
                f"{total:,}",
                f"{mean_change:.2f}",
                f"{std_change:.2f}",
                f"{above_1}\n({above_1/total*100:.1f}%)",
                f"{above_2}\n({above_2/total*100:.1f}%)",
                f"{above_3}\n({above_3/total*100:.1f}%)",
            ]
        )

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.12, 0.12, 0.12, 0.16, 0.16, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(7):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    ax.set_title("Statistical Summary", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    output_file2 = RESULTS_DIR / "pulse_height_summary_clean.png"
    plt.savefig(output_file2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_file2}")

    print("\n" + "=" * 80)


def create_summary_report():
    """Create a text summary report"""

    output_file = RESULTS_DIR / "PULSE_HEIGHT_SUMMARY.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PULSE HEIGHT ANALYSIS SUMMARY\n")
        f.write("Focus: Pulse Height Only (Peak - Valley)\n")
        f.write("=" * 80 + "\n\n")

        for module in MODULES:
            comparison_file = (
                RESULTS_DIR / f"{module}_comparison" / "detailed_comparison.csv"
            )

            if not comparison_file.exists():
                continue

            df = pd.read_csv(comparison_file)

            f.write(f"\n{module.upper()} MODULE ({len(df):,} LEDs)\n")
            f.write("-" * 80 + "\n")

            f.write("\nPULSE HEIGHT (Peak-Valley):\n")
            f.write(
                f"  Reference:   Mean={df['Ref_Height'].mean():.4f}, Std={df['Ref_Height'].std():.4f}\n"
            )
            f.write(
                f"  Cross-talk:  Mean={df['CT_Height'].mean():.4f}, Std={df['CT_Height'].std():.4f}\n"
            )
            f.write(
                f"  Difference:  Mean={df['Height_Diff'].mean():.4f}, Std={df['Height_Diff'].std():.4f}\n"
            )
            f.write(
                f"  Change:      Mean={df['Height_Change_%'].mean():.2f}%, Std={df['Height_Change_%'].std():.2f}%\n"
            )

            f.write("\nCHANGE DISTRIBUTION:\n")
            for threshold in [0.5, 1.0, 2.0, 3.0]:
                count = len(df[abs(df["Height_Change_%"]) > threshold])
                pct = count / len(df) * 100
                f.write(f"  >{threshold:>4.1f}%: {count:>5} LEDs ({pct:>5.2f}%)\n")

            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"✅ Summary report saved: {output_file}")


if __name__ == "__main__":
    create_simplified_pulse_height_csv()
    create_pulse_height_only_plots()
    create_summary_report()

    print("\n" + "=" * 80)
    print("PULSE HEIGHT ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nCreated Files:")
    print("  CSV Files (pulse height only):")
    print("    • Rear_pulse_height_only.csv")
    print("    • Middle_pulse_height_only.csv")
    print("    • Front_pulse_height_only.csv")
    print("\n  Visualizations:")
    print("    • pulse_height_comparison_clean.png")
    print("    • pulse_height_summary_clean.png")
    print("\n  Summary:")
    print("    • PULSE_HEIGHT_SUMMARY.txt")
    print("\n" + "=" * 80)
