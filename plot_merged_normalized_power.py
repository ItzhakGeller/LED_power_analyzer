"""
Plot Normalized Power vs LED Number from Merged Data
==================================================

Creates visualization of normalized LED power across all Spark modules
with clear boundaries between different Spark sections.

Author: LED Analysis Team
Date: October 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_normalized_power_vs_led():
    """Create normalized power vs LED number plot with Spark boundaries"""

    # Load the merged data
    merged_file = "W3_LPH_merged_data.csv"

    if not Path(merged_file).exists():
        print(f"❌ Error: {merged_file} not found!")
        print("Please ensure the merged file exists in the current directory.")
        return

    print(f"📊 Loading merged data from: {merged_file}")

    # Read the CSV file
    df = pd.read_csv(merged_file)
    print(f"✅ Loaded {len(df):,} LED measurements")

    # Display data info
    print("\n📋 Data columns:", list(df.columns))
    print(f"📊 LED range: {df['led_number'].min()} to {df['led_number'].max()}")

    # Create the plot
    plt.figure(figsize=(20, 8))

    # Plot normalized power vs LED number
    plt.plot(
        df["led_number"],
        df["normalized_height"],
        linewidth=0.5,
        alpha=0.8,
        color="#2E86AB",
        label="Normalized LED Power",
    )

    # Define Spark module boundaries based on LED ranges
    # From the analysis results:
    # Spark_Rear_W3: LEDs 2304-15615
    # Spark_Middle: LEDs 256-15615
    # Spark_Front: LEDs 256-21247

    # Determine boundaries from actual data
    led_numbers = df["led_number"].values

    # Find transitions between different Spark modules
    # Look for gaps or jumps in LED numbering to identify boundaries
    led_diff = np.diff(np.array(led_numbers))
    large_gaps = np.where(led_diff > 50)[0]  # Identify significant gaps

    boundaries = []
    if len(large_gaps) > 0:
        for gap_idx in large_gaps:
            boundary_led = led_numbers[gap_idx + 1]
            boundaries.append(boundary_led)

    # Use Spark section information to determine boundaries
    unique_sparks = df["spark_section"].unique()
    print(f"\n🔍 Spark sections found: {unique_sparks}")

    # Get LED ranges for each spark section
    boundaries = []
    spark_ranges = {}

    for spark in unique_sparks:
        spark_data = df[df["spark_section"] == spark]
        led_min = spark_data["led_number"].min()
        led_max = spark_data["led_number"].max()
        spark_ranges[spark] = (led_min, led_max)
        print(f"   {spark}: LEDs {led_min:,} - {led_max:,}")

    # Create boundaries between different spark sections
    # Sort spark ranges by minimum LED number
    sorted_sparks = sorted(spark_ranges.items(), key=lambda x: x[1][0])

    for i in range(len(sorted_sparks) - 1):
        current_spark, (_, current_max) = sorted_sparks[i]
        next_spark, (next_min, _) = sorted_sparks[i + 1]

        # Add boundary at the transition point
        if current_max < next_min:
            boundary = (current_max + next_min) // 2
        else:
            boundary = next_min
        boundaries.append(boundary)
        print(f"🎯 Boundary between {current_spark} and {next_spark}: LED {boundary}")

    print(f"🎯 Final boundaries: {boundaries}")

    # Add vertical dashed lines for Spark boundaries
    colors = ["red", "orange", "purple"]
    labels = ["Spark Boundary 1", "Spark Boundary 2", "Spark Boundary 3"]

    for i, boundary in enumerate(boundaries):
        if i < len(colors):
            plt.axvline(
                x=boundary,
                color=colors[i],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"{labels[i]} (LED {boundary})",
            )

    # Add text annotations for Spark regions
    if len(boundaries) >= 1:
        # Add region labels for each spark section
        for spark in unique_sparks:
            spark_data = df[df["spark_section"] == spark]
            led_min_spark = spark_data["led_number"].min()
            led_max_spark = spark_data["led_number"].max()
            mid_led = (led_min_spark + led_max_spark) / 2

            # Find corresponding normalized power value at midpoint
            mid_idx = np.argmin(np.abs(df["led_number"] - mid_led))
            mid_power = df["normalized_height"].iloc[mid_idx]

            plt.annotate(
                spark,
                xy=(mid_led, mid_power + 0.1),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

    # Formatting
    plt.xlabel("LED Number", fontsize=14, fontweight="bold")
    plt.ylabel("Normalized Power", fontsize=14, fontweight="bold")
    plt.title(
        "WLPH W3 LED Characterization: Normalized Power vs LED Number\n"
        + "Comprehensive Analysis Across All Spark Modules",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)

    # Add statistics box
    stats_text = f"""Analysis Summary:
Total LEDs: {len(df):,}
LED Range: {df['led_number'].min():,} - {df['led_number'].max():,}
Power Range: {df['normalized_height'].min():.3f} - {df['normalized_height'].max():.3f}
Mean Power: {df['normalized_height'].mean():.3f} ± {df['normalized_height'].std():.3f}
Spark Modules: {len(unique_sparks)}"""

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    # Save the plot
    output_file = "WLPH_W3_Normalized_Power_vs_LED_Number.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print(f"\n✅ Plot saved: {output_file}")
    print(f"📊 Found {len(boundaries)} Spark boundaries")
    if boundaries:
        print(f"🔍 Boundary LEDs: {boundaries}")

    plt.show()

    return output_file


if __name__ == "__main__":
    plot_normalized_power_vs_led()
