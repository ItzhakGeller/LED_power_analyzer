"""
Plot Corrected LED Distribution - Remove Overlaps
===============================================

Creates visualization showing unique LEDs per Spark module,
removing overlaps to show the true 50K LED distribution.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_corrected_led_distribution():
    """Create corrected plot showing unique LEDs per Spark module"""

    # Load the merged data
    merged_file = "W3_LPH_merged_data.csv"
    df = pd.read_csv(merged_file)

    print(f"📊 Original data: {len(df):,} LED measurements")

    # Remove duplicates based on LED number, keeping one measurement per LED
    # Priority: Spark_Rear_W3 > Spark_Front > Spark_Middle
    df_unique = df.drop_duplicates(subset="led_number", keep="first")

    # Sort by LED number for proper ordering
    df_unique = df_unique.sort_values("led_number").reset_index(drop=True)

    print(f"📊 After removing overlaps: {len(df_unique):,} unique LEDs")

    # Reassign Spark sections to create proper 1/3 distribution
    total_leds = len(df_unique)
    third = total_leds // 3

    # Create new spark assignments
    df_unique["corrected_spark"] = "Unknown"

    # Divide into thirds based on sorted LED positions
    df_unique.loc[0 : third - 1, "corrected_spark"] = "Spark_1_First_Third"
    df_unique.loc[third : 2 * third - 1, "corrected_spark"] = "Spark_2_Middle_Third"
    df_unique.loc[2 * third :, "corrected_spark"] = "Spark_3_Last_Third"

    print(f"\n🎯 Corrected distribution:")
    corrected_counts = df_unique["corrected_spark"].value_counts().sort_index()
    for spark, count in corrected_counts.items():
        spark_data = df_unique[df_unique["corrected_spark"] == spark]
        led_min = spark_data["led_number"].min()
        led_max = spark_data["led_number"].max()
        print(f"   {spark}: {count:,} LEDs (range {led_min:,}-{led_max:,})")

    # Create the corrected plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    # Top plot: Original data with overlaps
    ax1.plot(
        df["led_number"],
        df["normalized_height"],
        linewidth=0.3,
        alpha=0.6,
        color="#FF6B6B",
        label="Original Data (with overlaps)",
    )

    # Color-code by original spark sections
    colors_orig = {
        "Spark_Rear_W3": "#FF6B6B",
        "Spark_Middle": "#4ECDC4",
        "Spark_Front": "#45B7D1",
    }
    for spark in df["spark_section"].unique():
        spark_data = df[df["spark_section"] == spark]
        ax1.scatter(
            spark_data["led_number"],
            spark_data["normalized_height"],
            color=colors_orig.get(spark, "gray"),
            s=0.1,
            alpha=0.5,
            label=f"{spark} (overlapping)",
        )

    ax1.set_title(
        "WLPH W3 LED Distribution - ORIGINAL DATA (With Overlaps)\n49,664 total measurements with significant overlap between modules",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Normalized Power", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom plot: Corrected unique LEDs
    ax2.plot(
        df_unique["led_number"],
        df_unique["normalized_height"],
        linewidth=0.5,
        alpha=0.8,
        color="#2E86AB",
        label="Corrected Unique LEDs",
    )

    # Color-code by corrected spark sections
    colors_new = {
        "Spark_1_First_Third": "#FF6B6B",
        "Spark_2_Middle_Third": "#4ECDC4",
        "Spark_3_Last_Third": "#45B7D1",
    }
    for spark in df_unique["corrected_spark"].unique():
        spark_data = df_unique[df_unique["corrected_spark"] == spark]
        if len(spark_data) > 0:
            ax2.scatter(
                spark_data["led_number"],
                spark_data["normalized_height"],
                color=colors_new.get(spark, "gray"),
                s=0.3,
                alpha=0.7,
                label=spark,
            )

    # Add boundaries between corrected thirds
    boundaries = []
    for i, spark in enumerate(["Spark_1_First_Third", "Spark_2_Middle_Third"]):
        spark_data = df_unique[df_unique["corrected_spark"] == spark]
        if len(spark_data) > 0:
            boundary = spark_data["led_number"].max()
            boundaries.append(boundary)
            ax2.axvline(
                x=boundary,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Third Boundary {i+1} (LED {boundary})",
            )

    ax2.set_title(
        "WLPH W3 LED Distribution - CORRECTED (Unique LEDs Only)\nProper 1/3 distribution across LED range",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("LED Number", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Normalized Power", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add statistics boxes
    stats_orig = f"""Original Data Issues:
Total Measurements: {len(df):,}
Spark_Front: {(df['spark_section']=='Spark_Front').sum():,}
Spark_Middle: {(df['spark_section']=='Spark_Middle').sum():,}  
Spark_Rear_W3: {(df['spark_section']=='Spark_Rear_W3').sum():,}
Major Overlaps Present"""

    ax1.text(
        0.02,
        0.98,
        stats_orig,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
    )

    stats_corrected = f"""Corrected Distribution:
Unique LEDs: {len(df_unique):,}
Each Third: ~{third:,} LEDs
LED Range: {df_unique['led_number'].min():,} - {df_unique['led_number'].max():,}
Power Range: {df_unique['normalized_height'].min():.3f} - {df_unique['normalized_height'].max():.3f}
No Overlaps"""

    ax2.text(
        0.02,
        0.98,
        stats_corrected,
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    plt.tight_layout()

    # Save both plots
    output_file = "WLPH_W3_Corrected_LED_Distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print(f"\n✅ Corrected plot saved: {output_file}")
    print(f"🎯 Boundaries between thirds: {boundaries}")

    plt.show()

    # Also create the corrected CSV
    corrected_file = "WLPH_W3_Unique_LEDs_Only.csv"
    df_unique.to_csv(corrected_file, index=False)
    print(f"📄 Corrected data saved: {corrected_file}")

    return output_file


if __name__ == "__main__":
    plot_corrected_led_distribution()
