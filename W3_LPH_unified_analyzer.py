"""
W3 LPH Unified LED Analysis
==========================

Combines all three Spark measurements into one continuous LED sequence:
- Total LEDs: 49,664 (776mm × 64 LEDs/mm)
- Spark_Rear_W3: Original LEDs 2304-15615 → Unified LEDs 0-13,311
- Spark_Middle: Original LEDs 256-15615 → Unified LEDs 13,312-28,671
- Spark_Front: Original LEDs 256-21247 → Unified LEDs 28,672-49,663
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def load_spark_data():
    """Load and verify data from all three Spark analysis results"""

    base_path = Path("complete_analysis_output")

    # Define the three Spark configurations
    spark_configs = {
        "Rear": {
            "file": base_path
            / "Spark_Rear_W3_Analysis"
            / "Spark_Rear_W3_simplified_analysis.csv",
            "original_led_range": (2304, 15615),
            "expected_count": 13312,
            "unified_start": 0,
        },
        "Middle": {
            "file": base_path
            / "Spark_Middle_Analysis"
            / "Spark_middle_simplified_analysis.csv",
            "original_led_range": (256, 15615),
            "expected_count": 15360,
            "unified_start": 13312,
        },
        # 'Front': {  # Commented out - CSV file missing due to permission error
        #     'file': base_path / "Spark_Front_Analysis" / "Spark_front_simplified_analysis.csv",
        #     'original_led_range': (256, 21247),
        #     'expected_count': 20992,
        #     'unified_start': 28672
        # }
    }

    print("🔬 W3 LPH Unified LED Analysis")
    print("=" * 50)
    print("📏 Total Length: 776mm")
    print("🔢 LED Density: 64 LEDs/mm")
    print("📊 Expected Total LEDs: 49,664")
    print("=" * 50)

    all_data = []

    for spark_name, config in spark_configs.items():
        print(f"\n📁 Loading {spark_name} data...")

        if not config["file"].exists():
            print(f"❌ File not found: {config['file']}")
            continue

        # Load CSV data (skip comment lines starting with #)
        df = pd.read_csv(config["file"], comment="#")

        print(f"   ✅ Loaded {len(df):,} LEDs")
        print(
            f"   📋 Original LED range: {config['original_led_range'][0]}-{config['original_led_range'][1]}"
        )
        print(f"   🎯 Expected count: {config['expected_count']:,}")

        # Verify LED count matches expectation
        if len(df) != config["expected_count"]:
            print(
                f"   ⚠️  LED count mismatch! Expected {config['expected_count']}, got {len(df)}"
            )
        else:
            print(f"   ✅ LED count matches expectation")

        # Add unified LED numbering (starting from 0)
        df["Unified_LED_Number"] = range(
            config["unified_start"], config["unified_start"] + len(df)
        )

        # Add Spark identifier
        df["Spark_Section"] = spark_name

        # Add position in mm (assuming 64 LEDs per mm)
        df["Position_mm"] = df["Unified_LED_Number"] / 64.0

        print(
            f"   🔢 Unified LED range: {config['unified_start']}-{config['unified_start'] + len(df) - 1}"
        )
        print(
            f"   📏 Position range: {df['Position_mm'].min():.2f}-{df['Position_mm'].max():.2f} mm"
        )

        all_data.append(df)

    if not all_data:
        raise ValueError("No data files found!")

    # Combine all data
    unified_df = pd.concat(all_data, ignore_index=True)

    print(f"\n🎉 UNIFIED W3 LPH DATA CREATED")
    print("=" * 50)
    print(f"📊 Total LEDs: {len(unified_df):,}")
    print(
        f"🔢 LED range: {unified_df['Unified_LED_Number'].min()}-{unified_df['Unified_LED_Number'].max()}"
    )
    print(f"📏 Physical length: {unified_df['Position_mm'].max():.2f} mm")
    print(f"🎯 Coverage: {len(unified_df)/49664*100:.1f}% of expected 49,664 LEDs")

    return unified_df


def create_unified_analysis(unified_df):
    """Create comprehensive analysis of unified W3 LPH data"""

    print(f"\n📊 CREATING UNIFIED ANALYSIS")
    print("-" * 40)

    # Statistics by Spark section
    spark_stats = (
        unified_df.groupby("Spark_Section")
        .agg(
            {
                "Unified_LED_Number": ["count", "min", "max"],
                "Position_mm": ["min", "max"],
                "absolute_height": ["mean", "std", "min", "max"],
                "peak_std": ["mean", "std", "min", "max"],
            }
        )
        .round(4)
    )

    print("📋 Statistics by Spark Section:")
    print(spark_stats)

    # Overall statistics
    print(f"\n📈 OVERALL W3 LPH STATISTICS:")
    print(f"   Total LEDs: {len(unified_df):,}")
    print(f"   Height - Mean: {unified_df['absolute_height'].mean():.6f}")
    print(f"   Height - Std: {unified_df['absolute_height'].std():.6f}")
    print(
        f"   Height - Range: {unified_df['absolute_height'].min():.6f} to {unified_df['absolute_height'].max():.6f}"
    )
    print(f"   Peak Std - Mean: {unified_df['peak_std'].mean():.6f}")
    print(f"   Peak Std - Std: {unified_df['peak_std'].std():.6f}")
    print(
        f"   Peak Std - Range: {unified_df['peak_std'].min():.6f} to {unified_df['peak_std'].max():.6f}"
    )

    return spark_stats


def create_unified_visualizations(unified_df):
    """Create comprehensive visualizations of the unified W3 LPH data"""

    print(f"\n🖼️  CREATING UNIFIED VISUALIZATIONS")
    print("-" * 40)

    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Color mapping for Spark sections
    colors = {"Rear": "blue", "Middle": "green", "Front": "red"}

    # 1. LED Height across full W3 LPH
    ax1 = axes[0, 0]
    for spark in ["Rear", "Middle", "Front"]:
        spark_data = unified_df[unified_df["Spark_Section"] == spark]
        ax1.scatter(
            spark_data["Position_mm"],
            spark_data["absolute_height"],
            c=colors[spark],
            alpha=0.6,
            s=0.5,
            label=f"Spark {spark}",
        )

    ax1.set_xlabel("Position (mm)")
    ax1.set_ylabel("LED Absolute Height")
    ax1.set_title("LED Absolute Height Across W3 LPH (776mm)", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. SNR across full W3 LPH
    ax2 = axes[0, 1]
    for spark in ["Rear", "Middle", "Front"]:
        spark_data = unified_df[unified_df["Spark_Section"] == spark]
        ax2.scatter(
            spark_data["Position_mm"],
            spark_data["peak_std"],
            c=colors[spark],
            alpha=0.6,
            s=0.5,
            label=f"Spark {spark}",
        )

    ax2.set_xlabel("Position (mm)")
    ax2.set_ylabel("Peak Std Dev")
    ax2.set_title("Peak Standard Deviation Across W3 LPH (776mm)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Height distribution by Spark
    ax3 = axes[0, 2]
    for spark in ["Rear", "Middle", "Front"]:
        spark_data = unified_df[unified_df["Spark_Section"] == spark]
        ax3.hist(
            spark_data["absolute_height"],
            bins=50,
            alpha=0.6,
            label=f"Spark {spark}",
            color=colors[spark],
        )

    ax3.set_xlabel("Absolute Height")
    ax3.set_ylabel("Count")
    ax3.set_title("Absolute Height Distribution by Spark Section", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. LED continuity check
    ax4 = axes[1, 0]
    ax4.plot(unified_df["Unified_LED_Number"], "b-", linewidth=1, alpha=0.7)
    ax4.set_xlabel("Index")
    ax4.set_ylabel("Unified LED Number")
    ax4.set_title("LED Number Continuity Check", fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # Add section boundaries
    rear_end = len(unified_df[unified_df["Spark_Section"] == "Rear"])
    middle_end = rear_end + len(unified_df[unified_df["Spark_Section"] == "Middle"])

    ax4.axvline(x=rear_end, color="red", linestyle="--", alpha=0.7, label="Rear→Middle")
    ax4.axvline(
        x=middle_end, color="red", linestyle="--", alpha=0.7, label="Middle→Front"
    )
    ax4.legend()

    # 5. Position mapping verification
    ax5 = axes[1, 1]
    ax5.scatter(
        unified_df["Unified_LED_Number"],
        unified_df["Position_mm"],
        c=unified_df["Spark_Section"].map(colors),
        alpha=0.6,
        s=0.5,
    )
    ax5.set_xlabel("Unified LED Number")
    ax5.set_ylabel("Position (mm)")
    ax5.set_title("LED Number → Position Mapping", fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics
    ax6 = axes[1, 2]

    # Calculate coverage and statistics
    total_expected = 49664
    coverage_pct = len(unified_df) / total_expected * 100

    summary_text = f"""W3 LPH UNIFIED ANALYSIS SUMMARY

PHYSICAL SPECIFICATIONS:
• Total length: 776 mm
• LED density: 64 LEDs/mm
• Expected total LEDs: {total_expected:,}

MEASURED DATA:
• Total measured LEDs: {len(unified_df):,}
• Coverage: {coverage_pct:.1f}%
• LED range: 0-{len(unified_df)-1:,}
• Position range: 0-{unified_df['Position_mm'].max():.1f} mm

SPARK SECTIONS:
• Rear: {len(unified_df[unified_df['Spark_Section'] == 'Rear']):,} LEDs
• Middle: {len(unified_df[unified_df['Spark_Section'] == 'Middle']):,} LEDs  
• Front: {len(unified_df[unified_df['Spark_Section'] == 'Front']):,} LEDs

QUALITY METRICS:
• Avg Height: {unified_df['absolute_height'].mean():.6f}
• Height Std: {unified_df['absolute_height'].std():.6f}
• Avg Peak Std: {unified_df['peak_std'].mean():.6f}
• Peak Std Dev: {unified_df['peak_std'].std():.6f}"""

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax6.set_title("W3 LPH Summary", fontweight="bold")
    ax6.axis("off")

    plt.tight_layout()

    # Save visualization
    output_file = "W3_LPH_unified_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"   ✅ Unified visualization saved: {output_file}")

    return output_file


def export_unified_data(unified_df):
    """Export the unified W3 LPH data to CSV"""

    print(f"\n💾 EXPORTING UNIFIED DATA")
    print("-" * 30)

    # Prepare final dataset with clear column names
    export_df = unified_df[
        [
            "Unified_LED_Number",
            "Position_mm",
            "Spark_Section",
            "led_number",
            "absolute_height",
            "peak_std",
            "pulse_width",
        ]
    ].copy()

    # Rename columns for clarity
    export_df.columns = [
        "LED_Number_Unified",
        "Position_mm",
        "Spark_Section",
        "LED_Number_Original",
        "Absolute_Height",
        "Peak_Std",
        "Pulse_Width",
    ]

    # Sort by unified LED number
    export_df = export_df.sort_values("LED_Number_Unified").reset_index(drop=True)

    # Export to CSV
    output_file = "W3_LPH_unified_led_data.csv"
    export_df.to_csv(output_file, index=False)

    print(f"   ✅ Unified data exported: {output_file}")
    print(f"   📊 Exported {len(export_df):,} LEDs")
    print(f"   📋 Columns: {list(export_df.columns)}")

    # Create summary report
    summary_file = "W3_LPH_analysis_summary.txt"
    with open(summary_file, "w") as f:
        f.write("W3 LPH UNIFIED LED ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("PHYSICAL SPECIFICATIONS:\n")
        f.write(f"• Total length: 776 mm\n")
        f.write(f"• LED density: 64 LEDs/mm\n")
        f.write(f"• Expected total LEDs: 49,664\n\n")

        f.write("MEASURED DATA:\n")
        f.write(f"• Total measured LEDs: {len(export_df):,}\n")
        f.write(f"• Coverage: {len(export_df)/49664*100:.1f}%\n")
        f.write(f"• LED range: 0-{len(export_df)-1:,}\n")
        f.write(f"• Position range: 0-{export_df['Position_mm'].max():.1f} mm\n\n")

        f.write("SPARK SECTIONS:\n")
        for spark in ["Rear", "Middle", "Front"]:
            count = len(export_df[export_df["Spark_Section"] == spark])
            f.write(f"• {spark}: {count:,} LEDs\n")

        f.write(f"\nQUALITY METRICS:\n")
        f.write(f"• Average Height: {export_df['Absolute_Height'].mean():.6f}\n")
        f.write(f"• Height Std Dev: {export_df['Absolute_Height'].std():.6f}\n")
        f.write(f"• Average Peak Std: {export_df['Peak_Std'].mean():.6f}\n")
        f.write(f"• Peak Std Dev: {export_df['Peak_Std'].std():.6f}\n")

    print(f"   ✅ Summary report: {summary_file}")

    return output_file, summary_file


def main():
    """Main function to create unified W3 LPH analysis"""

    try:
        # Load all Spark data
        unified_df = load_spark_data()

        # Create analysis
        spark_stats = create_unified_analysis(unified_df)

        # Create visualizations
        viz_file = create_unified_visualizations(unified_df)

        # Export unified data
        csv_file, summary_file = export_unified_data(unified_df)

        print(f"\n🎉 W3 LPH UNIFIED ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"📄 Unified data: {csv_file}")
        print(f"🖼️  Visualization: {viz_file}")
        print(f"📋 Summary: {summary_file}")
        print(f"📊 Total LEDs processed: {len(unified_df):,}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
