"""
W3 LPH Simple CSV Merger
=======================

Simply combines the three Spark CSV files into one continuous LED sequence:
- Spark_Rear_W3: LEDs 0-13,311 (originally 2304-15615)
- Spark_Middle: LEDs 13,312-28,671 (originally 256-15615)
- Spark_Front: LEDs 28,672-49,663 (originally 256-21247)

Just merges the data with new LED numbering - no normalization or analysis.
"""

import pandas as pd
from pathlib import Path


def merge_spark_csv_files():
    """Simply merge the three Spark CSV files with continuous LED numbering"""

    print("🔗 W3 LPH CSV Merger")
    print("=" * 40)

    # Define the files and their new LED starting numbers
    spark_files = [
        {
            "name": "Spark_Rear_W3",
            "file": "complete_analysis_output/Spark_Rear_W3_Analysis/Spark_Rear_W3_simplified_analysis.csv",
            "new_led_start": 0,
            "expected_count": 13312,
        },
        {
            "name": "Spark_Middle",
            "file": "complete_analysis_output/Spark_Middle_Analysis/Spark_middle_simplified_analysis.csv",
            "new_led_start": 13312,
            "expected_count": 15360,
        },
        {
            "name": "Spark_Front",
            "file": "complete_analysis_output/Spark_Front_Analysis/Spark_front_simplified_analysis.csv",
            "new_led_start": 28672,
            "expected_count": 20992,
        },
    ]

    merged_data = []

    for spark in spark_files:
        print(f"\n📁 Processing {spark['name']}...")

        file_path = Path(spark["file"])
        if not file_path.exists():
            print(f"   ❌ File not found: {spark['file']}")
            continue

        # Read CSV (skip comment lines)
        df = pd.read_csv(spark["file"], comment="#")

        print(f"   ✅ Loaded {len(df):,} LEDs")
        print(
            f"   📋 Original LED range: {df['led_number'].min()}-{df['led_number'].max()}"
        )

        # Add new continuous LED numbering
        df["led_number_unified"] = range(
            spark["new_led_start"], spark["new_led_start"] + len(df)
        )

        # Add spark section identifier
        df["spark_section"] = spark["name"]

        print(
            f"   🔢 New LED range: {spark['new_led_start']}-{spark['new_led_start'] + len(df) - 1}"
        )

        merged_data.append(df)

    if not merged_data:
        print("❌ No CSV files found!")
        return None

    # Combine all dataframes
    combined_df = pd.concat(merged_data, ignore_index=True)

    # Reorder columns to put unified LED number first
    columns = ["led_number_unified", "spark_section", "led_number"] + [
        col
        for col in combined_df.columns
        if col not in ["led_number_unified", "spark_section", "led_number"]
    ]

    combined_df = combined_df[columns]

    print(f"\n🎉 MERGER COMPLETE!")
    print("=" * 40)
    print(f"📊 Total LEDs: {len(combined_df):,}")
    print(
        f"🔢 LED range: {combined_df['led_number_unified'].min()}-{combined_df['led_number_unified'].max()}"
    )

    # Show breakdown by section
    print(f"\n📋 BREAKDOWN BY SECTION:")
    for section in combined_df["spark_section"].unique():
        count = len(combined_df[combined_df["spark_section"] == section])
        min_led = combined_df[combined_df["spark_section"] == section][
            "led_number_unified"
        ].min()
        max_led = combined_df[combined_df["spark_section"] == section][
            "led_number_unified"
        ].max()
        print(f"   {section}: {count:,} LEDs (range {min_led}-{max_led})")

    return combined_df


def export_merged_csv(combined_df):
    """Export the merged data to CSV"""

    if combined_df is None:
        return None

    print(f"\n💾 EXPORTING MERGED DATA")
    print("-" * 30)

    # Sort by unified LED number to ensure proper order
    combined_df = combined_df.sort_values("led_number_unified").reset_index(drop=True)

    # Export to CSV
    output_file = "W3_LPH_merged_data.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"   ✅ Merged CSV saved: {output_file}")
    print(f"   📊 Total rows: {len(combined_df):,}")
    print(f"   📋 Columns: {len(combined_df.columns)}")
    print(f"   📝 Column names: {', '.join(combined_df.columns[:5])}...")

    # Show sample of data
    print(f"\n📄 SAMPLE DATA (first 5 rows):")
    print(
        combined_df.head()[
            ["led_number_unified", "spark_section", "led_number", "absolute_height"]
        ].to_string(index=False)
    )

    print(f"\n📄 SAMPLE DATA (last 5 rows):")
    print(
        combined_df.tail()[
            ["led_number_unified", "spark_section", "led_number", "absolute_height"]
        ].to_string(index=False)
    )

    return output_file


def main():
    """Main function - simple CSV merger"""

    try:
        # Merge the CSV files
        combined_df = merge_spark_csv_files()

        # Export merged data
        if combined_df is not None:
            output_file = export_merged_csv(combined_df)

            print(f"\n🎉 SUCCESS!")
            print(f"📄 Merged file: {output_file}")
            print(f"📊 Total LEDs: {len(combined_df):,}")

        else:
            print("❌ Merge failed - no data to export")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
