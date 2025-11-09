"""
Investigate LED Count Discrepancy
================================

Analyze why we have only ~20K LEDs instead of expected 50K
by examining the original analysis results and data structure.
"""

import pandas as pd
from pathlib import Path


def investigate_led_counts():
    """Investigate the LED count discrepancy"""

    print("🔍 INVESTIGATING LED COUNT DISCREPANCY")
    print("=" * 60)

    # 1. Check original analysis results
    print("\n1. ORIGINAL ANALYSIS RESULTS FROM COMPLETE ANALYZER:")
    print("-" * 50)

    # Check if individual CSV files exist
    csv_files = [
        "complete_analysis_output/Spark_Rear_W3_simplified_analysis.csv",
        "complete_analysis_output/Spark_middle_simplified_analysis.csv",
        "complete_analysis_output/Spark_front_simplified_analysis.csv",
    ]

    total_expected = 0
    total_actual = 0

    for csv_file in csv_files:
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            spark_name = Path(csv_file).stem.replace("_simplified_analysis", "")
            led_count = len(df)
            led_min = (
                df["LED_Number"].min()
                if "LED_Number" in df.columns
                else df.iloc[:, 0].min()
            )
            led_max = (
                df["LED_Number"].max()
                if "LED_Number" in df.columns
                else df.iloc[:, 0].max()
            )

            print(f"✅ {spark_name}:")
            print(f"   📊 LEDs detected: {led_count:,}")
            print(f"   📊 LED range: {led_min:,} - {led_max:,}")
            print(f"   📊 Expected range span: {led_max - led_min + 1:,}")
            print()

            total_actual += led_count
        else:
            print(f"❌ File not found: {csv_file}")

    print(f"🎯 TOTAL LEDs from individual files: {total_actual:,}")

    # 2. Check merged file
    print("\n2. MERGED FILE ANALYSIS:")
    print("-" * 30)

    merged_file = "W3_LPH_merged_data.csv"
    if Path(merged_file).exists():
        df_merged = pd.read_csv(merged_file)
        print(f"📊 Merged file total rows: {len(df_merged):,}")

        # Check for duplicates by LED number
        unique_leds = df_merged["led_number"].nunique()
        print(f"📊 Unique LED numbers: {unique_leds:,}")

        # Check overlaps between spark sections
        print("\n📊 LED ranges per Spark section:")
        for spark in df_merged["spark_section"].unique():
            spark_data = df_merged[df_merged["spark_section"] == spark]
            led_min = spark_data["led_number"].min()
            led_max = spark_data["led_number"].max()
            count = len(spark_data)
            print(f"   {spark}: {count:,} LEDs, range {led_min:,}-{led_max:,}")

    else:
        print(f"❌ Merged file not found: {merged_file}")

    # 3. Check expected vs actual
    print(f"\n3. EXPECTED VS ACTUAL ANALYSIS:")
    print("-" * 35)

    # From the original analysis output
    expected_counts = {
        "Spark_Rear_W3": 13312,  # LEDs 2304-15615
        "Spark_middle": 15360,  # LEDs 256-15615
        "Spark_front": 20992,  # LEDs 256-21247
    }

    expected_ranges = {
        "Spark_Rear_W3": (2304, 15615),
        "Spark_middle": (256, 15615),
        "Spark_front": (256, 21247),
    }

    print("📋 Expected from analysis output:")
    total_expected = 0
    for spark, count in expected_counts.items():
        range_start, range_end = expected_ranges[spark]
        total_expected += count
        print(f"   {spark}: {count:,} LEDs (range {range_start:,}-{range_end:,})")

    print(f"\n🎯 Total expected (with overlaps): {total_expected:,}")

    # 4. Identify the discrepancy source
    print(f"\n4. DISCREPANCY ANALYSIS:")
    print("-" * 25)

    print("🔍 The issue is OVERLAPPING LED RANGES:")
    print(
        f"   • Spark_middle (256-15615) and Spark_front (256-21247) share LEDs 256-15615"
    )
    print(
        f"   • Spark_middle (256-15615) and Spark_Rear_W3 (2304-15615) share LEDs 2304-15615"
    )
    print(f"   • This means many LEDs appear in multiple Spark files!")

    # Calculate overlaps
    middle_range = set(range(256, 15616))  # 256-15615
    front_range = set(range(256, 21248))  # 256-21247
    rear_range = set(range(2304, 15616))  # 2304-15615

    middle_front_overlap = len(middle_range & front_range)
    middle_rear_overlap = len(middle_range & rear_range)
    front_rear_overlap = len(front_range & rear_range)

    print(f"\n📊 Calculated overlaps:")
    print(f"   • Middle-Front overlap: {middle_front_overlap:,} LEDs")
    print(f"   • Middle-Rear overlap: {middle_rear_overlap:,} LEDs")
    print(f"   • Front-Rear overlap: {front_rear_overlap:,} LEDs")

    # Find actual unique LEDs
    all_leds = front_range.union(middle_range).union(rear_range)
    unique_led_count = len(all_leds)

    print(f"\n🎯 ACTUAL UNIQUE LEDs: {unique_led_count:,}")
    print(f"   LED range: {min(all_leds):,} - {max(all_leds):,}")

    # 5. Explain why not 50K
    print(f"\n5. WHY NOT 50K LEDs?")
    print("-" * 20)
    print("🔍 Possible reasons:")
    print("   1. ❓ WLPH W3 design has ~21K LEDs, not 50K")
    print("   2. ❓ Data collection covered only part of the full LED array")
    print("   3. ❓ Some LEDs were not functional/detectable")
    print("   4. ❓ LED numbering scheme skips ranges (gaps in numbering)")
    print("   5. ❓ Expected 50K was an overestimate")

    return unique_led_count


if __name__ == "__main__":
    unique_count = investigate_led_counts()
