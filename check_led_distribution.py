import pandas as pd

# Load the merged data
df = pd.read_csv("W3_LPH_merged_data.csv")

print("LED counts per Spark:")
print(df["spark_section"].value_counts())
print()

print("LED ranges per Spark:")
for spark in df["spark_section"].unique():
    spark_data = df[df["spark_section"] == spark]
    print(
        f"{spark}: {len(spark_data):,} LEDs, range {spark_data['led_number'].min()}-{spark_data['led_number'].max()}"
    )

print(f"\nTotal LEDs: {len(df):,}")

# Check if there are overlaps
print("\nChecking for overlaps:")
middle_range = set(df[df["spark_section"] == "Spark_Middle"]["led_number"])
front_range = set(df[df["spark_section"] == "Spark_Front"]["led_number"])
rear_range = set(df[df["spark_section"] == "Spark_Rear_W3"]["led_number"])

print(f"Middle-Front overlap: {len(middle_range & front_range):,} LEDs")
print(f"Middle-Rear overlap: {len(middle_range & rear_range):,} LEDs")
print(f"Front-Rear overlap: {len(front_range & rear_range):,} LEDs")
