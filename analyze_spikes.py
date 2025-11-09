import pandas as pd
import numpy as np

# Analyze Front
print("=" * 70)
print("FRONT MODULE - Analysis of Spikes in Change%")
print("=" * 70)

df_front = pd.read_csv(r"crosstalk_comparison_results\Front_pulse_height_only.csv")

print("\n1. Top 20 LEDs with LARGEST POSITIVE changes:")
print("-" * 70)
top_pos = df_front.nlargest(20, "Height_Change_%")[
    ["LED", "Ref_Height", "CT_Height", "Height_Diff", "Height_Change_%"]
]
print(top_pos.to_string(index=False))
print(f"\nAverage Ref_Height of these LEDs: {top_pos['Ref_Height'].mean():.4f}")
print(
    f"Are these in high-amplitude regions? {top_pos['Ref_Height'].mean() > df_front['Ref_Height'].mean()}"
)

print("\n2. Top 20 LEDs with LARGEST NEGATIVE changes:")
print("-" * 70)
top_neg = df_front.nsmallest(20, "Height_Change_%")[
    ["LED", "Ref_Height", "CT_Height", "Height_Diff", "Height_Change_%"]
]
print(top_neg.to_string(index=False))
print(f"\nAverage Ref_Height of these LEDs: {top_neg['Ref_Height'].mean():.4f}")
print(
    f"Are these in high-amplitude regions? {top_neg['Ref_Height'].mean() > df_front['Ref_Height'].mean()}"
)

print("\n3. Checking specific spike region (LED 9000-10500):")
print("-" * 70)
spike_region = df_front[(df_front["LED"] >= 9000) & (df_front["LED"] <= 10500)]
print(f"Number of LEDs in region: {len(spike_region)}")
print(f"Mean Change% in this region: {spike_region['Height_Change_%'].mean():.3f}%")
print(f"Std Change% in this region: {spike_region['Height_Change_%'].std():.3f}%")
print(f"Mean Change% overall: {df_front['Height_Change_%'].mean():.3f}%")
print(f"Std Change% overall: {df_front['Height_Change_%'].std():.3f}%")
print(
    f"\nIs this region different? {abs(spike_region['Height_Change_%'].mean() - df_front['Height_Change_%'].mean()) > 0.1}"
)

# Analyze Rear
print("\n" + "=" * 70)
print("REAR MODULE - Analysis of Spikes in Change%")
print("=" * 70)

df_rear = pd.read_csv(r"crosstalk_comparison_results\Rear_pulse_height_only.csv")

print("\n1. Top 20 LEDs with LARGEST POSITIVE changes:")
print("-" * 70)
top_pos_r = df_rear.nlargest(20, "Height_Change_%")[
    ["LED", "Ref_Height", "CT_Height", "Height_Diff", "Height_Change_%"]
]
print(top_pos_r.to_string(index=False))
print(f"\nAverage Ref_Height of these LEDs: {top_pos_r['Ref_Height'].mean():.4f}")
print(
    f"Are these in high-amplitude regions? {top_pos_r['Ref_Height'].mean() > df_rear['Ref_Height'].mean()}"
)

print("\n2. Top 20 LEDs with LARGEST NEGATIVE changes:")
print("-" * 70)
top_neg_r = df_rear.nsmallest(20, "Height_Change_%")[
    ["LED", "Ref_Height", "CT_Height", "Height_Diff", "Height_Change_%"]
]
print(top_neg_r.to_string(index=False))
print(f"\nAverage Ref_Height of these LEDs: {top_neg_r['Ref_Height'].mean():.4f}")
print(
    f"Are these in high-amplitude regions? {top_neg_r['Ref_Height'].mean() > df_rear['Ref_Height'].mean()}"
)

print("\n3. Checking specific spike regions in Rear:")
print("-" * 70)
for start, end in [(3000, 4000), (6000, 7000), (10000, 11000)]:
    region = df_rear[(df_rear["LED"] >= start) & (df_rear["LED"] <= end)]
    if len(region) > 0:
        print(f"\nLED {start}-{end}:")
        print(f"  Mean Change%: {region['Height_Change_%'].mean():.3f}%")
        print(f"  Std: {region['Height_Change_%'].std():.3f}%")
        print(
            f"  Different from overall? {abs(region['Height_Change_%'].mean() - df_rear['Height_Change_%'].mean()) > 0.1}"
        )

print("\n" + "=" * 70)
print("SUMMARY - Does amplitude correlate with change%?")
print("=" * 70)

for name, df in [("Front", df_front), ("Rear", df_rear)]:
    print(f"\n{name}:")
    corr = df["Ref_Height"].corr(df["Height_Change_%"])
    print(f"  Correlation (Ref_Height vs Change%): {corr:.4f}")
    print(
        f"  Interpretation: {'STRONG' if abs(corr) > 0.3 else 'WEAK' if abs(corr) > 0.1 else 'NO'} correlation"
    )
