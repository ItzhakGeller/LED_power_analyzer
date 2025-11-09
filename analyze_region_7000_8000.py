import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all three modules
df_rear = pd.read_csv(r"crosstalk_comparison_results\Rear_pulse_height_only.csv")
df_middle = pd.read_csv(r"crosstalk_comparison_results\Middle_pulse_height_only.csv")
df_front = pd.read_csv(r"crosstalk_comparison_results\Front_pulse_height_only.csv")

# Focus on region 7000-8000
region_rear = df_rear[(df_rear["LED"] >= 7000) & (df_rear["LED"] <= 8000)]
region_middle = df_middle[(df_middle["LED"] >= 7000) & (df_middle["LED"] <= 8000)]
region_front = df_front[(df_front["LED"] >= 7000) & (df_front["LED"] <= 8000)]

print("=" * 80)
print("ANALYSIS: LED Region 7000-8000 Across All Modules")
print("=" * 80)

for name, region, df_full in [
    ("Rear", region_rear, df_rear),
    ("Middle", region_middle, df_middle),
    ("Front", region_front, df_front),
]:
    print(f"\n{name} Module:")
    print("-" * 40)
    print(f"  Total LEDs in region: {len(region)}")
    print(
        f"  Mean Change%: {region['Height_Change_%'].mean():.3f}% (Overall: {df_full['Height_Change_%'].mean():.3f}%)"
    )
    print(
        f"  Std Change%: {region['Height_Change_%'].std():.3f}% (Overall: {df_full['Height_Change_%'].std():.3f}%)"
    )
    print(f"  Max Change%: {region['Height_Change_%'].max():.3f}%")
    print(f"  Min Change%: {region['Height_Change_%'].min():.3f}%")
    print(
        f"  LEDs with |change| > 0.5%: {len(region[abs(region['Height_Change_%']) > 0.5])} ({100*len(region[abs(region['Height_Change_%']) > 0.5])/len(region):.1f}%)"
    )
    print(
        f"  LEDs with |change| > 1.0%: {len(region[abs(region['Height_Change_%']) > 1.0])} ({100*len(region[abs(region['Height_Change_%']) > 1.0])/len(region):.1f}%)"
    )
    print(
        f"  LEDs with |change| > 1.5%: {len(region[abs(region['Height_Change_%']) > 1.5])} ({100*len(region[abs(region['Height_Change_%']) > 1.5])/len(region):.1f}%)"
    )
    print(
        f"  LEDs with |change| > 2.0%: {len(region[abs(region['Height_Change_%']) > 2.0])} ({100*len(region[abs(region['Height_Change_%']) > 2.0])/len(region):.1f}%)"
    )

# Create detailed plot
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle(
    "LED Region 7000-8000: Detailed Analysis Across All Modules",
    fontsize=16,
    fontweight="bold",
)

for idx, (name, region, color) in enumerate(
    [
        ("Rear", region_rear, "red"),
        ("Middle", region_middle, "green"),
        ("Front", region_front, "blue"),
    ]
):
    # Left panel: Amplitude comparison
    ax_left = axes[idx, 0]
    ax_left.plot(
        region["LED"],
        region["Ref_Height"],
        label="Reference",
        color=color,
        alpha=0.7,
        linewidth=1,
    )
    ax_left.plot(
        region["LED"],
        region["CT_Height"],
        label="Cross-Talk",
        color="orange",
        alpha=0.7,
        linewidth=1,
    )
    ax_left.set_xlabel("LED Number", fontsize=10)
    ax_left.set_ylabel("Pulse Height (Peak-Valley)", fontsize=10)
    ax_left.set_title(
        f"{name}: Amplitude Comparison (LED 7000-8000)", fontsize=11, fontweight="bold"
    )
    ax_left.legend()
    ax_left.grid(True, alpha=0.3)

    # Add statistics box
    stats_text = f"Mean Δ: {region['Height_Change_%'].mean():.3f}%\nStd: {region['Height_Change_%'].std():.3f}%\n|Δ|>1%: {len(region[abs(region['Height_Change_%']) > 1.0])}"
    ax_left.text(
        0.02,
        0.98,
        stats_text,
        transform=ax_left.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Right panel: Percentage change
    ax_right = axes[idx, 1]
    ax_right.scatter(region["LED"], region["Height_Change_%"], c=color, alpha=0.6, s=10)
    ax_right.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax_right.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, label="±0.5%")
    ax_right.axhline(y=-0.5, color="gray", linestyle=":", linewidth=0.8)
    ax_right.axhline(y=1.0, color="orange", linestyle=":", linewidth=0.8, label="±1.0%")
    ax_right.axhline(y=-1.0, color="orange", linestyle=":", linewidth=0.8)
    ax_right.set_xlabel("LED Number", fontsize=10)
    ax_right.set_ylabel("Percentage Change (%)", fontsize=10)
    ax_right.set_title(
        f"{name}: Change% (LED 7000-8000)", fontsize=11, fontweight="bold"
    )
    ax_right.set_ylim(-3, 3)
    ax_right.legend(fontsize=8)
    ax_right.grid(True, alpha=0.3)

    # Highlight outliers > 1%
    outliers = region[abs(region["Height_Change_%"]) > 1.0]
    if len(outliers) > 0:
        ax_right.scatter(
            outliers["LED"],
            outliers["Height_Change_%"],
            c="red",
            s=50,
            marker="o",
            edgecolors="black",
            linewidths=1,
            label=f"Outliers (n={len(outliers)})",
            zorder=5,
        )

plt.tight_layout()
plt.savefig(
    r"crosstalk_comparison_results\region_7000_8000_analysis.png",
    dpi=150,
    bbox_inches="tight",
)
print(f"\n\n✅ Plot saved: crosstalk_comparison_results\\region_7000_8000_analysis.png")
print("=" * 80)
