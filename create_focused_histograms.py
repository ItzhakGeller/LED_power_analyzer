import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_rear = pd.read_csv(r"crosstalk_comparison_results\Rear_pulse_height_only.csv")
df_middle = pd.read_csv(r"crosstalk_comparison_results\Middle_pulse_height_only.csv")
df_front = pd.read_csv(r"crosstalk_comparison_results\Front_pulse_height_only.csv")

# Define bins according to requirements: 0-0.5%, 0.5-1%, 1-1.5%, 1.5-2%, >2%
bins = [0, 0.5, 1.0, 1.5, 2.0, 100]
labels = ["0-0.5%", "0.5-1%", "1-1.5%", "1.5-2%", ">2%"]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "LED Change Distribution by Module (Focused on 0-2% Range)",
    fontsize=16,
    fontweight="bold",
)

colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

for idx, (name, df, ax) in enumerate(
    [
        ("Rear", df_rear, axes[0]),
        ("Middle", df_middle, axes[1]),
        ("Front", df_front, axes[2]),
    ]
):

    # Calculate absolute changes
    abs_changes = abs(df["Height_Change_%"])

    # Bin the data
    binned = pd.cut(abs_changes, bins=bins, labels=labels, include_lowest=True)
    counts = binned.value_counts().sort_index()
    percentages = counts / len(df) * 100

    # Create bar plot
    bars = ax.bar(
        range(len(labels)), percentages, color=colors, edgecolor="black", linewidth=1.5
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=0)
    ax.set_ylabel("Percentage of LEDs (%)", fontsize=11)
    ax.set_xlabel("Absolute Change Range", fontsize=11)
    ax.set_title(f"{name} Module (n={len(df):,} LEDs)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{int(count):,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add statistics box
    stats_text = f"Mean: {df['Height_Change_%'].mean():.3f}%\nStd: {df['Height_Change_%'].std():.3f}%\n≤0.5%: {percentages.iloc[0]:.1f}%\n>0.5%: {100-percentages.iloc[0]:.1f}%"
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor="lightyellow",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        ),
    )

plt.tight_layout()
plt.savefig(
    r"crosstalk_comparison_results\distribution_focused_histograms.png",
    dpi=150,
    bbox_inches="tight",
)
print("✅ Saved: distribution_focused_histograms.png")

# Create detailed histogram for each module (zoomed to ±2%)
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle(
    "LED Change% Distribution (Histogram - Focused on ±2% Range)",
    fontsize=16,
    fontweight="bold",
)

for idx, (name, df, ax, color) in enumerate(
    [
        ("Rear", df_rear, axes2[0], "red"),
        ("Middle", df_middle, axes2[1], "green"),
        ("Front", df_front, axes2[2], "blue"),
    ]
):

    # Create histogram with focus on center
    ax.hist(
        df["Height_Change_%"],
        bins=100,
        range=(-2, 2),
        color=color,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Zero Change")
    ax.axvline(
        x=0.5,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="±0.5% threshold",
    )
    ax.axvline(x=-0.5, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Percentage Change (%)", fontsize=11)
    ax.set_ylabel("Number of LEDs", fontsize=11)
    ax.set_title(f"{name} Module", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(-2, 2)

    # Add statistics
    mean_val = df["Height_Change_%"].mean()
    std_val = df["Height_Change_%"].std()
    within_half = len(df[abs(df["Height_Change_%"]) <= 0.5]) / len(df) * 100

    stats_text = f"μ = {mean_val:.3f}%\nσ = {std_val:.3f}%\n≤0.5%: {within_half:.1f}%"
    ax.text(
        0.02,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        ),
    )

plt.tight_layout()
plt.savefig(
    r"crosstalk_comparison_results\distribution_histograms_focused.png",
    dpi=150,
    bbox_inches="tight",
)
print("✅ Saved: distribution_histograms_focused.png")

print("\n" + "=" * 80)
print("Summary Statistics (Updated Bins)")
print("=" * 80)
for name, df in [("Rear", df_rear), ("Middle", df_middle), ("Front", df_front)]:
    abs_changes = abs(df["Height_Change_%"])
    binned = pd.cut(abs_changes, bins=bins, labels=labels, include_lowest=True)
    counts = binned.value_counts().sort_index()
    print(f"\n{name}:")
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count:5d} LEDs ({pct:5.2f}%)")
