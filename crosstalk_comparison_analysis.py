"""
Cross-Talk Comparison Analysis
==============================

This script analyzes cross-talk effects by comparing reference measurements
with cross-talk test measurements for Front, Middle (Center), and Rear Spark modules.

For each Spark module:
1. Runs complete LED analysis on both reference and cross-talk files
2. Compares LED intensities and intensity ratios
3. Generates comparative visualizations and statistical analysis
4. Provides summary of cross-talk effects

LED Numbering Ranges:
- Spark REAR (0):  LEDs 2304 - 15615  (13,312 LEDs)
- Spark CEN  (1):  LEDs 256  - 15615  (15,360 LEDs)
- Spark FRONT (2): LEDs 256  - 21247  (20,992 LEDs)

Author: Cross-Talk Analysis System
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import os

# Import the complete LED analyzer
from complete_led_analyzer import CompleteLEDAnalyzer

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_FOLDER = r"C:\Users\geller\OneDrive - HP Inc\data\manual LED power measurement analyzer\cross talk test"

# File configurations for cross-talk analysis
CROSSTALK_CONFIG = [
    {
        "name": "Rear",
        "ref_file": "Rear_Ref.bin",
        "crosstalk_file": "Rear_CrossTalk.bin",
        "first_led": 2304,
        "last_led": 15615,
        "expected_leds": 13312,
    },
    {
        "name": "Middle",
        "ref_file": "Midle_Ref.bin",
        "crosstalk_file": "Midle_CrossTalk.bin",
        "first_led": 256,
        "last_led": 15615,
        "expected_leds": 15360,
    },
    {
        "name": "Front",
        "ref_file": "Front_Ref.bin",
        "crosstalk_file": "Front_Crosstalk.bin",
        "first_led": 256,
        "last_led": 21247,
        "expected_leds": 20992,
    },
]

# Output directory
OUTPUT_DIR = Path("crosstalk_comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True)


class CrosstalkComparisonAnalyzer:
    """Analyzes cross-talk effects by comparing reference and cross-talk measurements"""

    def __init__(self, config, base_folder, output_dir):
        """Initialize the cross-talk comparison analyzer"""
        self.config = config
        self.name = config["name"]
        self.base_folder = Path(base_folder)
        self.output_dir = Path(output_dir)

        # Create module-specific output directory
        self.module_output_dir = self.output_dir / f"{self.name}_comparison"
        self.module_output_dir.mkdir(exist_ok=True)

        self.ref_analyzer = None
        self.crosstalk_analyzer = None
        self.comparison_results = None

    def run_analysis(self):
        """Run complete cross-talk comparison analysis"""
        print("\n" + "=" * 80)
        print(f"CROSS-TALK ANALYSIS: {self.name}")
        print("=" * 80)
        print(f"Reference file: {self.config['ref_file']}")
        print(f"Cross-talk file: {self.config['crosstalk_file']}")
        print(
            f"LED range: {self.config['first_led']} - {self.config['last_led']} ({self.config['expected_leds']} LEDs)"
        )
        print("=" * 80)

        # Analyze reference file
        print(f"\n{'='*60}")
        print(f"STEP 1: Analyzing Reference File - {self.config['ref_file']}")
        print(f"{'='*60}")
        ref_path = self.base_folder / self.config["ref_file"]

        # Close all existing plots to prevent display
        plt.close("all")

        self.ref_analyzer = CompleteLEDAnalyzer(
            str(ref_path), first_led_number=self.config["first_led"], show_plots=False
        )
        ref_success = self.ref_analyzer.run_complete_analysis()

        # Close all plots after analysis
        plt.close("all")

        if not ref_success:
            print(f"❌ Failed to analyze reference file for {self.name}")
            return False

        print(
            f"✅ Reference analysis complete: {len(self.ref_analyzer.detected_pulses)} LEDs detected"
        )

        # Analyze cross-talk file
        print(f"\n{'='*60}")
        print(f"STEP 2: Analyzing Cross-talk File - {self.config['crosstalk_file']}")
        print(f"{'='*60}")
        crosstalk_path = self.base_folder / self.config["crosstalk_file"]

        # Close all existing plots to prevent display
        plt.close("all")

        self.crosstalk_analyzer = CompleteLEDAnalyzer(
            str(crosstalk_path),
            first_led_number=self.config["first_led"],
            show_plots=False,
        )
        crosstalk_success = self.crosstalk_analyzer.run_complete_analysis()

        # Close all plots after analysis
        plt.close("all")

        if not crosstalk_success:
            print(f"❌ Failed to analyze cross-talk file for {self.name}")
            return False

        print(
            f"✅ Cross-talk analysis complete: {len(self.crosstalk_analyzer.detected_pulses)} LEDs detected"
        )

        # Compare results
        print(f"\n{'='*60}")
        print(f"STEP 3: Comparing Results and Analyzing Cross-talk Effects")
        print(f"{'='*60}")
        self.compare_results()

        # Generate visualizations
        self.generate_comparison_visualizations()

        # Generate summary report
        self.generate_summary_report()

        print(f"\n✅ Cross-talk analysis complete for {self.name}")
        print(f"📁 Results saved to: {self.module_output_dir}")

        return True

    def compare_results(self):
        """Compare reference and cross-talk measurements"""
        # Load CSV files to get peak_std data
        ref_csv_file = self.ref_analyzer.output_dir / f"{Path(self.ref_analyzer.bin_file_path).stem}_simplified_analysis.csv"
        ct_csv_file = self.crosstalk_analyzer.output_dir / f"{Path(self.crosstalk_analyzer.bin_file_path).stem}_simplified_analysis.csv"
        
        # Read CSV files (skip header lines starting with #)
        ref_df = pd.read_csv(ref_csv_file, comment='#')
        ct_df = pd.read_csv(ct_csv_file, comment='#')
        
        # Create dictionaries with all data including peak_std
        ref_pulses = self.ref_analyzer.detected_pulses
        crosstalk_pulses = self.crosstalk_analyzer.detected_pulses

        print(f"\n📊 Pulse Detection Summary:")
        print(f"   Reference: {len(ref_pulses)} pulses detected")
        print(f"   Cross-talk: {len(crosstalk_pulses)} pulses detected")

        # Match pulses by LED number and add peak_std from CSV
        ref_dict = {}
        for p in ref_pulses:
            led_num = p["led_number"]
            ref_dict[led_num] = p.copy()
            # Add peak_std from CSV
            peak_std_row = ref_df[ref_df['led_number'] == led_num]
            if not peak_std_row.empty:
                ref_dict[led_num]["peak_std"] = peak_std_row['peak_std'].values[0]
            else:
                ref_dict[led_num]["peak_std"] = 0.0
        
        crosstalk_dict = {}
        for p in crosstalk_pulses:
            led_num = p["led_number"]
            crosstalk_dict[led_num] = p.copy()
            # Add peak_std from CSV
            peak_std_row = ct_df[ct_df['led_number'] == led_num]
            if not peak_std_row.empty:
                crosstalk_dict[led_num]["peak_std"] = peak_std_row['peak_std'].values[0]
            else:
                crosstalk_dict[led_num]["peak_std"] = 0.0

        # Find common LED numbers
        common_leds = sorted(set(ref_dict.keys()) & set(crosstalk_dict.keys()))
        print(f"   Common LEDs: {len(common_leds)}")

        if len(common_leds) == 0:
            print("⚠️ No common LEDs found between reference and cross-talk data")
            return

        # Compare measurements for common LEDs
        comparison_data = []

        for led_num in common_leds:
            ref_pulse = ref_dict[led_num]
            ct_pulse = crosstalk_dict[led_num]

            # Calculate differences and ratios
            height_diff = ct_pulse["pulse_height"] - ref_pulse["pulse_height"]
            height_ratio = (
                ct_pulse["pulse_height"] / ref_pulse["pulse_height"]
                if ref_pulse["pulse_height"] > 0
                else 0
            )
            height_change_pct = (
                (height_diff / ref_pulse["pulse_height"]) * 100
                if ref_pulse["pulse_height"] > 0
                else 0
            )

            peak_diff = ct_pulse["peak_value"] - ref_pulse["peak_value"]
            peak_ratio = (
                ct_pulse["peak_value"] / ref_pulse["peak_value"]
                if ref_pulse["peak_value"] > 0
                else 0
            )
            peak_change_pct = (
                (peak_diff / ref_pulse["peak_value"]) * 100
                if ref_pulse["peak_value"] > 0
                else 0
            )

            valley_diff = ct_pulse["valley_value"] - ref_pulse["valley_value"]

            snr_diff = ct_pulse["snr"] - ref_pulse["snr"]
            snr_change_pct = (
                (snr_diff / ref_pulse["snr"]) * 100 if ref_pulse["snr"] > 0 else 0
            )

            # Peak stability (STD) comparison
            ref_peak_std = ref_pulse.get("peak_std", 0)
            ct_peak_std = ct_pulse.get("peak_std", 0)
            peak_std_diff = ct_peak_std - ref_peak_std
            peak_std_ratio = ct_peak_std / ref_peak_std if ref_peak_std > 0 else 0
            peak_std_change_pct = (
                (peak_std_diff / ref_peak_std) * 100 if ref_peak_std > 0 else 0
            )

            comparison_data.append(
                {
                    "LED": led_num,
                    "Ref_Height": ref_pulse["pulse_height"],
                    "CT_Height": ct_pulse["pulse_height"],
                    "Height_Diff": height_diff,
                    "Height_Ratio": height_ratio,
                    "Height_Change_%": height_change_pct,
                    "Ref_Peak": ref_pulse["peak_value"],
                    "CT_Peak": ct_pulse["peak_value"],
                    "Peak_Diff": peak_diff,
                    "Peak_Ratio": peak_ratio,
                    "Peak_Change_%": peak_change_pct,
                    "Ref_Valley": ref_pulse["valley_value"],
                    "CT_Valley": ct_pulse["valley_value"],
                    "Valley_Diff": valley_diff,
                    "Ref_SNR": ref_pulse["snr"],
                    "CT_SNR": ct_pulse["snr"],
                    "SNR_Diff": snr_diff,
                    "SNR_Change_%": snr_change_pct,
                    "Ref_Width": ref_pulse["pulse_width"],
                    "CT_Width": ct_pulse["pulse_width"],
                    "Ref_Peak_STD": ref_peak_std,
                    "CT_Peak_STD": ct_peak_std,
                    "Peak_STD_Diff": peak_std_diff,
                    "Peak_STD_Ratio": peak_std_ratio,
                    "Peak_STD_Change_%": peak_std_change_pct,
                }
            )

        self.comparison_results = pd.DataFrame(comparison_data)

        # Save comparison data
        comparison_file = self.module_output_dir / "detailed_comparison.csv"
        self.comparison_results.to_csv(comparison_file, index=False)
        print(f"   ✅ Detailed comparison saved: {comparison_file}")

        # Calculate statistics
        print(f"\n📈 Statistical Summary:")
        print(f"\n   Pulse Height Changes:")
        print(
            f"      Mean change: {self.comparison_results['Height_Change_%'].mean():.2f}%"
        )
        print(
            f"      Std deviation: {self.comparison_results['Height_Change_%'].std():.2f}%"
        )
        print(
            f"      Min change: {self.comparison_results['Height_Change_%'].min():.2f}%"
        )
        print(
            f"      Max change: {self.comparison_results['Height_Change_%'].max():.2f}%"
        )
        print(
            f"      Median change: {self.comparison_results['Height_Change_%'].median():.2f}%"
        )

        print(f"\n   Peak Value Changes:")
        print(
            f"      Mean change: {self.comparison_results['Peak_Change_%'].mean():.2f}%"
        )
        print(
            f"      Std deviation: {self.comparison_results['Peak_Change_%'].std():.2f}%"
        )
        print(
            f"      Min change: {self.comparison_results['Peak_Change_%'].min():.2f}%"
        )
        print(
            f"      Max change: {self.comparison_results['Peak_Change_%'].max():.2f}%"
        )

        print(f"\n   SNR Changes:")
        print(
            f"      Mean change: {self.comparison_results['SNR_Change_%'].mean():.2f}%"
        )
        print(
            f"      Std deviation: {self.comparison_results['SNR_Change_%'].std():.2f}%"
        )

        print(f"\n   Peak Stability (STD) Changes:")
        print(
            f"      Mean Ref Peak STD: {self.comparison_results['Ref_Peak_STD'].mean():.4f}"
        )
        print(
            f"      Mean CT Peak STD: {self.comparison_results['CT_Peak_STD'].mean():.4f}"
        )
        print(
            f"      Mean STD change: {self.comparison_results['Peak_STD_Change_%'].mean():.2f}%"
        )
        print(
            f"      Std deviation: {self.comparison_results['Peak_STD_Change_%'].std():.2f}%"
        )
        print(
            f"      Min change: {self.comparison_results['Peak_STD_Change_%'].min():.2f}%"
        )
        print(
            f"      Max change: {self.comparison_results['Peak_STD_Change_%'].max():.2f}%"
        )

        # Count significant changes
        significant_height_changes = len(
            self.comparison_results[abs(self.comparison_results["Height_Change_%"]) > 5]
        )
        significant_peak_changes = len(
            self.comparison_results[abs(self.comparison_results["Peak_Change_%"]) > 5]
        )
        significant_std_changes = len(
            self.comparison_results[
                abs(self.comparison_results["Peak_STD_Change_%"]) > 10
            ]
        )

        print(f"\n   Significant Changes:")
        print(
            f"      Height changes (>5%): {significant_height_changes} LEDs ({(significant_height_changes/len(common_leds)*100):.1f}%)"
        )
        print(
            f"      Peak changes (>5%): {significant_peak_changes} LEDs ({(significant_peak_changes/len(common_leds)*100):.1f}%)"
        )
        print(
            f"      Peak STD changes (>10%): {significant_std_changes} LEDs ({(significant_std_changes/len(common_leds)*100):.1f}%)"
        )

    def generate_comparison_visualizations(self):
        """Generate comparative visualizations"""
        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("⚠️ No comparison data to visualize")
            return

        print(f"\n📊 Generating comparison visualizations...")

        # Figure 1: Height and Peak comparison scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Height comparison scatter
        ax = axes[0, 0]
        ax.scatter(
            self.comparison_results["Ref_Height"],
            self.comparison_results["CT_Height"],
            alpha=0.5,
            s=20,
        )
        ax.plot(
            [
                self.comparison_results["Ref_Height"].min(),
                self.comparison_results["Ref_Height"].max(),
            ],
            [
                self.comparison_results["Ref_Height"].min(),
                self.comparison_results["Ref_Height"].max(),
            ],
            "r--",
            label="No change line",
        )
        ax.set_xlabel("Reference Height")
        ax.set_ylabel("Cross-talk Height")
        ax.set_title(
            f"{self.name}: Pulse Height Comparison\n(Points above red line = increased intensity)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Height change percentage histogram
        ax = axes[0, 1]
        ax.hist(
            self.comparison_results["Height_Change_%"],
            bins=50,
            edgecolor="black",
            alpha=0.7,
        )
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="No change")
        ax.axvline(
            x=self.comparison_results["Height_Change_%"].mean(),
            color="g",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.comparison_results['Height_Change_%'].mean():.2f}%",
        )
        ax.set_xlabel("Height Change (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{self.name}: Distribution of Height Changes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Peak comparison scatter
        ax = axes[1, 0]
        ax.scatter(
            self.comparison_results["Ref_Peak"],
            self.comparison_results["CT_Peak"],
            alpha=0.5,
            s=20,
        )
        ax.plot(
            [
                self.comparison_results["Ref_Peak"].min(),
                self.comparison_results["Ref_Peak"].max(),
            ],
            [
                self.comparison_results["Ref_Peak"].min(),
                self.comparison_results["Ref_Peak"].max(),
            ],
            "r--",
            label="No change line",
        )
        ax.set_xlabel("Reference Peak")
        ax.set_ylabel("Cross-talk Peak")
        ax.set_title(
            f"{self.name}: Peak Value Comparison\n(Points above red line = increased intensity)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Peak change percentage histogram
        ax = axes[1, 1]
        ax.hist(
            self.comparison_results["Peak_Change_%"],
            bins=50,
            edgecolor="black",
            alpha=0.7,
        )
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="No change")
        ax.axvline(
            x=self.comparison_results["Peak_Change_%"].mean(),
            color="g",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.comparison_results['Peak_Change_%'].mean():.2f}%",
        )
        ax.set_xlabel("Peak Change (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{self.name}: Distribution of Peak Changes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig1_file = self.module_output_dir / "comparison_scatter_histograms.png"
        plt.savefig(fig1_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {fig1_file.name}")

        # Figure 2: Changes vs LED number
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # Height change vs LED number
        ax = axes[0]
        ax.plot(
            self.comparison_results["LED"],
            self.comparison_results["Height_Change_%"],
            "b-",
            alpha=0.6,
            linewidth=1,
        )
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.axhline(
            y=5, color="orange", linestyle=":", linewidth=1, label="±5% threshold"
        )
        ax.axhline(y=-5, color="orange", linestyle=":", linewidth=1)
        ax.set_xlabel("LED Number")
        ax.set_ylabel("Height Change (%)")
        ax.set_title(f"{self.name}: Pulse Height Change vs LED Number")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Peak change vs LED number
        ax = axes[1]
        ax.plot(
            self.comparison_results["LED"],
            self.comparison_results["Peak_Change_%"],
            "g-",
            alpha=0.6,
            linewidth=1,
        )
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.axhline(
            y=5, color="orange", linestyle=":", linewidth=1, label="±5% threshold"
        )
        ax.axhline(y=-5, color="orange", linestyle=":", linewidth=1)
        ax.set_xlabel("LED Number")
        ax.set_ylabel("Peak Change (%)")
        ax.set_title(f"{self.name}: Peak Value Change vs LED Number")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # SNR change vs LED number
        ax = axes[2]
        ax.plot(
            self.comparison_results["LED"],
            self.comparison_results["SNR_Change_%"],
            "purple",
            alpha=0.6,
            linewidth=1,
        )
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("LED Number")
        ax.set_ylabel("SNR Change (%)")
        ax.set_title(f"{self.name}: SNR Change vs LED Number")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2_file = self.module_output_dir / "changes_vs_led_number.png"
        plt.savefig(fig2_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {fig2_file.name}")

        # Figure 3: Ratio comparisons
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Height ratio vs LED number
        ax = axes[0]
        ax.plot(
            self.comparison_results["LED"],
            self.comparison_results["Height_Ratio"],
            "b-",
            alpha=0.6,
            linewidth=1,
        )
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=2, label="Ratio = 1.0")
        ax.axhline(
            y=1.05,
            color="orange",
            linestyle=":",
            linewidth=1,
            label="±5% threshold",
        )
        ax.axhline(y=0.95, color="orange", linestyle=":", linewidth=1)
        ax.set_xlabel("LED Number")
        ax.set_ylabel("Height Ratio (CT/Ref)")
        ax.set_title(
            f"{self.name}: Pulse Height Ratio vs LED Number\n(Ratio > 1.0 = increased intensity)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Peak ratio vs LED number
        ax = axes[1]
        ax.plot(
            self.comparison_results["LED"],
            self.comparison_results["Peak_Ratio"],
            "g-",
            alpha=0.6,
            linewidth=1,
        )
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=2, label="Ratio = 1.0")
        ax.axhline(
            y=1.05,
            color="orange",
            linestyle=":",
            linewidth=1,
            label="±5% threshold",
        )
        ax.axhline(y=0.95, color="orange", linestyle=":", linewidth=1)
        ax.set_xlabel("LED Number")
        ax.set_ylabel("Peak Ratio (CT/Ref)")
        ax.set_title(
            f"{self.name}: Peak Value Ratio vs LED Number\n(Ratio > 1.0 = increased intensity)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig3_file = self.module_output_dir / "ratio_comparisons.png"
        plt.savefig(fig3_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {fig3_file.name}")

        # Figure 4: Peak Stability (STD) Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Peak STD comparison scatter
        ax = axes[0, 0]
        ax.scatter(
            self.comparison_results["Ref_Peak_STD"],
            self.comparison_results["CT_Peak_STD"],
            alpha=0.5,
            s=20,
            c="purple",
        )
        ax.plot(
            [
                self.comparison_results["Ref_Peak_STD"].min(),
                self.comparison_results["Ref_Peak_STD"].max(),
            ],
            [
                self.comparison_results["Ref_Peak_STD"].min(),
                self.comparison_results["Ref_Peak_STD"].max(),
            ],
            "r--",
            label="No change line",
        )
        ax.set_xlabel("Reference Peak STD")
        ax.set_ylabel("Cross-talk Peak STD")
        ax.set_title(
            f"{self.name}: Peak Stability Comparison\n(Points above red line = decreased stability)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Peak STD change percentage histogram
        ax = axes[0, 1]
        ax.hist(
            self.comparison_results["Peak_STD_Change_%"],
            bins=50,
            edgecolor="black",
            alpha=0.7,
            color="purple",
        )
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="No change")
        ax.axvline(
            x=self.comparison_results["Peak_STD_Change_%"].mean(),
            color="g",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.comparison_results['Peak_STD_Change_%'].mean():.2f}%",
        )
        ax.set_xlabel("Peak STD Change (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{self.name}: Distribution of Peak Stability Changes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Peak STD change vs LED number
        ax = axes[1, 0]
        ax.plot(
            self.comparison_results["LED"],
            self.comparison_results["Peak_STD_Change_%"],
            "purple",
            alpha=0.6,
            linewidth=1,
        )
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.axhline(
            y=10, color="orange", linestyle=":", linewidth=1, label="±10% threshold"
        )
        ax.axhline(y=-10, color="orange", linestyle=":", linewidth=1)
        ax.set_xlabel("LED Number")
        ax.set_ylabel("Peak STD Change (%)")
        ax.set_title(
            f"{self.name}: Peak Stability Change vs LED Number\n(Positive = Increased noise/instability)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Peak STD ratio vs LED number
        ax = axes[1, 1]
        ax.plot(
            self.comparison_results["LED"],
            self.comparison_results["Peak_STD_Ratio"],
            "purple",
            alpha=0.6,
            linewidth=1,
        )
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=2, label="Ratio = 1.0")
        ax.axhline(
            y=1.1,
            color="orange",
            linestyle=":",
            linewidth=1,
            label="±10% threshold",
        )
        ax.axhline(y=0.9, color="orange", linestyle=":", linewidth=1)
        ax.set_xlabel("LED Number")
        ax.set_ylabel("Peak STD Ratio (CT/Ref)")
        ax.set_title(
            f"{self.name}: Peak Stability Ratio vs LED Number\n(Ratio > 1.0 = Increased instability)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig4_file = self.module_output_dir / "peak_stability_analysis.png"
        plt.savefig(fig4_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {fig4_file.name}")

    def generate_summary_report(self):
        """Generate a summary report of the cross-talk analysis"""
        if self.comparison_results is None or len(self.comparison_results) == 0:
            print("⚠️ No comparison data to report")
            return

        report_file = self.module_output_dir / "summary_report.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"CROSS-TALK ANALYSIS SUMMARY REPORT: {self.name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("FILE INFORMATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Reference file: {self.config['ref_file']}\n")
            f.write(f"Cross-talk file: {self.config['crosstalk_file']}\n")
            f.write(
                f"LED range: {self.config['first_led']} - {self.config['last_led']}\n"
            )
            f.write(f"Expected LEDs: {self.config['expected_leds']}\n\n")

            f.write("DETECTION SUMMARY:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Reference LEDs detected: {len(self.ref_analyzer.detected_pulses)}\n"
            )
            f.write(
                f"Cross-talk LEDs detected: {len(self.crosstalk_analyzer.detected_pulses)}\n"
            )
            f.write(f"Common LEDs analyzed: {len(self.comparison_results)}\n\n")

            f.write("PULSE HEIGHT ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Mean height change: {self.comparison_results['Height_Change_%'].mean():.2f}%\n"
            )
            f.write(
                f"Std deviation: {self.comparison_results['Height_Change_%'].std():.2f}%\n"
            )
            f.write(
                f"Min change: {self.comparison_results['Height_Change_%'].min():.2f}%\n"
            )
            f.write(
                f"Max change: {self.comparison_results['Height_Change_%'].max():.2f}%\n"
            )
            f.write(
                f"Median change: {self.comparison_results['Height_Change_%'].median():.2f}%\n"
            )

            significant_height = len(
                self.comparison_results[
                    abs(self.comparison_results["Height_Change_%"]) > 5
                ]
            )
            f.write(
                f"\nLEDs with >5% height change: {significant_height} ({(significant_height/len(self.comparison_results)*100):.1f}%)\n\n"
            )

            f.write("PEAK VALUE ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Mean peak change: {self.comparison_results['Peak_Change_%'].mean():.2f}%\n"
            )
            f.write(
                f"Std deviation: {self.comparison_results['Peak_Change_%'].std():.2f}%\n"
            )
            f.write(
                f"Min change: {self.comparison_results['Peak_Change_%'].min():.2f}%\n"
            )
            f.write(
                f"Max change: {self.comparison_results['Peak_Change_%'].max():.2f}%\n"
            )

            significant_peak = len(
                self.comparison_results[
                    abs(self.comparison_results["Peak_Change_%"]) > 5
                ]
            )
            f.write(
                f"\nLEDs with >5% peak change: {significant_peak} ({(significant_peak/len(self.comparison_results)*100):.1f}%)\n\n"
            )

            f.write("SNR ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Mean SNR change: {self.comparison_results['SNR_Change_%'].mean():.2f}%\n"
            )
            f.write(
                f"Std deviation: {self.comparison_results['SNR_Change_%'].std():.2f}%\n"
            )
            f.write(
                f"Reference mean SNR: {self.comparison_results['Ref_SNR'].mean():.2f}\n"
            )
            f.write(
                f"Cross-talk mean SNR: {self.comparison_results['CT_SNR'].mean():.2f}\n\n"
            )

            f.write("PEAK STABILITY (STD) ANALYSIS:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Reference mean Peak STD: {self.comparison_results['Ref_Peak_STD'].mean():.6f}\n"
            )
            f.write(
                f"Cross-talk mean Peak STD: {self.comparison_results['CT_Peak_STD'].mean():.6f}\n"
            )
            f.write(
                f"Mean Peak STD change: {self.comparison_results['Peak_STD_Change_%'].mean():.2f}%\n"
            )
            f.write(
                f"Std deviation: {self.comparison_results['Peak_STD_Change_%'].std():.2f}%\n"
            )
            f.write(
                f"Min change: {self.comparison_results['Peak_STD_Change_%'].min():.2f}%\n"
            )
            f.write(
                f"Max change: {self.comparison_results['Peak_STD_Change_%'].max():.2f}%\n"
            )

            significant_std = len(
                self.comparison_results[
                    abs(self.comparison_results["Peak_STD_Change_%"]) > 10
                ]
            )
            f.write(
                f"\nLEDs with >10% Peak STD change: {significant_std} ({(significant_std/len(self.comparison_results)*100):.1f}%)\n"
            )

            # Stability assessment
            mean_std_change = self.comparison_results["Peak_STD_Change_%"].mean()
            if mean_std_change > 20:
                stability_status = (
                    "SIGNIFICANTLY DEGRADED - Cross-talk causes major instability"
                )
            elif mean_std_change > 10:
                stability_status = "DEGRADED - Noticeable increase in peak instability"
            elif mean_std_change > 5:
                stability_status = (
                    "SLIGHTLY DEGRADED - Minor increase in peak variations"
                )
            elif mean_std_change > -5:
                stability_status = "STABLE - No significant change in peak stability"
            else:
                stability_status = "IMPROVED - Peak stability has improved (unusual)"

            f.write(f"\nStability Assessment: {stability_status}\n\n")

            # Cross-talk effect assessment
            f.write("CROSS-TALK EFFECT ASSESSMENT:\n")
            f.write("-" * 80 + "\n")

            mean_height_change = abs(self.comparison_results["Height_Change_%"].mean())
            mean_peak_change = abs(self.comparison_results["Peak_Change_%"].mean())

            if mean_height_change < 1 and mean_peak_change < 1:
                assessment = "MINIMAL - Cross-talk effects are negligible (<1%)"
            elif mean_height_change < 3 and mean_peak_change < 3:
                assessment = "LOW - Small cross-talk effects detected (1-3%)"
            elif mean_height_change < 5 and mean_peak_change < 5:
                assessment = "MODERATE - Noticeable cross-talk effects (3-5%)"
            else:
                assessment = "SIGNIFICANT - Cross-talk effects require attention (>5%)"

            f.write(f"Overall assessment: {assessment}\n\n")

            # Top 10 most affected LEDs
            f.write("TOP 10 MOST AFFECTED LEDS (by height change):\n")
            f.write("-" * 80 + "\n")
            top_affected = self.comparison_results.nlargest(
                10, "Height_Change_%", keep="all"
            )[
                [
                    "LED",
                    "Height_Change_%",
                    "Peak_Change_%",
                    "Peak_STD_Change_%",
                    "Ref_Height",
                    "CT_Height",
                    "SNR_Change_%",
                ]
            ]
            f.write(top_affected.to_string(index=False) + "\n\n")

            # Bottom 10 most affected LEDs (negative changes)
            f.write("TOP 10 MOST DECREASED LEDS (by height change):\n")
            f.write("-" * 80 + "\n")
            bottom_affected = self.comparison_results.nsmallest(
                10, "Height_Change_%", keep="all"
            )[
                [
                    "LED",
                    "Height_Change_%",
                    "Peak_Change_%",
                    "Peak_STD_Change_%",
                    "Ref_Height",
                    "CT_Height",
                    "SNR_Change_%",
                ]
            ]
            f.write(bottom_affected.to_string(index=False) + "\n\n")

            # Top 10 most unstable LEDs (by Peak STD change)
            f.write("TOP 10 MOST UNSTABLE LEDS (by Peak STD increase):\n")
            f.write("-" * 80 + "\n")
            most_unstable = self.comparison_results.nlargest(
                10, "Peak_STD_Change_%", keep="all"
            )[
                [
                    "LED",
                    "Peak_STD_Change_%",
                    "Ref_Peak_STD",
                    "CT_Peak_STD",
                    "Height_Change_%",
                    "Peak_Change_%",
                ]
            ]
            f.write(most_unstable.to_string(index=False) + "\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"   ✅ Summary report saved: {report_file}")


def run_complete_crosstalk_analysis():
    """Run cross-talk analysis for all configured modules"""
    print("\n" + "=" * 80)
    print("COMPLETE CROSS-TALK ANALYSIS SYSTEM")
    print("=" * 80)
    print(f"Base folder: {BASE_FOLDER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Modules to analyze: {len(CROSSTALK_CONFIG)}")
    print("=" * 80)

    results = []

    for config in CROSSTALK_CONFIG:
        analyzer = CrosstalkComparisonAnalyzer(config, BASE_FOLDER, OUTPUT_DIR)
        success = analyzer.run_analysis()

        if success:
            results.append(
                {
                    "name": config["name"],
                    "analyzer": analyzer,
                    "success": True,
                }
            )
        else:
            results.append(
                {
                    "name": config["name"],
                    "analyzer": None,
                    "success": False,
                }
            )

    # Generate overall summary
    print("\n" + "=" * 80)
    print("OVERALL CROSS-TALK ANALYSIS SUMMARY")
    print("=" * 80)

    summary_file = OUTPUT_DIR / "overall_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OVERALL CROSS-TALK ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result in results:
            f.write(f"\n{result['name']}:\n")
            f.write("-" * 40 + "\n")

            if result["success"]:
                analyzer = result["analyzer"]
                comp_results = analyzer.comparison_results

                mean_height_change = comp_results["Height_Change_%"].mean()
                mean_peak_change = comp_results["Peak_Change_%"].mean()
                mean_std_change = comp_results["Peak_STD_Change_%"].mean()
                significant_changes = len(
                    comp_results[abs(comp_results["Height_Change_%"]) > 5]
                )
                significant_std_changes = len(
                    comp_results[abs(comp_results["Peak_STD_Change_%"]) > 10]
                )

                f.write(f"✅ Analysis successful\n")
                f.write(f"LEDs analyzed: {len(comp_results)}\n")
                f.write(f"Mean height change: {mean_height_change:.2f}%\n")
                f.write(f"Mean peak change: {mean_peak_change:.2f}%\n")
                f.write(f"Mean Peak STD change: {mean_std_change:.2f}%\n")
                f.write(
                    f"LEDs with >5% amplitude change: {significant_changes} ({(significant_changes/len(comp_results)*100):.1f}%)\n"
                )
                f.write(
                    f"LEDs with >10% stability change: {significant_std_changes} ({(significant_std_changes/len(comp_results)*100):.1f}%)\n"
                )

                print(f"\n{result['name']}:")
                print(f"   ✅ Analysis successful")
                print(f"   LEDs analyzed: {len(comp_results)}")
                print(f"   Mean height change: {mean_height_change:.2f}%")
                print(f"   Mean peak change: {mean_peak_change:.2f}%")
                print(f"   Mean Peak STD change: {mean_std_change:.2f}%")
                print(
                    f"   LEDs with >5% amplitude change: {significant_changes} ({(significant_changes/len(comp_results)*100):.1f}%)"
                )
                print(
                    f"   LEDs with >10% stability change: {significant_std_changes} ({(significant_std_changes/len(comp_results)*100):.1f}%)"
                )
            else:
                f.write(f"❌ Analysis failed\n")
                print(f"\n{result['name']}:")
                print(f"   ❌ Analysis failed")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSIONS:\n")
        f.write("-" * 80 + "\n")

        successful_results = [r for r in results if r["success"]]

        if len(successful_results) > 0:
            all_mean_changes = [
                abs(r["analyzer"].comparison_results["Height_Change_%"].mean())
                for r in successful_results
            ]
            overall_mean = np.mean(all_mean_changes)

            f.write(
                f"\nOverall mean height change across all modules: {overall_mean:.2f}%\n\n"
            )

            if overall_mean < 1:
                conclusion = "Cross-talk effects are MINIMAL across all tested modules. The intensity variations are within normal measurement noise (<1%)."
            elif overall_mean < 3:
                conclusion = "Cross-talk effects are LOW but detectable. Changes are small (1-3%) and may not be significant for most applications."
            elif overall_mean < 5:
                conclusion = "Cross-talk effects are MODERATE. Changes (3-5%) are noticeable and should be considered in calibration."
            else:
                conclusion = "Cross-talk effects are SIGNIFICANT. Changes (>5%) require attention and may impact system performance."

            f.write(f"Assessment: {conclusion}\n")
            print(f"\n{'='*80}")
            print("OVERALL ASSESSMENT:")
            print(f"{'='*80}")
            print(f"Overall mean height change: {overall_mean:.2f}%")
            print(f"\n{conclusion}")
        else:
            f.write("No successful analyses to summarize.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\n✅ Overall summary saved: {summary_file}")
    print(f"📁 All results saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 80)
    print("CROSS-TALK ANALYSIS COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_complete_crosstalk_analysis()
