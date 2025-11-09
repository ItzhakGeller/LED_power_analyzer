"""
Simple Single File LED Pulse Analyzer
======================================

This script analyzes a SINGLE BIN file for LED pulses.
Just edit the parameters below and run!

Author: LED Analysis System
Date: October 2025
"""

import os
from complete_led_analyzer import CompleteLEDAnalyzer

# ============================================================================
# CONFIGURATION - Edit these parameters for your analysis
# ============================================================================

# Path to your BIN file
BIN_FILE_PATH = (
    r"C:\Users\geller\OneDrive - HP Inc\data\manual LED power measurement analyzer"
)
BIN_FILE_NAME = "kedem_alpha_95.bin"
# First LED number in the file
FIRST_LED_NUMBER = 0
LAST_LED_NUMBER = 21503
# Show plots during analysis? (True/False)
SHOW_PLOTS = True

# ============================================================================
# Auto-calculated values - Don't change below
# ============================================================================
FULL_BIN_FILE_PATH = os.path.join(BIN_FILE_PATH, BIN_FILE_NAME)
EXPECTED_LEDS = LAST_LED_NUMBER - FIRST_LED_NUMBER + 1

# ============================================================================
# Run Analysis
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SINGLE FILE LED PULSE ANALYSIS")
    print("=" * 80)
    print(f"📁 Folder: {BIN_FILE_PATH}")
    print(f"� File: {BIN_FILE_NAME}")
    print(f"📍 Full path: {FULL_BIN_FILE_PATH}")
    print(f"🔢 LED range: {FIRST_LED_NUMBER} to {LAST_LED_NUMBER}")
    print(f"📊 Expected LEDs: {EXPECTED_LEDS}")
    print(f"📊 Show plots: {SHOW_PLOTS}")
    print("=" * 80)

    # Create analyzer
    analyzer = CompleteLEDAnalyzer(
        bin_file_path=FULL_BIN_FILE_PATH,
        first_led_number=FIRST_LED_NUMBER,
        show_plots=SHOW_PLOTS,
    )

    # Run complete analysis
    success = analyzer.run_complete_analysis()

    if success:
        detected_leds = len(analyzer.detected_pulses)
        actual_last_led = FIRST_LED_NUMBER + detected_leds - 1
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 Total pulses detected: {detected_leds}")
        print(f"🎯 Expected LEDs: {EXPECTED_LEDS}")
        print(f"🎯 Actual LED range: {FIRST_LED_NUMBER} to {actual_last_led}")
        print(f"✓  Match: {'YES ✅' if detected_leds == EXPECTED_LEDS else 'NO ⚠️'}")
        print(f"📁 Results saved in: {analyzer.output_dir}")
        print("=" * 80)
        
        # Create summary plots
        print("\n📊 Creating Summary Plots...")
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get data
        led_numbers = [p["led_number"] for p in analyzer.detected_pulses]
        absolute_heights = [p["absolute_height"] for p in analyzer.detected_pulses]
        normalized_heights = [p["normalized_height"] for p in analyzer.detected_pulses]
        pulse_widths = [p["pulse_width"] for p in analyzer.detected_pulses]
        inter_distances = [p.get("inter_pulse_distance", 0) for p in analyzer.detected_pulses[1:]]
        
        # Create comprehensive summary figure
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Absolute Height vs LED Number
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(led_numbers, absolute_heights, 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_title('Absolute Height vs LED Number', fontweight='bold', fontsize=12)
        ax1.set_xlabel('LED Number')
        ax1.set_ylabel('Absolute Height')
        ax1.grid(True, alpha=0.3)
        
        # 2. Normalized Height vs LED Number
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(led_numbers, normalized_heights, 'g-', linewidth=0.5, alpha=0.7)
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Average (1.0)')
        ax2.set_title('Normalized Height vs LED Number', fontweight='bold', fontsize=12)
        ax2.set_xlabel('LED Number')
        ax2.set_ylabel('Normalized Height')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Pulse Width Distribution
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(pulse_widths, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(x=np.mean(pulse_widths), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(pulse_widths):.1f}')
        ax3.set_title('Pulse Width Distribution', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Pulse Width (samples)')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Inter-Pulse Distance Distribution
        ax4 = plt.subplot(3, 3, 4)
        if inter_distances:
            ax4.hist(inter_distances, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(x=np.mean(inter_distances), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(inter_distances):.1f}')
            ax4.set_title('Inter-Pulse Distance Distribution', fontweight='bold', fontsize=12)
            ax4.set_xlabel('Distance (samples)')
            ax4.set_ylabel('Count')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Sample Pulses (First 10)
        ax5 = plt.subplot(3, 3, 5)
        sample_pulses = analyzer.detected_pulses[:10]
        for i, pulse in enumerate(sample_pulses):
            start = pulse["start_position"]
            end = pulse["end_position"]
            center = pulse["center_position"]
            
            # Get data range
            start_clean = start - analyzer.valid_start
            end_clean = end - analyzer.valid_start
            
            if 0 <= start_clean < len(analyzer.clean_data) and 0 <= end_clean <= len(analyzer.clean_data):
                x = np.arange(start, end)
                y = analyzer.clean_data[start_clean:end_clean]
                ax5.plot(x, y, linewidth=1, alpha=0.7, label=f'LED {pulse["led_number"]}')
        
        ax5.axhline(y=analyzer.threshold, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Threshold')
        ax5.set_title('Sample Pulses (First 10 LEDs)', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Sample Position')
        ax5.set_ylabel('Amplitude')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Pulse Width vs LED Number (check for anomalies)
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(led_numbers, pulse_widths, 'mo', markersize=2, alpha=0.5)
        mean_width = np.mean(pulse_widths)
        std_width = np.std(pulse_widths)
        ax6.axhline(y=mean_width, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_width:.1f}')
        ax6.axhline(y=mean_width + 2*std_width, color='orange', linestyle=':', linewidth=2, label=f'+2σ')
        ax6.axhline(y=mean_width - 2*std_width, color='orange', linestyle=':', linewidth=2, label=f'-2σ')
        ax6.set_title('Pulse Width vs LED (Anomaly Detection)', fontweight='bold', fontsize=12)
        ax6.set_xlabel('LED Number')
        ax6.set_ylabel('Pulse Width (samples)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Inter-Distance vs LED Number (check for spacing anomalies)
        ax7 = plt.subplot(3, 3, 7)
        if inter_distances:
            ax7.plot(led_numbers[1:], inter_distances, 'co', markersize=2, alpha=0.5)
            mean_dist = np.mean(inter_distances)
            std_dist = np.std(inter_distances)
            ax7.axhline(y=mean_dist, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.1f}')
            ax7.axhline(y=mean_dist + 2*std_dist, color='orange', linestyle=':', linewidth=2, label=f'+2σ')
            ax7.axhline(y=mean_dist - 2*std_dist, color='orange', linestyle=':', linewidth=2, label=f'-2σ')
            ax7.set_title('Inter-Pulse Distance vs LED (Spacing Anomaly)', fontweight='bold', fontsize=12)
            ax7.set_xlabel('LED Number')
            ax7.set_ylabel('Distance (samples)')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Absolute Height Distribution
        ax8 = plt.subplot(3, 3, 8)
        ax8.hist(absolute_heights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax8.axvline(x=np.mean(absolute_heights), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(absolute_heights):.4f}')
        ax8.set_title('Absolute Height Distribution', fontweight='bold', fontsize=12)
        ax8.set_xlabel('Absolute Height')
        ax8.set_ylabel('Count')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Statistics Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate anomalies
        width_anomalies = np.sum(np.abs(pulse_widths - mean_width) > 2*std_width)
        if inter_distances:
            dist_anomalies = np.sum(np.abs(inter_distances - mean_dist) > 2*std_dist)
        else:
            dist_anomalies = 0
        
        stats_text = f"""STATISTICAL SUMMARY

LED Information:
• Total LEDs: {detected_leds}
• LED Range: {FIRST_LED_NUMBER} - {actual_last_led}
• Expected: {EXPECTED_LEDS}
• Match: {'✅ YES' if detected_leds == EXPECTED_LEDS else '⚠️ NO'}

Pulse Width Statistics:
• Mean: {mean_width:.2f} samples
• Std Dev: {std_width:.2f} samples
• Min: {np.min(pulse_widths)} samples
• Max: {np.max(pulse_widths)} samples
• Anomalies (>2σ): {width_anomalies} ({width_anomalies/detected_leds*100:.1f}%)

Inter-Pulse Distance:
• Mean: {mean_dist:.2f} samples
• Std Dev: {std_dist:.2f} samples
• Min: {np.min(inter_distances) if inter_distances else 0:.0f} samples
• Max: {np.max(inter_distances) if inter_distances else 0:.0f} samples
• Anomalies (>2σ): {dist_anomalies} ({dist_anomalies/(detected_leds-1)*100:.1f}%)

Height Statistics:
• Mean Absolute: {np.mean(absolute_heights):.4f}
• Std Dev: {np.std(absolute_heights):.4f}
• Min: {np.min(absolute_heights):.4f}
• Max: {np.max(absolute_heights):.4f}

Threshold: {analyzer.threshold:.6f}
"""
        
        ax9.text(0.05, 0.95, stats_text.strip(), 
                transform=ax9.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        summary_file = analyzer.output_dir / "SUMMARY_ANALYSIS.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Summary plot saved: {summary_file}")
        plt.show()
        
        # Print detailed statistics to console
        print("\n" + "=" * 80)
        print("📊 DETAILED STATISTICS")
        print("=" * 80)
        print(stats_text)
        print("=" * 80)
        
        # Create final plot: LED Number vs Normalized Pulse Height (Interactive)
        print("\n📊 Creating LED vs Normalized Height Plot (Interactive - You can zoom and pan!)...")
        
        fig_final = plt.figure(figsize=(16, 8))
        ax_final = fig_final.add_subplot(111)
        
        # Plot normalized heights
        ax_final.plot(led_numbers, normalized_heights, 'g-', linewidth=1, alpha=0.7, label='Normalized Height')
        ax_final.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Average (1.0)', alpha=0.7)
        
        # Add ±2σ bands
        mean_norm = np.mean(normalized_heights)
        std_norm = np.std(normalized_heights)
        ax_final.axhline(y=mean_norm + 2*std_norm, color='orange', linestyle=':', linewidth=2, 
                        label=f'+2σ ({mean_norm + 2*std_norm:.3f})', alpha=0.7)
        ax_final.axhline(y=mean_norm - 2*std_norm, color='orange', linestyle=':', linewidth=2, 
                        label=f'-2σ ({mean_norm - 2*std_norm:.3f})', alpha=0.7)
        
        ax_final.set_title('LED Number vs Normalized Pulse Height (Interactive - Use toolbar to zoom/pan)', 
                          fontweight='bold', fontsize=14)
        ax_final.set_xlabel('LED Number', fontsize=12)
        ax_final.set_ylabel('Normalized Height', fontsize=12)
        ax_final.legend(fontsize=10)
        ax_final.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_box = f"""Statistics:
Mean: {mean_norm:.4f}
Std Dev: {std_norm:.4f}
Min: {np.min(normalized_heights):.4f}
Max: {np.max(normalized_heights):.4f}
Range: {np.max(normalized_heights) - np.min(normalized_heights):.4f}"""
        
        ax_final.text(0.02, 0.98, stats_box, transform=ax_final.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        final_plot_file = analyzer.output_dir / "LED_vs_NORMALIZED_HEIGHT.png"
        plt.savefig(final_plot_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ LED vs Normalized Height plot saved: {final_plot_file}")
        
        # Show interactive plot
        print("\n" + "=" * 80)
        print("🎯 INTERACTIVE PLOT MODE")
        print("=" * 80)
        print("Use the matplotlib toolbar to:")
        print("  🔍 Zoom: Click the magnifying glass icon, then drag on the plot")
        print("  ↔️  Pan: Click the cross arrows icon, then drag to move around")
        print("  🏠 Home: Click home icon to reset view")
        print("  💾 Save: Click save icon to save current view")
        print("=" * 80)
        plt.show()
        
    else:
        print("\n" + "=" * 80)
        print("❌ ANALYSIS FAILED")
        print("=" * 80)
