import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy import stats

# Module configurations from crosstalk_comparison_analysis.py
CROSSTALK_CONFIG = [
    {
        "name": "Rear",
        "module_id": 0,
        "first_led": 2304,
        "last_led": 15615,
        "expected_leds": 13312,
    },
    {
        "name": "Middle",
        "module_id": 1,
        "first_led": 256,
        "last_led": 15615,
        "expected_leds": 15360,
    },
    {
        "name": "Front",
        "module_id": 2,
        "first_led": 256,
        "last_led": 21247,
        "expected_leds": 20992,
    },
]


class LEDAnalyzer:
    def __init__(
        self, bin_file_path, first_led_number=1, last_led_number=None, show_plots=True
    ):
        self.bin_file_path = Path(bin_file_path)
        self.first_led_number = first_led_number
        self.last_led_number = last_led_number
        self.expected_led_count = (
            last_led_number - first_led_number + 1 if last_led_number else None
        )
        self.show_plots = show_plots
        self.output_dir = (
            self.bin_file_path.parent / f"{self.bin_file_path.stem}_analysis"
        )
        self.output_dir.mkdir(exist_ok=True)
        self.detected_pulses = []
        self.voltage_data = None
        self.original_voltage_data = None  # שמירה של הדאטה המקורי
        self.threshold = None
        self.pulse_width = None
        self.pulse_interval = None
        self.peak_amplitudes = []  # רשימת אמפליטודות הפולסים

    def run_analysis(self):
        print(f"Reading: {self.bin_file_path}")

        try:
            with open(self.bin_file_path, "rb") as f:
                raw_bytes = f.read()

            file_size = len(raw_bytes)
            print(f"File size: {file_size:,} bytes")

            # Skip 128 bytes header and read voltage data
            skip_bytes = 128
            data_bytes = raw_bytes[skip_bytes:]
            voltage_data = np.frombuffer(data_bytes, dtype=np.float32)

            print(f"\n📊 Voltage Data Analysis:")
            print(f"Values count: {len(voltage_data):,}")
            print(f"Range: {voltage_data.min():.6f}V to {voltage_data.max():.6f}V")
            print(f"Mean: {voltage_data.mean():.6f}V")
            print(f"Std: {voltage_data.std():.6f}V")
            print(f"First 20 values: {voltage_data[:20]}")

            # Look for pulse patterns
            print(f"\n🔍 Looking for LED pulse patterns...")

            # Find values above certain thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            for thresh in thresholds:
                above_thresh = np.sum(voltage_data > thresh)
                percentage = (above_thresh / len(voltage_data)) * 100
                print(f"Values > {thresh}V: {above_thresh:,} ({percentage:.2f}%)")

            # Store the voltage data for further analysis
            self.voltage_data = voltage_data
            self.original_voltage_data = (
                voltage_data.copy()
            )  # שמירת עותק של הדאטה המקורי

            # Analysis with timing
            total_start_time = time.time()
            print(f"\n🚀 Starting LED Analysis Steps...")

            # שלב 1: ניתוח היסטוגרמה וקביעת סף
            step_start = time.time()
            self._step1_histogram_threshold_analysis()
            step1_time = time.time() - step_start
            print(f"⏱️ Step 1 completed in {step1_time:.2f} seconds")

            # שלב 2: זיהוי פולסים וחישוב מאפיינים
            step_start = time.time()
            self._step2_pulse_detection_analysis()
            step2_time = time.time() - step_start
            print(f"⏱️ Step 2 completed in {step2_time:.2f} seconds")

            # שלב 3: הסרת איזורים מתים
            step_start = time.time()
            self._step3_dead_zone_removal()
            step3_time = time.time() - step_start
            print(f"⏱️ Step 3 completed in {step3_time:.2f} seconds")

            # שלב 4: זיהוי מדויק של כל הפולסים
            step_start = time.time()
            self._step4_precise_pulse_detection()
            step4_time = time.time() - step_start
            print(f"⏱️ Step 4 completed in {step4_time:.2f} seconds")

            # שלב 5: חישוב Peak Valley ואמפליטודות
            step_start = time.time()
            self._step5_peak_valley_calculation()
            step5_time = time.time() - step_start
            print(f"⏱️ Step 5 completed in {step5_time:.2f} seconds")

            total_time = time.time() - total_start_time
            print(f"\n🎯 Total analysis completed in {total_time:.2f} seconds")
            print(f"📈 Time breakdown:")
            print(
                f"  Step 1 (Histogram): {step1_time:.2f}s ({100*step1_time/total_time:.1f}%)"
            )
            print(
                f"  Step 2 (Pulse Detection): {step2_time:.2f}s ({100*step2_time/total_time:.1f}%)"
            )
            print(
                f"  Step 3 (Dead Zone): {step3_time:.2f}s ({100*step3_time/total_time:.1f}%)"
            )
            print(
                f"  Step 4 (Precise Pulses): {step4_time:.2f}s ({100*step4_time/total_time:.1f}%)"
            )
            print(
                f"  Step 5 (Peak Valley): {step5_time:.2f}s ({100*step5_time/total_time:.1f}%)"
            )

            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    def _plot_sample_data(self):
        """Plot downsampled data from the center for faster visualization"""
        if self.voltage_data is None or len(self.voltage_data) == 0:
            print("No voltage data to plot")
            return

        # Take 20K values from the center and downsample by factor of 4
        total_samples = len(self.voltage_data)
        sample_size = min(20000, total_samples)
        center_start = (total_samples - sample_size) // 2
        center_end = center_start + sample_size

        sample_data = self.voltage_data[center_start:center_end]

        # Downsample: take every 4th point for faster plotting
        downsample_factor = 4
        downsampled_data = sample_data[::downsample_factor]
        downsampled_indices = np.arange(center_start, center_end, downsample_factor)

        print(
            f"\n📊 Plotting {len(downsampled_data):,} downsampled points (every {downsample_factor}th sample) from center"
        )

        # Create the plot
        plt.figure(figsize=(15, 8))
        plt.plot(downsampled_indices, downsampled_data, linewidth=0.5)
        plt.title(
            f"LED Voltage Data - {len(downsampled_data):,} downsampled points (1:{downsample_factor})\n{self.bin_file_path.name}"
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Voltage (V)")
        plt.grid(True, alpha=0.3)

        # Add threshold line if available
        if self.threshold is not None:
            plt.axhline(
                float(self.threshold),
                color="green",
                linestyle="-",
                linewidth=2,
                label=f"Threshold: {float(self.threshold):.6f}V",
            )
            plt.legend()

        # Add some statistics to the plot
        plt.text(
            0.02,
            0.98,
            f"Range: {sample_data.min():.3f}V to {sample_data.max():.3f}V\n"
            f"Mean: {sample_data.mean():.3f}V\n"
            f"Std: {sample_data.std():.3f}V",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Save plot
        # plot_path = self.output_dir / "voltage_sample_plot.png"
        # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        # print(f"📈 Plot saved: {plot_path}")

        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def _step1_histogram_threshold_analysis(self):
        """Step 1: Histogram Analysis and Threshold Determination"""
        if self.voltage_data is None:
            print("No voltage data available for analysis")
            return

        print(f"\n🔍 Step 1: Histogram Analysis and Threshold Determination")

        # Work on all data instead of just center
        total_samples = len(self.voltage_data)
        print(
            f"📊 Processing all {total_samples:,} data points instead of just 20K from center"
        )

        center_data = self.voltage_data  # All data
        print(
            f"Processing all {len(center_data):,} data points instead of just 20K from center"
        )

        # יצירת היסטוגרמה
        hist, bin_edges = np.histogram(center_data, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # חיפוש שני הפיקים הגבוהים ביותר (שיא ושפל)
        # מיון הפיקים לפי גובה
        peak_indices = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peak_indices.append(i)

        # לקיחת שני הפיקים הגבוהים ביותר
        peak_heights = [(i, hist[i]) for i in peak_indices]
        peak_heights.sort(key=lambda x: x[1], reverse=True)

        if len(peak_heights) >= 2:
            # השיא (הערך הגבוה יותר) והשפל (הערך הנמוך יותר)
            peak1_idx, peak1_height = peak_heights[0]
            peak2_idx, peak2_height = peak_heights[1]

            peak1_voltage = bin_centers[peak1_idx]
            peak2_voltage = bin_centers[peak2_idx]

            # קביעה איזה הוא השיא ואיזה השפל
            if peak1_voltage > peak2_voltage:
                high_peak = peak1_voltage
                low_peak = peak2_voltage
            else:
                high_peak = peak2_voltage
                low_peak = peak1_voltage

            # חישוב הסף - בדיוק באמצע בין השיא לשפל
            self.threshold = (high_peak + low_peak) / 2

            print(f"Identified two clusters:")
            print(f"  Peak (high values): {high_peak:.6f}V")
            print(f"  Valley (low values): {low_peak:.6f}V")
            print(f"  Threshold set to: {self.threshold:.6f}V")

            # הצגת היסטוגרמה
            if self.show_plots:
                plt.figure(figsize=(12, 6))
                plt.bar(
                    bin_centers,
                    hist,
                    width=(bin_edges[1] - bin_edges[0]),
                    alpha=0.7,
                    color="lightblue",
                )
                plt.axvline(
                    float(high_peak),
                    color="red",
                    linestyle="--",
                    label=f"Peak: {float(high_peak):.6f}V",
                )
                plt.axvline(
                    float(low_peak),
                    color="blue",
                    linestyle="--",
                    label=f"Valley: {float(low_peak):.6f}V",
                )
                plt.axvline(
                    float(self.threshold),
                    color="green",
                    linestyle="-",
                    linewidth=2,
                    label=f"Threshold: {float(self.threshold):.6f}V",
                )
                plt.title("Histogram of 20K Center Points - Peak and Valley Detection")
                plt.xlabel("Voltage (V)")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
        else:
            print("⚠️ Could not find two distinct peaks in histogram")
            # Default threshold based on mean + standard deviation
            self.threshold = center_data.mean() + center_data.std()
            print(f"Default threshold set to: {self.threshold:.6f}V")

    def _step2_pulse_detection_analysis(self):
        """Step 2: Pulse Detection and Characteristic Calculation"""
        if self.voltage_data is None or self.threshold is None:
            print("⚠️ No voltage data or threshold available for analysis")
            return

        print(f"\n🔍 Step 2: Pulse Detection and Characteristic Calculation")

        # לקיחת אותן 20K נקודות מהמרכז לצורך בדיקת העקביות
        total_samples = len(self.voltage_data)
        sample_size = min(20000, total_samples)
        center_start = (total_samples - sample_size) // 2
        center_end = center_start + sample_size

        center_data = self.voltage_data[center_start:center_end]

        # יצירת מסכת בוליאנית - True כאשר הערך מעל הסף
        above_threshold = center_data > self.threshold

        # חיפוש חיתוכי סף (עליות וירידות)
        crossings = []
        crossing_types = []  # 'up' או 'down'

        for i in range(1, len(above_threshold)):
            # עליה - מתחת הסף לעל הסף
            if not above_threshold[i - 1] and above_threshold[i]:
                crossings.append(i)
                crossing_types.append("up")
            # ירידה - מעל הסף למתחת הסף
            elif above_threshold[i - 1] and not above_threshold[i]:
                crossings.append(i)
                crossing_types.append("down")

        print(f"Found {len(crossings)} threshold crossings:")
        print(f"  Rising edges: {crossing_types.count('up')}")
        print(f"  Falling edges: {crossing_types.count('down')}")

        # בדיקת עקביות - אחרי כל עליה צריכה להיות ירידה
        consistency_check = True
        pulse_starts = []
        pulse_ends = []

        for i in range(len(crossing_types) - 1):
            if crossing_types[i] == "up":
                # חיפוש הירידה הבאה
                next_down = None
                for j in range(i + 1, len(crossing_types)):
                    if crossing_types[j] == "down":
                        next_down = j
                        break

                if next_down is not None:
                    pulse_starts.append(crossings[i])
                    pulse_ends.append(crossings[next_down])
                else:
                    consistency_check = False
                    break

        if consistency_check and len(pulse_starts) > 0:
            print(f"✅ Consistency found! Detected {len(pulse_starts)} complete pulses")

            # חישוב אורכי פולסים
            pulse_widths = [
                pulse_ends[i] - pulse_starts[i] for i in range(len(pulse_starts))
            ]
            self.pulse_width = np.mean(pulse_widths)

            # חישוב מרחקים בין פולסים
            if len(pulse_starts) > 1:
                pulse_intervals = [
                    pulse_starts[i + 1] - pulse_ends[i]
                    for i in range(len(pulse_starts) - 1)
                ]
                self.pulse_interval = np.mean(pulse_intervals)

            print(f"📏 Pulse Characteristics:")
            print(f"  Average pulse width: {self.pulse_width:.1f} samples")
            if self.pulse_interval is not None:
                print(
                    f"  Average inter-pulse distance: {self.pulse_interval:.1f} samples"
                )
                total_cycle = self.pulse_width + self.pulse_interval
                print(f"  Cycle length (pulse + gap): {total_cycle:.1f} samples")

            # הצגת הפולסים שנמצאו - רק חלק מהנתונים להצגה ברורה
            if self.show_plots and len(pulse_starts) > 0:
                # חישוב כמה פולסים להציג (~100 פולסים)
                pulses_to_show = min(100, len(pulse_starts))
                if pulses_to_show > 0:
                    # חישוב הטווח להצגה
                    first_pulse_start = pulse_starts[0]
                    last_pulse_end = pulse_ends[pulses_to_show - 1]

                    # הוספת מעט רווח לפני ואחרי
                    display_margin = (
                        int(self.pulse_interval * 2) if self.pulse_interval else 100
                    )
                    display_start = max(0, first_pulse_start - display_margin)
                    display_end = min(len(center_data), last_pulse_end + display_margin)

                    display_data = center_data[display_start:display_end]
                    display_indices = np.arange(display_start, display_end)

                    plt.figure(figsize=(15, 8))
                    plt.plot(
                        display_indices,
                        display_data,
                        linewidth=0.8,
                        alpha=0.8,
                        color="blue",
                    )
                    plt.axhline(
                        float(self.threshold),
                        color="green",
                        linestyle="-",
                        linewidth=2,
                        label=f"Threshold: {float(self.threshold):.6f}V",
                    )

                    # סימון הפולסים
                    for i in range(pulses_to_show):
                        start_idx = pulse_starts[i]
                        end_idx = pulse_ends[i]
                        plt.axvspan(
                            start_idx,
                            end_idx,
                            alpha=0.3,
                            color="red",
                            label="Pulse" if i == 0 else "",
                        )

                        # סימון נקודות חיתוך
                        plt.axvline(
                            start_idx,
                            color="orange",
                            linestyle=":",
                            alpha=0.7,
                            label="Threshold Crossing" if i == 0 else "",
                        )
                        plt.axvline(end_idx, color="orange", linestyle=":", alpha=0.7)

                    plt.title(
                        f"Pulse Detection Analysis - {pulses_to_show} Pulses Shown (of {len(pulse_starts)} total)"
                    )
                    plt.xlabel("Sample Index")
                    plt.ylabel("Voltage (V)")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.show()

        else:
            print("⚠️ No pulse consistency found or no complete pulses detected")

    def _step3_dead_zone_removal(self):
        """Step 3: Dead Zone Removal"""
        if (
            self.voltage_data is None
            or self.threshold is None
            or self.pulse_width is None
        ):
            print(
                "⚠️ No voltage data, threshold, or pulse characteristics available for dead zone removal"
            )
            return

        print(f"\n🔍 Step 3: Dead Zone Removal")

        step3_start = time.time()

        original_length = len(self.voltage_data)
        center_index = original_length // 2

        # Calculate STD window size (5 pulse cycles for better boundary detection)
        pulse_cycle = (
            int(self.pulse_width + self.pulse_interval) if self.pulse_interval else 50
        )
        std_window_size = pulse_cycle * 5  # 5 pulse cycles for better resolution

        print(f"⏱️ [Step 3] Initialization: {time.time() - step3_start:.3f}s")

        print(f"  Pulse cycle length: {pulse_cycle} samples")
        print(f"  STD window size: {std_window_size} samples (5 cycles)")

        # STD ANALYSIS METHOD - Fast dead zone detection
        print("\n  📊 STD Analysis Method")

        # Calculate STD at sampled positions across the data
        # Sample every 5 pulse cycles for better resolution
        sample_step = pulse_cycle * 5
        sample_positions = np.arange(
            std_window_size // 2, original_length - std_window_size // 2, sample_step
        )

        std_start = time.time()
        std_values = []
        for pos in sample_positions:
            window = self.voltage_data[
                pos - std_window_size // 2 : pos + std_window_size // 2
            ]
            std_values.append(np.std(window))
        std_values = np.array(std_values)
        print(
            f"⏱️ [Step 3] STD calculation at {len(sample_positions)} positions: {time.time() - std_start:.3f}s"
        )

        # Find nominal STD at center
        center_std_idx = len(std_values) // 2
        nominal_std = std_values[center_std_idx]
        std_threshold_value = nominal_std * 0.25  # 25% threshold (more aggressive)

        print(f"    Nominal STD (center): {nominal_std:.6f}")
        print(f"    Dead zone threshold (25%): {std_threshold_value:.6f}")

        # Scan left from center
        scan_start = time.time()
        left_boundary_idx = 0
        for i in range(center_std_idx, -1, -1):
            if std_values[i] < std_threshold_value:
                left_boundary_idx = i + 1  # Take the last good position
                break

        # Scan right from center
        right_boundary_idx = len(std_values) - 1
        for i in range(center_std_idx, len(std_values)):
            if std_values[i] < std_threshold_value:
                right_boundary_idx = i - 1  # Take the last good position
                break

        print(f"⏱️ [Step 3] Bidirectional scan: {time.time() - scan_start:.3f}s")

        # Convert sample indices back to data indices with safety margin
        # Add small margin (5 pulse cycles) on each side to avoid cutting relevant data
        safety_margin = pulse_cycle * 5
        active_start = max(
            0,
            int(
                sample_positions[left_boundary_idx]
                - std_window_size // 2
                - safety_margin
            ),
        )
        active_end = min(
            original_length,
            int(
                sample_positions[right_boundary_idx]
                + std_window_size // 2
                + safety_margin
            ),
        )

        method_used = "STD Analysis (25% threshold, +5 cycle margin)"

        print(
            f"    Left boundary: sample {left_boundary_idx}/{len(std_values)}, data index {active_start:,}"
        )
        print(
            f"    Right boundary: sample {right_boundary_idx}/{len(std_values)}, data index {active_end:,}"
        )
        print(
            f"    Safety margin applied: {safety_margin:,} samples ({safety_margin // pulse_cycle} cycles) on each side"
        )

        # הגנה מפני חיתוך יתר
        min_data_length = original_length // 4  # לפחות 25% מהדאטה
        if (active_end - active_start) < min_data_length:
            # אם חותכים יותר מדי, השאר יותר דאטה
            margin = (min_data_length - (active_end - active_start)) // 2
            active_start = max(0, active_start - margin)
            active_end = min(original_length, active_end + margin)
            method_used += " (with minimum data protection)"

        # עדכון נתוני המתח
        if active_start > 0 or active_end < original_length:
            self.voltage_data = self.voltage_data[active_start:active_end]

        active_length = active_end - active_start
        removed_left = active_start
        removed_right = original_length - active_end
        total_removed = removed_left + removed_right

        percentage_kept = (active_length / original_length) * 100
        percentage_removed = (total_removed / original_length) * 100

        print(f"\n📊 Dead Zone Removal Results ({method_used}):")
        print(f"  Original data length: {original_length:,} samples")
        print(f"  Active data length: {active_length:,} samples")
        print(f"  Left dead zone removed: {removed_left:,} samples")
        print(f"  Right dead zone removed: {removed_right:,} samples")
        print(f"  Total removed: {total_removed:,} samples ({percentage_removed:.2f}%)")
        print(f"  Data retained: {percentage_kept:.2f}%")

        computation_time = time.time() - step3_start
        print(f"⏱️ [Step 3] Total computation time: {computation_time:.3f}s")

        # הצגה גרפית
        if self.show_plots:
            viz_start = time.time()
            print(f"📊 [Step 3] Starting visualization...")
            plt.figure(figsize=(16, 12))

            # גרף עליון - דאטה מקורי עם STD בציר ימני (שורה 1, כל הרוחב)
            ax1 = plt.subplot(3, 2, (1, 2))  # שורה 1, תופס 2 עמודות
            # השתמש בדאטה הזמין
            data_for_plot = (
                self.original_voltage_data
                if self.original_voltage_data is not None
                else self.voltage_data
            )
            if data_for_plot is not None:
                # Downsample to 10K points for faster plotting
                downsample_start = time.time()
                target_points = 10000
                downsample_factor = max(1, len(data_for_plot) // target_points)
                sample_indices = np.arange(0, len(data_for_plot), downsample_factor)
                sample_data = data_for_plot[sample_indices]
                print(
                    f"⏱️ [Step 3] Downsample top plot ({len(data_for_plot):,} -> {len(sample_data):,}): {time.time() - downsample_start:.3f}s"
                )

                # Plot voltage data on left Y-axis
                plot_start = time.time()
                ax1.plot(
                    sample_indices,
                    sample_data,
                    linewidth=0.3,
                    alpha=0.7,
                    color="blue",
                    label="Voltage",
                )
                print(f"⏱️ [Step 3] Plot top data: {time.time() - plot_start:.3f}s")

                # Create right Y-axis for STD
                ax2 = ax1.twinx()
                ax2.plot(
                    sample_positions,
                    std_values,
                    "orange",
                    linewidth=2,
                    alpha=0.8,
                    label="STD",
                )
                ax2.axhline(
                    std_threshold_value,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"STD Threshold (50%)",
                )
                ax2.axhline(
                    nominal_std,
                    color="green",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Nominal STD",
                )
                ax2.set_ylabel("STD (Standard Deviation)", color="orange")
                ax2.tick_params(axis="y", labelcolor="orange")

                # Display STD analysis info
                info_text = f"STD Method: Nominal={nominal_std:.6f}, Threshold={std_threshold_value:.6f} (25% threshold)\nSafety Margin: {safety_margin:,} samples ({safety_margin // pulse_cycle} cycles)"
                ax1.text(
                    0.02,
                    0.98,
                    info_text,
                    transform=ax1.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    fontsize=9,
                )

                # סימון איזורים מתים
                if removed_left > 0:
                    ax1.axvspan(
                        0,
                        active_start,
                        alpha=0.3,
                        color="red",
                        label="Dead Zone (Left)",
                    )
                if removed_right > 0:
                    ax1.axvspan(
                        active_end,
                        original_length,
                        alpha=0.3,
                        color="red",
                        label="Dead Zone (Right)" if removed_left == 0 else "",
                    )

                ax1.set_title(
                    f"Original Data with Dead Zones Marked (Total: {original_length:,} samples)"
                )
                ax1.set_xlabel("Sample Index")
                ax1.set_ylabel("Voltage (V)", color="blue")
                ax1.tick_params(axis="y", labelcolor="blue")

                # Combine legends from both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                ax1.grid(True, alpha=0.3)

            # Middle plot - Data after removal (שורה 2, כל הרוחב)
            plt.subplot(3, 2, (3, 4))  # שורה 2, תופס 2 עמודות
            # Downsample to 10K points for faster plotting
            downsample_start2 = time.time()
            target_points = 10000
            downsample_factor = max(1, len(self.voltage_data) // target_points)
            active_sample_indices = np.arange(
                0, len(self.voltage_data), downsample_factor
            )
            active_sample_data = self.voltage_data[active_sample_indices]
            print(
                f"⏱️ [Step 3] Downsample bottom plot ({len(self.voltage_data):,} -> {len(active_sample_data):,}): {time.time() - downsample_start2:.3f}s"
            )

            plot_start2 = time.time()
            plt.plot(
                active_sample_indices,
                active_sample_data,
                linewidth=0.3,
                alpha=0.7,
                color="darkblue",
            )
            print(f"⏱️ [Step 3] Plot bottom data: {time.time() - plot_start2:.3f}s")

            plt.title(
                f"Active Data After Dead Zone Removal ({len(self.voltage_data):,} samples, {percentage_kept:.1f}% retained)"
            )
            plt.xlabel("Sample Index")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Bottom left - Zoom on left cut area (100 pulses) - שורה 3, עמודה שמאלית
            zoom_start_time = time.time()
            ax_left = plt.subplot(3, 2, 5)
            zoom_pulses = 100
            zoom_samples = pulse_cycle * zoom_pulses

            # Calculate left zoom range - show 50 pulses before cut and 50 after
            left_zoom_start = max(0, active_start - zoom_samples // 2)
            left_zoom_end = min(original_length, active_start + zoom_samples // 2)

            if data_for_plot is not None and left_zoom_end > left_zoom_start:
                zoom_data_left = data_for_plot[left_zoom_start:left_zoom_end]
                zoom_indices_left = np.arange(left_zoom_start, left_zoom_end)

                ax_left.plot(
                    zoom_indices_left,
                    zoom_data_left,
                    linewidth=0.5,
                    color="blue",
                    alpha=0.7,
                    label="Voltage",
                )
                ax_left.axvline(
                    active_start,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Cut Point",
                )
                ax_left.axvspan(
                    left_zoom_start,
                    active_start,
                    alpha=0.2,
                    color="red",
                    label="Removed",
                )
                ax_left.axvspan(
                    active_start, left_zoom_end, alpha=0.2, color="green", label="Kept"
                )

                # Add STD overlay on right Y-axis
                ax_left_std = ax_left.twinx()
                # Filter STD points that fall in the zoom range
                std_mask = (sample_positions >= left_zoom_start) & (
                    sample_positions <= left_zoom_end
                )
                num_std_points = np.sum(std_mask)
                print(
                    f"🔍 Left zoom: {num_std_points} STD points in range [{left_zoom_start}, {left_zoom_end}]"
                )
                if np.any(std_mask):
                    zoom_std_positions = sample_positions[std_mask]
                    zoom_std_values = std_values[std_mask]
                    print(f"    STD positions: {zoom_std_positions}")
                    print(f"    STD values: {zoom_std_values}")
                    ax_left_std.plot(
                        zoom_std_positions,
                        zoom_std_values,
                        "o-",
                        color="darkorange",
                        linewidth=3,
                        markersize=8,
                        alpha=1.0,
                        label="STD",
                        zorder=10,
                    )
                    ax_left_std.axhline(
                        std_threshold_value,
                        color="red",
                        linestyle=":",
                        linewidth=3,
                        alpha=0.9,
                        label="STD Threshold",
                        zorder=9,
                    )

                    # Set Y-axis range for STD to show the actual values in this zoom area
                    std_min = np.min(zoom_std_values)
                    std_max = np.max(zoom_std_values)
                    std_range = std_max - std_min
                    margin = std_range * 0.1 if std_range > 0 else 0.05
                    ax_left_std.set_ylim(max(0, std_min - margin), std_max + margin)

                    ax_left_std.set_ylabel("STD", color="orange", fontsize=9)
                    ax_left_std.tick_params(axis="y", labelcolor="orange", labelsize=8)

                    # Add legend for STD axis
                    lines_std, labels_std = ax_left_std.get_legend_handles_labels()
                    if lines_std:
                        ax_left_std.legend(
                            lines_std, labels_std, fontsize=7, loc="upper right"
                        )

                ax_left.set_title(
                    f"Left Cut Zone (±{zoom_pulses//2} pulses)", fontsize=10
                )
                ax_left.set_xlabel("Sample Index", fontsize=9)
                ax_left.set_ylabel("Voltage (V)", color="blue", fontsize=9)
                ax_left.tick_params(axis="y", labelcolor="blue", labelsize=8)
                ax_left.legend(fontsize=7, loc="upper left")
                ax_left.grid(True, alpha=0.3)
                ax_left.tick_params(labelsize=8)

            # Bottom right - Zoom on right cut area (100 pulses) - שורה 3, עמודה ימנית
            ax_right = plt.subplot(3, 2, 6)

            # Calculate right zoom range - show 50 pulses before cut and 50 after
            right_zoom_start = max(0, active_end - zoom_samples // 2)
            right_zoom_end = min(original_length, active_end + zoom_samples // 2)

            if data_for_plot is not None and right_zoom_end > right_zoom_start:
                zoom_data_right = data_for_plot[right_zoom_start:right_zoom_end]
                zoom_indices_right = np.arange(right_zoom_start, right_zoom_end)

                ax_right.plot(
                    zoom_indices_right,
                    zoom_data_right,
                    linewidth=0.5,
                    color="blue",
                    alpha=0.7,
                    label="Voltage",
                )
                ax_right.axvline(
                    active_end,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Cut Point",
                )
                ax_right.axvspan(
                    right_zoom_start, active_end, alpha=0.2, color="green", label="Kept"
                )
                ax_right.axvspan(
                    active_end, right_zoom_end, alpha=0.2, color="red", label="Removed"
                )

                # Add STD overlay on right Y-axis
                ax_right_std = ax_right.twinx()
                # Filter STD points that fall in the zoom range
                std_mask_right = (sample_positions >= right_zoom_start) & (
                    sample_positions <= right_zoom_end
                )
                if np.any(std_mask_right):
                    zoom_std_positions_right = sample_positions[std_mask_right]
                    zoom_std_values_right = std_values[std_mask_right]
                    ax_right_std.plot(
                        zoom_std_positions_right,
                        zoom_std_values_right,
                        "o-",
                        color="orange",
                        linewidth=2,
                        markersize=6,
                        alpha=0.8,
                        label="STD",
                    )
                    ax_right_std.axhline(
                        std_threshold_value,
                        color="red",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.8,
                        label="STD Threshold",
                    )

                    # Set Y-axis range for STD to show the actual values in this zoom area
                    std_min_right = np.min(zoom_std_values_right)
                    std_max_right = np.max(zoom_std_values_right)
                    std_range_right = std_max_right - std_min_right
                    margin_right = (
                        std_range_right * 0.1 if std_range_right > 0 else 0.05
                    )
                    ax_right_std.set_ylim(
                        max(0, std_min_right - margin_right),
                        std_max_right + margin_right,
                    )

                    ax_right_std.set_ylabel("STD", color="orange", fontsize=9)
                    ax_right_std.tick_params(axis="y", labelcolor="orange", labelsize=8)

                    # Add legend for STD axis
                    lines_std_right, labels_std_right = (
                        ax_right_std.get_legend_handles_labels()
                    )
                    if lines_std_right:
                        ax_right_std.legend(
                            lines_std_right,
                            labels_std_right,
                            fontsize=7,
                            loc="upper right",
                        )

                ax_right.set_title(
                    f"Right Cut Zone (±{zoom_pulses//2} pulses)", fontsize=10
                )
                ax_right.set_xlabel("Sample Index", fontsize=9)
                ax_right.set_ylabel("Voltage (V)", color="blue", fontsize=9)
                ax_right.tick_params(axis="y", labelcolor="blue", labelsize=8)
                ax_right.legend(fontsize=7, loc="upper left")
                ax_right.grid(True, alpha=0.3)
                ax_right.tick_params(labelsize=8)

            print(
                f"⏱️ [Step 3] Zoom plots created: {time.time() - zoom_start_time:.3f}s"
            )

            layout_start = time.time()
            plt.tight_layout()
            print(f"⏱️ [Step 3] Layout adjustment: {time.time() - layout_start:.3f}s")

            total_viz_time = time.time() - viz_start
            print(f"⏱️ [Step 3] Total visualization prep: {total_viz_time:.3f}s")
            print(f"📊 [Step 3] Showing plot (this will wait for user to close)...")

            show_start = time.time()
            plt.show()
            print(
                f"⏱️ [Step 3] plt.show() returned after: {time.time() - show_start:.3f}s"
            )

    def _step4_precise_pulse_detection(self):
        """Step 4: Precise Pulse Detection with Peak/Valley Calculation"""
        if self.voltage_data is None or self.threshold is None:
            print(
                "⚠️ No voltage data or threshold available for precise pulse detection"
            )
            return

        print(f"\n{'='*80}")
        print(f"🎯 STEP 4: PRECISE PULSE DETECTION")
        print(f"{'='*80}")

        voltage = self.voltage_data
        threshold = self.threshold
        n_samples = len(voltage)

        # 1. זיהוי חציות סף (Threshold Crossings)
        print(f"\n📊 Phase 1: Detecting Threshold Crossings...")

        above_threshold = voltage > threshold
        crossings = np.diff(above_threshold.astype(int))

        rising_edges = np.where(crossings == 1)[0] + 1  # עליות (crossing up)
        falling_edges = np.where(crossings == -1)[0] + 1  # ירידות (crossing down)

        print(f"  Rising edges detected: {len(rising_edges):,}")
        print(f"  Falling edges detected: {len(falling_edges):,}")

        # 2. וידוא עקביות - חייב להתחיל בעליה ולסיים בירידה
        print(f"\n🔍 Phase 2: Validating Edge Consistency...")

        # מוודא שיש עליה ראשונה לפני ירידה ראשונה
        if len(rising_edges) > 0 and len(falling_edges) > 0:
            if rising_edges[0] > falling_edges[0]:
                print(f"  ⚠️ First falling edge before first rising edge - removing it")
                falling_edges = falling_edges[1:]

        # מוודא שיש ירידה אחרונה אחרי עליה אחרונה
        if len(rising_edges) > 0 and len(falling_edges) > 0:
            if rising_edges[-1] > falling_edges[-1]:
                print(f"  ⚠️ Last rising edge after last falling edge - removing it")
                rising_edges = rising_edges[:-1]

        # מספר הפולסים
        num_pulses = min(len(rising_edges), len(falling_edges))
        print(f"  ✅ Valid pulses after consistency check: {num_pulses:,}")

        if num_pulses == 0:
            print("⚠️ No valid pulses found!")
            return

        # השוואה למספר מצופה
        if self.expected_led_count:
            print(f"\n📈 Expected vs Detected:")
            print(f"  Expected LEDs: {self.expected_led_count:,}")
            print(f"  Detected pulses: {num_pulses:,}")
            diff = num_pulses - self.expected_led_count
            diff_percent = 100*diff/self.expected_led_count
            print(f"  Difference: {diff:+,} ({diff_percent:+.2f}%)")
            
            # הודעת אימות
            if diff == 0:
                print(f"  ✅ Perfect match: {num_pulses:,} pulses = {self.expected_led_count:,} expected LEDs")
            elif abs(diff_percent) < 0.1:
                print(f"  ✅ Excellent match (within 0.1%)")
            elif abs(diff_percent) < 1.0:
                print(f"  ⚠️ Good match (within 1%)")
            else:
                print(f"  ❌ Significant difference (>{abs(diff_percent):.1f}%)")

        # 3. חישוב Peak ו-Valley לכל פולס
        print(f"\n🔬 Phase 3: Calculating Peak and Valley for Each Pulse...")

        pulse_data = []

        for i in range(num_pulses):
            rise_idx = rising_edges[i]
            fall_idx = falling_edges[i]

            # Peak: חישוב בין עליה לירידה (50% מהאמצע)
            pulse_length = fall_idx - rise_idx
            peak_center = (rise_idx + fall_idx) // 2
            peak_half_width = int(pulse_length * 0.25)  # 50% / 2 = 25% מכל צד
            peak_start = max(rise_idx, peak_center - peak_half_width)
            peak_end = min(fall_idx, peak_center + peak_half_width)
            peak_region = voltage[peak_start:peak_end]
            peak_value = np.max(peak_region) if len(peak_region) > 0 else 0
            peak_idx = (
                peak_start + np.argmax(peak_region)
                if len(peak_region) > 0
                else peak_center
            )
            # חישוב STD של נקודות השיא
            peak_std = np.std(peak_region) if len(peak_region) > 0 else 0

            # Valley: חישוב בין ירידה לעליה הבאה (50% מהאמצע)
            if i < num_pulses - 1:
                next_rise_idx = rising_edges[i + 1]
                valley_length = next_rise_idx - fall_idx
                valley_center = (fall_idx + next_rise_idx) // 2
                valley_half_width = int(valley_length * 0.25)
                valley_start = max(fall_idx, valley_center - valley_half_width)
                valley_end = min(next_rise_idx, valley_center + valley_half_width)
            else:
                # פולס אחרון - לוקח valley עד סוף הדאטה
                valley_start = fall_idx
                valley_end = min(n_samples, fall_idx + pulse_length)

            valley_region = voltage[valley_start:valley_end]
            valley_value = np.min(valley_region) if len(valley_region) > 0 else 0
            valley_idx = (
                valley_start + np.argmin(valley_region)
                if len(valley_region) > 0
                else valley_start
            )

            amplitude = peak_value - valley_value

            pulse_data.append(
                {
                    "pulse_num": i + self.first_led_number,
                    "rise_idx": rise_idx,
                    "fall_idx": fall_idx,
                    "pulse_width": pulse_length,
                    "peak_value": peak_value,
                    "peak_idx": peak_idx,
                    "peak_start": peak_start,
                    "peak_end": peak_end,
                    "peak_std": peak_std,
                    "valley_value": valley_value,
                    "valley_idx": valley_idx,
                    "valley_start": valley_start,
                    "valley_end": valley_end,
                    "amplitude": amplitude,
                    "interval": (
                        rising_edges[i + 1] - rising_edges[i]
                        if i < num_pulses - 1
                        else 0
                    ),
                }
            )

        self.pulse_data = pulse_data

        print(f"  ✅ Calculated peak/valley for {len(pulse_data):,} pulses")

        # 4. סטטיסטיקה של אורך פולס ומרחק בין פולסים
        print(f"\n📊 Phase 4: Pulse Statistics...")

        pulse_widths = [p["pulse_width"] for p in pulse_data]
        intervals = [p["interval"] for p in pulse_data if p["interval"] > 0]
        amplitudes = [p["amplitude"] for p in pulse_data]

        print(f"\n  Pulse Width Statistics:")
        print(f"    Mean: {np.mean(pulse_widths):.1f} samples")
        print(f"    Std: {np.std(pulse_widths):.1f} samples")
        print(f"    Min: {np.min(pulse_widths)} samples")
        print(f"    Max: {np.max(pulse_widths)} samples")

        print(f"\n  Pulse Interval Statistics:")
        print(f"    Mean: {np.mean(intervals):.1f} samples")
        print(f"    Std: {np.std(intervals):.1f} samples")
        print(f"    Min: {np.min(intervals)} samples")
        print(f"    Max: {np.max(intervals)} samples")

        print(f"\n  Amplitude Statistics:")
        print(f"    Mean: {np.mean(amplitudes):.6f} V")
        print(f"    Std: {np.std(amplitudes):.6f} V")
        print(f"    Min: {np.min(amplitudes):.6f} V")
        print(f"    Max: {np.max(amplitudes):.6f} V")

        # 5. יצירת היסטוגרמות
        print(f"\n📈 Phase 5: Creating Histograms...")

        if self.show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Step 4: Pulse Statistics", fontsize=16, fontweight="bold")

            # Histogram: Pulse Width
            axes[0, 0].hist(
                pulse_widths, bins=50, color="blue", alpha=0.7, edgecolor="black"
            )
            axes[0, 0].axvline(
                np.mean(pulse_widths),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean={np.mean(pulse_widths):.1f}",
            )
            axes[0, 0].set_xlabel("Pulse Width (samples)")
            axes[0, 0].set_ylabel("Count")
            axes[0, 0].set_title("Pulse Width Distribution")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Histogram: Pulse Interval
            axes[0, 1].hist(
                intervals, bins=50, color="green", alpha=0.7, edgecolor="black"
            )
            axes[0, 1].axvline(
                np.mean(intervals),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean={np.mean(intervals):.1f}",
            )
            axes[0, 1].set_xlabel("Pulse Interval (samples)")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title("Pulse Interval Distribution")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Histogram: Amplitude (Absolute)
            axes[1, 0].hist(
                amplitudes, bins=50, color="orange", alpha=0.7, edgecolor="black"
            )
            axes[1, 0].axvline(
                np.mean(amplitudes),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean={np.mean(amplitudes):.6f}V",
            )
            axes[1, 0].set_xlabel("Amplitude (V)")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Amplitude Distribution")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Histogram: Amplitude (Normalized)
            mean_amplitude = np.mean(amplitudes)
            normalized_amplitudes = [a / mean_amplitude for a in amplitudes]
            axes[1, 1].hist(
                normalized_amplitudes,
                bins=50,
                color="purple",
                alpha=0.7,
                edgecolor="black",
            )
            axes[1, 1].axvline(
                1.0, color="red", linestyle="--", linewidth=2, label="Mean=1.0"
            )
            axes[1, 1].set_xlabel("Normalized Amplitude (a.u.)")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Normalized Amplitude Distribution")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # שמירה
            save_path = self.output_dir / "step4_statistics_histograms.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  💾 Saved: {save_path}")

            plt.show()

        # 6. גרף LED Number vs Amplitude (Absolute & Normalized)
        print(f"\n📈 Phase 6: Creating LED vs Amplitude Plots...")

        if self.show_plots:
            led_numbers = [p["pulse_num"] for p in pulse_data]
            amplitudes = [p["amplitude"] for p in pulse_data]
            mean_amplitude = np.mean(amplitudes)
            normalized_amplitudes = [a / mean_amplitude for a in amplitudes]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(
                "Step 4: LED Number vs Amplitude", fontsize=16, fontweight="bold"
            )

            # Absolute Amplitude
            ax1.plot(led_numbers, amplitudes, "b-", linewidth=0.5, alpha=0.7)
            ax1.axhline(
                mean_amplitude,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean={mean_amplitude:.6f}V",
            )
            ax1.set_xlabel("LED Number")
            ax1.set_ylabel("Amplitude (V)")
            ax1.set_title("Absolute Amplitude")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Normalized Amplitude
            ax2.plot(led_numbers, normalized_amplitudes, "g-", linewidth=0.5, alpha=0.7)
            ax2.axhline(1.0, color="red", linestyle="--", linewidth=2, label="Mean=1.0")
            ax2.set_xlabel("LED Number")
            ax2.set_ylabel("Normalized Amplitude (a.u.)")
            ax2.set_title("Normalized Amplitude (divided by mean)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # שמירה
            save_path = self.output_dir / "step4_led_vs_amplitude.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  💾 Saved: {save_path}")

            plt.show()

        # 7. גרף Peak STD per LED
        print(f"\n📈 Phase 7: Creating Peak STD Plot...")

        if self.show_plots:
            led_numbers = [p["pulse_num"] for p in pulse_data]
            peak_stds = [p["peak_std"] for p in pulse_data]
            mean_std = np.mean(peak_stds)

            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            fig.suptitle(
                "Step 4: Peak STD per LED", fontsize=16, fontweight="bold"
            )

            ax.plot(led_numbers, peak_stds, "purple", linewidth=0.5, alpha=0.7)
            ax.axhline(
                mean_std,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean STD={mean_std:.6f}V",
            )
            ax.set_xlabel("LED Number")
            ax.set_ylabel("Peak STD (V)")
            ax.set_title("Standard Deviation of Peak Region Points")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # שמירה
            save_path = self.output_dir / "step4_peak_std.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  💾 Saved: {save_path}")

            plt.show()

        print(f"\n✅ Step 4 completed successfully!")
        
        # זיהוי אנומליות
        self._detect_anomalies(pulse_data)
        
        # יצירת קובץ CSV
        self._export_to_csv(pulse_data)
        
        print(f"{'='*80}\n")

    def _detect_anomalies(self, pulse_data):
        """זיהוי אנומליות בפולסים"""
        print(f"\n🔍 Anomaly Detection...")
        
        # חישוב סטטיסטיקות
        amplitudes = [p["amplitude"] for p in pulse_data]
        pulse_widths = [p["pulse_width"] for p in pulse_data]
        intervals = [p["interval"] for p in pulse_data if p["interval"] > 0]
        peak_stds = [p["peak_std"] for p in pulse_data]
        
        amp_mean = np.mean(amplitudes)
        amp_std = np.std(amplitudes)
        width_mean = np.mean(pulse_widths)
        width_std = np.std(pulse_widths)
        interval_mean = np.mean(intervals)
        interval_std = np.std(intervals)
        std_mean = np.mean(peak_stds)
        std_std = np.std(peak_stds)
        
        # סף לזיהוי אנומליות (3 סטיות תקן)
        anomaly_threshold = 3
        
        anomalies = {
            "amplitude": [],
            "pulse_width": [],
            "interval": [],
            "peak_std": []
        }
        
        for p in pulse_data:
            led_num = p["pulse_num"]
            
            # אנומליות באמפליטודה
            if abs(p["amplitude"] - amp_mean) > anomaly_threshold * amp_std:
                anomalies["amplitude"].append({
                    "led": led_num,
                    "value": p["amplitude"],
                    "deviation": (p["amplitude"] - amp_mean) / amp_std
                })
            
            # אנומליות בגודל פולס
            if abs(p["pulse_width"] - width_mean) > anomaly_threshold * width_std:
                anomalies["pulse_width"].append({
                    "led": led_num,
                    "value": p["pulse_width"],
                    "deviation": (p["pulse_width"] - width_mean) / width_std
                })
            
            # אנומליות במרחק בין פולסים
            if p["interval"] > 0 and abs(p["interval"] - interval_mean) > anomaly_threshold * interval_std:
                anomalies["interval"].append({
                    "led": led_num,
                    "value": p["interval"],
                    "deviation": (p["interval"] - interval_mean) / interval_std
                })
            
            # אנומליות ב-STD של השיא
            if abs(p["peak_std"] - std_mean) > anomaly_threshold * std_std:
                anomalies["peak_std"].append({
                    "led": led_num,
                    "value": p["peak_std"],
                    "deviation": (p["peak_std"] - std_mean) / std_std
                })
        
        # הדפסת דוח אנומליות
        total_anomalies = sum(len(v) for v in anomalies.values())
        print(f"\n📊 Anomaly Report (>{anomaly_threshold}σ):")
        print(f"  Total anomalies found: {total_anomalies}")
        
        if anomalies["amplitude"]:
            print(f"\n  ⚠️ Amplitude anomalies: {len(anomalies['amplitude'])}")
            for a in anomalies["amplitude"][:5]:  # הצגת 5 ראשונים
                print(f"    LED {a['led']}: {a['value']:.6f}V ({a['deviation']:+.2f}σ)")
            if len(anomalies["amplitude"]) > 5:
                print(f"    ... and {len(anomalies['amplitude'])-5} more")
        
        if anomalies["pulse_width"]:
            print(f"\n  ⚠️ Pulse width anomalies: {len(anomalies['pulse_width'])}")
            for a in anomalies["pulse_width"][:5]:
                print(f"    LED {a['led']}: {a['value']} samples ({a['deviation']:+.2f}σ)")
            if len(anomalies["pulse_width"]) > 5:
                print(f"    ... and {len(anomalies['pulse_width'])-5} more")
        
        if anomalies["interval"]:
            print(f"\n  ⚠️ Interval anomalies: {len(anomalies['interval'])}")
            for a in anomalies["interval"][:5]:
                print(f"    LED {a['led']}: {a['value']} samples ({a['deviation']:+.2f}σ)")
            if len(anomalies["interval"]) > 5:
                print(f"    ... and {len(anomalies['interval'])-5} more")
        
        if anomalies["peak_std"]:
            print(f"\n  ⚠️ Peak STD anomalies: {len(anomalies['peak_std'])}")
            for a in anomalies["peak_std"][:5]:
                print(f"    LED {a['led']}: {a['value']:.6f}V ({a['deviation']:+.2f}σ)")
            if len(anomalies["peak_std"]) > 5:
                print(f"    ... and {len(anomalies['peak_std'])-5} more")
        
        if total_anomalies == 0:
            print("  ✅ No significant anomalies detected")
        
        self.anomalies = anomalies
        return anomalies
    
    def _export_to_csv(self, pulse_data):
        """ייצוא תוצאות ל-CSV"""
        import csv
        
        # יצירת שם קובץ מבוסס על שם קובץ המקור
        csv_filename = self.bin_file_path.stem + "_analyzed.csv"
        csv_path = self.output_dir / csv_filename
        
        print(f"\n💾 Exporting to CSV: {csv_path}")
        
        # זיהוי אנומליות עבור כל LED
        amp_mean = np.mean([p["amplitude"] for p in pulse_data])
        amp_std = np.std([p["amplitude"] for p in pulse_data])
        width_mean = np.mean([p["pulse_width"] for p in pulse_data])
        width_std = np.std([p["pulse_width"] for p in pulse_data])
        std_mean = np.mean([p["peak_std"] for p in pulse_data])
        std_std_value = np.std([p["peak_std"] for p in pulse_data])
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'LED_Number', 'Peak_V', 'Valley_V', 'Amplitude_V', 'Peak_STD_V',
                'Pulse_Width_samples', 'Interval_samples',
                'Amplitude_Anomaly', 'Width_Anomaly', 'STD_Anomaly'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for p in pulse_data:
                # חישוב סטיות תקן
                amp_sigma = (p["amplitude"] - amp_mean) / amp_std if amp_std > 0 else 0
                width_sigma = (p["pulse_width"] - width_mean) / width_std if width_std > 0 else 0
                std_sigma = (p["peak_std"] - std_mean) / std_std_value if std_std_value > 0 else 0
                
                writer.writerow({
                    'LED_Number': p["pulse_num"],
                    'Peak_V': f"{p['peak_value']:.6f}",
                    'Valley_V': f"{p['valley_value']:.6f}",
                    'Amplitude_V': f"{p['amplitude']:.6f}",
                    'Peak_STD_V': f"{p['peak_std']:.6f}",
                    'Pulse_Width_samples': p['pulse_width'],
                    'Interval_samples': p['interval'],
                    'Amplitude_Anomaly': 'YES' if abs(amp_sigma) > 3 else 'NO',
                    'Width_Anomaly': 'YES' if abs(width_sigma) > 3 else 'NO',
                    'STD_Anomaly': 'YES' if abs(std_sigma) > 3 else 'NO'
                })
        
        print(f"  ✅ CSV exported successfully: {len(pulse_data):,} rows")

    def _step5_peak_valley_calculation(self):
        """Step 5: Peak Valley Calculation and Pulse Amplitudes"""
        if self.voltage_data is None or self.threshold is None:
            print(
                "⚠️ No voltage data or threshold available for Peak Valley calculation"
            )
            return

        print(f"\n🔍 Step 5: Peak Valley Calculation")

        # זיהוי עליות וירידות מחדש על הדאטה הנקי
        above_threshold = self.voltage_data > self.threshold

        # חיפוש חיתוכי סף
        crossings = []
        crossing_types = []

        for i in range(1, len(above_threshold)):
            if not above_threshold[i - 1] and above_threshold[i]:  # עליה
                crossings.append(i)
                crossing_types.append("up")
            elif above_threshold[i - 1] and not above_threshold[i]:  # ירידה
                crossings.append(i)
                crossing_types.append("down")

        # זיהוי פולסים מלאים (עליה + ירידה)
        pulse_starts = []
        pulse_ends = []

        for i in range(len(crossing_types) - 1):
            if crossing_types[i] == "up":
                # חיפוש הירידה הבאה
                for j in range(i + 1, len(crossing_types)):
                    if crossing_types[j] == "down":
                        pulse_starts.append(crossings[i])
                        pulse_ends.append(crossings[j])
                        break

        print(f"Found {len(pulse_starts)} complete pulses for amplitude analysis")

        # בדיקת התאמה לרצפת LEDs
        if self.expected_led_count:
            if len(pulse_starts) != self.expected_led_count:
                print(
                    f"⚠️ Warning: Expected {self.expected_led_count} LEDs (#{self.first_led_number}-#{self.last_led_number})"
                )
                print(f"⚠️ Warning: Found {len(pulse_starts)} pulses - mismatch!")
            else:
                print(
                    f"✅ Perfect match: {len(pulse_starts)} pulses = {self.expected_led_count} expected LEDs"
                )

        # חישוב אמפליטודות
        amplitudes = []
        peak_values = []
        valley_values = []

        for i, (start_idx, end_idx) in enumerate(zip(pulse_starts, pulse_ends)):
            # Peak: חישוב במרכז הפולס - 50% המרכזיים מבחינת מיקום
            pulse_region = self.voltage_data[start_idx : end_idx + 1]
            pulse_length = len(pulse_region)
            # קח 50% מרכזיים (רבע מכל צד)
            quarter = pulse_length // 4
            center_start = quarter
            center_end = pulse_length - quarter
            central_50_percent_indices = np.arange(center_start, center_end)
            peak_value = np.mean(pulse_region[central_50_percent_indices])

            # Valley: חישוב באזור בין פולסים
            if i < len(pulse_starts) - 1:  # לא הפולס האחרון
                # Valley באזור בין הפולס הנוכחי לפולס הבא
                valley_start = end_idx + 1
                valley_end = pulse_starts[i + 1] - 1
                if valley_end > valley_start:
                    valley_region = self.voltage_data[valley_start : valley_end + 1]
                else:
                    # אם האזור קטן מדי, קח כמה נקודות אחרי הפולס
                    valley_region = self.voltage_data[
                        end_idx + 1 : min(len(self.voltage_data), end_idx + 10)
                    ]
            else:
                # עבור הפולס האחרון, קח נקודות אחרי הפולס
                valley_region = self.voltage_data[
                    end_idx + 1 : min(len(self.voltage_data), end_idx + 20)
                ]

            if len(valley_region) > 0:
                # Valley: קח 50% מרכזיים מבחינת מיקום (רבע מכל צד)
                valley_length = len(valley_region)
                quarter = valley_length // 4
                center_start = quarter
                center_end = valley_length - quarter
                central_valley_indices = np.arange(center_start, center_end)
                valley_value = np.mean(valley_region[central_valley_indices])
            else:
                # אם אין אזור valley, השתמש בנקודה אחרי הפולס
                valley_value = self.voltage_data[
                    min(len(self.voltage_data) - 1, end_idx + 1)
                ]

            # חישוב אמפליטודה
            amplitude = peak_value - valley_value

            amplitudes.append(amplitude)
            peak_values.append(peak_value)
            valley_values.append(valley_value)

            # הדפסת פרטי LED (רק אם verbose mode)
            # led_number = self.first_led_number + i
            # print(
            #     f"  LED #{led_number}: Peak={peak_value:.6f}V, Valley={valley_value:.6f}V, Amplitude={amplitude:.6f}V"
            # )

        # שמירת התוצאות
        self.peak_amplitudes = amplitudes
        self.detected_pulses = [
            {
                "led_number": self.first_led_number + i,
                "start_index": pulse_starts[i],
                "end_index": pulse_ends[i],
                "peak_value": peak_values[i],
                "valley_value": valley_values[i],
                "amplitude": amplitudes[i],
            }
            for i in range(len(pulse_starts))
        ]

        # סטטיסטיקות
        if amplitudes:
            mean_amplitude = np.mean(amplitudes)
            std_amplitude = np.std(amplitudes)
            min_amplitude = np.min(amplitudes)
            max_amplitude = np.max(amplitudes)

            print(f"\n📊 Amplitude Statistics:")
            print(f"  Mean amplitude: {mean_amplitude:.6f}V")
            print(f"  Std deviation: {std_amplitude:.6f}V (Peak STD)")
            print(f"  Min amplitude: {min_amplitude:.6f}V")
            print(f"  Max amplitude: {max_amplitude:.6f}V")
            print(f"  Amplitude range: {max_amplitude - min_amplitude:.6f}V")
            print(
                f"  CV (Std/Mean): {std_amplitude/mean_amplitude:.4f} ({100*std_amplitude/mean_amplitude:.2f}%)"
            )

        # ויזואליזציה
        if self.show_plots and len(pulse_starts) > 0:
            self._plot_peak_valley_analysis(
                pulse_starts[: min(20, len(pulse_starts))],
                pulse_ends[: min(20, len(pulse_starts))],
                peak_values[: min(20, len(peak_values))],
                valley_values[: min(20, len(valley_values))],
            )

    def _plot_peak_valley_analysis(
        self, pulse_starts, pulse_ends, peak_values, valley_values
    ):
        """ויזואליזציה של ניתוח Peak Valley עם הדגמת 50% selection"""
        print("📊 Creating Peak Valley visualization with 50% selection demo...")

        # בחירת איזור להצגה - סביב הפולסים הראשונים
        if len(pulse_starts) == 0 or self.voltage_data is None:
            return

        display_start = max(0, pulse_starts[0] - 100)
        display_end = min(
            len(self.voltage_data), pulse_ends[min(19, len(pulse_ends) - 1)] + 100
        )

        display_data = self.voltage_data[display_start:display_end]
        display_indices = np.arange(display_start, display_end)

        plt.figure(figsize=(15, 12))

        # גרף עליון - סיגנל עם סימון פולסים ו-50% selection
        plt.subplot(3, 1, 1)
        # Downsample for faster plotting if data is large (max 5K points)
        if len(display_data) > 5000:
            downsample_factor = max(1, len(display_data) // 5000)
            plot_indices = display_indices[::downsample_factor]
            plot_data = display_data[::downsample_factor]
        else:
            plot_indices = display_indices
            plot_data = display_data

        plt.plot(
            plot_indices,
            plot_data,
            "b-",
            linewidth=1,
            alpha=0.8,
            label="Voltage Signal",
        )

        if self.threshold is not None:
            plt.axhline(
                float(self.threshold),
                color="green",
                linestyle="-",
                linewidth=2,
                label=f"Threshold: {float(self.threshold):.6f}V",
            )

        # Display 50% selection for ALL pulses in display range (not just first 3)
        for i, (start_idx, end_idx) in enumerate(
            zip(pulse_starts, pulse_ends)
        ):  # All pulses shown
            if start_idx >= display_start and end_idx <= display_end:
                # Peak calculation: 50% מרכזיים של הפולס (רבע מכל צד)
                pulse_data = self.voltage_data[start_idx : end_idx + 1]
                pulse_length = len(pulse_data)
                quarter = pulse_length // 4
                center_start = quarter
                center_end = pulse_length - quarter
                peak_50_percent = np.arange(center_start, center_end)
                peak_actual_indices = start_idx + peak_50_percent

                # Valley calculation: 50% מרכזיים של אזור השפל
                if i < len(pulse_starts) - 1:
                    valley_start = end_idx + 1
                    valley_end = pulse_starts[i + 1] - 1
                    if valley_end > valley_start:
                        valley_data = self.voltage_data[valley_start : valley_end + 1]
                        valley_indices_range = np.arange(valley_start, valley_end + 1)
                    else:
                        valley_data = self.voltage_data[
                            end_idx + 1 : min(len(self.voltage_data), end_idx + 10)
                        ]
                        valley_indices_range = np.arange(
                            end_idx + 1, min(len(self.voltage_data), end_idx + 10)
                        )
                else:
                    valley_data = self.voltage_data[
                        end_idx + 1 : min(len(self.voltage_data), end_idx + 20)
                    ]
                    valley_indices_range = np.arange(
                        end_idx + 1, min(len(self.voltage_data), end_idx + 20)
                    )

                if len(valley_data) > 0:
                    valley_length = len(valley_data)
                    quarter_v = valley_length // 4
                    center_start_v = quarter_v
                    center_end_v = valley_length - quarter_v
                    valley_50_percent = np.arange(center_start_v, center_end_v)
                    valley_actual_indices = valley_indices_range[valley_50_percent]
                else:
                    valley_actual_indices = np.array([end_idx + 1])

                # סימון אזור הפולס
                plt.axvspan(
                    start_idx,
                    end_idx,
                    alpha=0.15,
                    color="gray",
                    label="Pulse Region" if i == 0 else "",
                )

                # סימון אזור Valley
                if len(valley_indices_range) > 0:
                    plt.axvspan(
                        valley_indices_range[0],
                        valley_indices_range[-1],
                        alpha=0.15,
                        color="lightblue",
                        label="Valley Region" if i == 0 else "",
                    )

                # סימון 50% central points used for Peak/Valley calculation
                if len(peak_actual_indices) > 0:
                    plt.scatter(
                        peak_actual_indices,
                        pulse_data[peak_50_percent],
                        color="red",
                        s=20,
                        alpha=0.7,
                        label="Peak 50% Points" if i == 0 else "",
                    )

                if len(valley_actual_indices) > 0 and len(valley_data) > 0:
                    plt.scatter(
                        valley_actual_indices,
                        valley_data[valley_50_percent],
                        color="blue",
                        s=20,
                        alpha=0.7,
                        label="Valley 50% Points" if i == 0 else "",
                    )

                # Final Peak and Valley points
                peak_value = (
                    np.mean(pulse_data[peak_50_percent])
                    if len(peak_50_percent) > 0
                    else pulse_data.max()
                )
                valley_value = (
                    np.mean(valley_data[valley_50_percent])
                    if len(valley_50_percent) > 0 and len(valley_data) > 0
                    else valley_data.min() if len(valley_data) > 0 else 0
                )

                plt.scatter(
                    start_idx + (end_idx - start_idx) // 2,
                    peak_value,
                    color="red",
                    s=150,
                    marker="^",
                    edgecolor="black",
                    linewidth=2,
                    label="Final Peak" if i == 0 else "",
                    zorder=5,
                )

                # Valley נמצא באזור בין הפולסים, לא במרכז הפולס
                valley_center = (
                    valley_indices_range[len(valley_indices_range) // 2]
                    if len(valley_indices_range) > 0
                    else end_idx + 1
                )
                plt.scatter(
                    valley_center,
                    valley_value,
                    color="blue",
                    s=150,
                    marker="v",
                    edgecolor="black",
                    linewidth=2,
                    label="Final Valley" if i == 0 else "",
                    zorder=5,
                )

                # LED number annotation
                led_number = self.first_led_number + i
                plt.annotate(
                    f"LED {led_number}",
                    xy=(start_idx + (end_idx - start_idx) // 2, float(peak_value)),
                    xytext=(5, 15),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                )

        plt.title("Peak Valley Analysis - Signal with 50% Point Selection Method")
        plt.xlabel("Sample Index")
        plt.ylabel("Voltage (V)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # Middle graph - Show 10 pulses with 50% points marked
        plt.subplot(3, 1, 2)
        num_pulses_to_show = min(10, len(pulse_starts))
        if num_pulses_to_show > 0:
            # Determine display range to show 10 pulses
            demo_start = max(0, pulse_starts[0] - 50)
            demo_end = min(
                len(self.voltage_data), pulse_ends[num_pulses_to_show - 1] + 50
            )

            demo_data = self.voltage_data[demo_start:demo_end]
            demo_indices = np.arange(demo_start, demo_end)

            # Plot signal
            plt.plot(
                demo_indices,
                demo_data,
                "b-",
                linewidth=1.5,
                alpha=0.8,
                label="Voltage Signal",
            )

            # Add threshold line
            if self.threshold is not None:
                plt.axhline(
                    float(self.threshold),
                    color="green",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.5,
                    label=f"Threshold: {float(self.threshold):.6f}V",
                )

            # Mark 50% central points for each pulse
            for i in range(num_pulses_to_show):
                start_idx = pulse_starts[i]
                end_idx = pulse_ends[i]

                # Calculate Peak 50% - central 50% of pulse (quarter from each side)
                pulse_data = self.voltage_data[start_idx : end_idx + 1]
                pulse_length = len(pulse_data)
                quarter = pulse_length // 4
                center_start = quarter
                center_end = pulse_length - quarter
                peak_50_percent = np.arange(center_start, center_end)
                peak_actual_indices = start_idx + peak_50_percent

                # Calculate Valley 50% - central 50% of valley region
                if i < len(pulse_starts) - 1:
                    valley_start = end_idx + 1
                    valley_end = pulse_starts[i + 1] - 1
                    if valley_end > valley_start:
                        valley_data = self.voltage_data[valley_start : valley_end + 1]
                        valley_indices_range = np.arange(valley_start, valley_end + 1)
                    else:
                        valley_data = self.voltage_data[
                            end_idx + 1 : min(len(self.voltage_data), end_idx + 10)
                        ]
                        valley_indices_range = np.arange(
                            end_idx + 1, min(len(self.voltage_data), end_idx + 10)
                        )
                else:
                    valley_data = self.voltage_data[
                        end_idx + 1 : min(len(self.voltage_data), end_idx + 20)
                    ]
                    valley_indices_range = np.arange(
                        end_idx + 1, min(len(self.voltage_data), end_idx + 20)
                    )

                if len(valley_data) > 0:
                    valley_length = len(valley_data)
                    quarter_v = valley_length // 4
                    center_start_v = quarter_v
                    center_end_v = valley_length - quarter_v
                    valley_50_percent = np.arange(center_start_v, center_end_v)
                    valley_actual_indices = valley_indices_range[valley_50_percent]

                    # Plot 50% central points
                    plt.scatter(
                        peak_actual_indices,
                        pulse_data[peak_50_percent],
                        color="red",
                        s=15,
                        alpha=0.6,
                        label="Peak 50%" if i == 0 else "",
                    )
                    plt.scatter(
                        valley_actual_indices,
                        valley_data[valley_50_percent],
                        color="blue",
                        s=15,
                        alpha=0.6,
                        label="Valley 50%" if i == 0 else "",
                    )

            plt.title(
                f"First {num_pulses_to_show} Pulses - Red=Peak 50% Used, Blue=Valley 50% Used"
            )
            plt.xlabel("Sample Index")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # גרף תחתון - היסטוגרמת אמפליטודות
        plt.subplot(3, 1, 3)
        if len(self.peak_amplitudes) > 1:
            plt.hist(
                self.peak_amplitudes,
                bins=min(20, len(self.peak_amplitudes)),
                alpha=0.7,
                color="lightblue",
                edgecolor="black",
            )
            plt.axvline(
                float(np.mean(self.peak_amplitudes)),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {float(np.mean(self.peak_amplitudes)):.6f}V",
            )
            plt.axvline(
                float(np.mean(self.peak_amplitudes) + np.std(self.peak_amplitudes)),
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"+1 STD",
            )
            plt.axvline(
                float(np.mean(self.peak_amplitudes) - np.std(self.peak_amplitudes)),
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"-1 STD",
            )
        else:
            plt.bar(
                range(len(self.peak_amplitudes)),
                self.peak_amplitudes,
                alpha=0.7,
                color="lightblue",
                edgecolor="black",
            )

        plt.title(f"Amplitude Distribution - {len(self.peak_amplitudes)} LEDs")
        plt.xlabel("Amplitude (V)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import sys

    # Allow file path as command-line argument
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = r"C:\Users\geller\OneDrive - HP Inc\data\W3\complete thermal test\led_data_06.bin"

    # אפשרות להגדיר module או ערכים ידניים
    # ניתן להעביר כפרמטר: module_name או module_id או first_led,last_led
    # לדוגמה:
    #   python led_reader_test.py file.bin Rear
    #   python led_reader_test.py file.bin 0
    #   python led_reader_test.py file.bin 2304 15615
    
    module_config = None
    if len(sys.argv) > 2:
        module_param = sys.argv[2]
        
        # בדיקה אם זה module_name או module_id
        for config in CROSSTALK_CONFIG:
            if module_param.lower() == config["name"].lower() or module_param == str(config["module_id"]):
                module_config = config
                break
        
        # אם לא מצאנו module, בדיקה אם זה טווח ידני
        if module_config is None and len(sys.argv) > 3:
            try:
                first_led = int(sys.argv[2])
                last_led = int(sys.argv[3])
                print(f"📋 Using manual LED range: {first_led} - {last_led}")
            except ValueError:
                print(f"⚠️ Invalid module parameter: {module_param}")
                modules_list = ", ".join([f"{c['name']} ({c['module_id']})" for c in CROSSTALK_CONFIG])
                print(f"Available modules: {modules_list}")
                sys.exit(1)
    
    # אם לא הוגדר כלום, ברירת מחדל ל-Rear
    if module_config is None and 'first_led' not in locals():
        module_config = CROSSTALK_CONFIG[0]  # Rear
    
    if module_config:
        first_led = module_config["first_led"]
        last_led = module_config["last_led"]
        expected_leds = module_config["expected_leds"]
        print(f"📋 Selected Module: {module_config['name']} (ID: {module_config['module_id']})")
        print(f"📋 LED Range: {first_led} - {last_led} ({expected_leds:,} LEDs expected)")

    # Set show_plots=True to see visualizations (False for faster timing)
    analyzer = LEDAnalyzer(
        test_file, first_led_number=first_led, last_led_number=last_led, show_plots=True
    )
    analyzer.run_analysis()
