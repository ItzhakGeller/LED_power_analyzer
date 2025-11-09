# LED Signal Processing Analyzer

A comprehensive signal processing toolkit for analyzing oscilloscope data from LED measurements stored in binary (BIN) format.

## Project Overview

This project implements multiple approaches for analyzing LED measurement data, starting with a threshold-based peak detection method and expanding to more sophisticated techniques including wavelet analysis and machine learning approaches.

## Current Implementation: Approach 1

**Threshold-Based Peak Detection with Statistical Analysis**

### Features
- Binary file reading and data transformation
- Signal preprocessing with filtering and baseline removal
- Adaptive pulse detection using statistical thresholding
- Pulse amplitude calculation and plateau averaging
- Data integrity checks and quality assessment
- Comprehensive visualization and reporting
- CSV export functionality

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test with Synthetic Data**
   ```bash
   python test_analyzer.py
   ```

3. **Analyze Real Data**
   ```python
   from led_analyzer import LEDSignalAnalyzer, AnalysisParameters
   
   # Configure analysis parameters
   params = AnalysisParameters(
       sampling_rate=1e6,      # 1 MHz sampling rate
       filter_cutoff=1000,     # 1 kHz low-pass filter
       min_pulse_height=0.1,   # 100 mV minimum pulse height
       min_pulse_width=1e-3,   # 1 ms minimum pulse width
       max_pulse_width=10e-3   # 10 ms maximum pulse width
   )
   
   # Initialize analyzer
   analyzer = LEDSignalAnalyzer(params)
   
   # Load and analyze data
   time_data, voltage_data = analyzer.read_bin_file("your_data.bin")
   filtered_data = analyzer.preprocess_signal()
   pulses = analyzer.detect_pulses()
   
   # Generate results
   analyzer.plot_signal_overview("analysis_overview.png")
   analyzer.export_results_to_csv("results.csv")
   report = analyzer.generate_analysis_report()
   ```

## Project Structure

```
led-signal-analyzer/
├── ANALYSIS_APPROACHES.md     # Documentation of all approaches
├── led_analyzer.py            # Main analysis implementation
├── test_analyzer.py           # Testing and validation script
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Analysis Pipeline

1. **Data Reading**: Convert BIN files to time/voltage arrays
2. **Preprocessing**: Apply filtering and baseline correction
3. **Pulse Detection**: Identify LED pulse boundaries
4. **Parameter Extraction**: Calculate amplitude, width, and plateau averages
5. **Quality Assessment**: Validate pulses and detect anomalies
6. **Visualization**: Generate comprehensive analysis plots
7. **Export**: Save results to CSV with detailed metrics

## Key Parameters

### Signal Processing
- `sampling_rate`: Data acquisition sampling rate (Hz)
- `filter_cutoff`: Low-pass filter cutoff frequency (Hz)
- `filter_order`: Digital filter order

### Pulse Detection
- `min_pulse_height`: Minimum detectable pulse amplitude (V)
- `min_pulse_width`: Minimum acceptable pulse duration (s)
- `max_pulse_width`: Maximum acceptable pulse duration (s)
- `baseline_window`: Window size for baseline estimation
- `plateau_safety_margin`: Safety margin for plateau averaging

### Quality Assessment
- `snr_threshold`: Minimum signal-to-noise ratio
- `cv_threshold`: Maximum coefficient of variation for consistency

## Data Integrity Checks

### Implemented
- Pulse width consistency monitoring
- Amplitude range validation
- Signal-to-noise ratio assessment
- Baseline stability tracking

### Planned
- Inter-pulse spacing analysis
- Spectral content validation
- Crosstalk detection
- Timing jitter analysis

## Success Criteria (Approach 1)

- **Pulse Detection Accuracy**: >95% correct identification
- **Amplitude Precision**: <2% coefficient of variation
- **Baseline Stability**: <1% drift over measurement
- **Processing Speed**: <10 seconds per dataset
- **False Positive Rate**: <5% incorrect detections
- **Missing LED Detection**: >90% detection rate

## Output Files

### CSV Results (`results.csv`)
- LED_Number: Sequential LED identifier (0-based)
- Power_Voltage: Averaged plateau voltage (V)
- Peak_Voltage: Maximum pulse voltage (V)
- Amplitude: Peak-to-bottom difference (V)
- Pulse_Width_ms: Pulse duration (milliseconds)
- Quality_Score: Pulse quality metric (0-1)
- Is_Valid: Boolean validity flag

### Analysis Plots (`analysis_overview.png`)
1. **Signal Overview**: Raw data, filtered data, and baseline
2. **Pulse Detection**: Detected pulses with validity indicators
3. **LED Power Results**: LED number vs measured power

## Future Approaches

See `ANALYSIS_APPROACHES.md` for detailed descriptions of planned implementations:

- **Approach 2**: Morphological Signal Processing with Template Matching
- **Approach 3**: Wavelet-Based Multi-Resolution Analysis
- **Approach 4**: Machine Learning-Based Detection (CNN/LSTM)
- **Approach 5**: Hybrid Statistical-Physical Model

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
   ```bash
   pip install numpy scipy matplotlib pandas
   ```

2. **File Reading Error**: Check BIN file format and byte order
   - Verify `data_format` parameter ('float32' or 'float64')
   - Check `byte_order` parameter ('little' or 'big')

3. **No Pulses Detected**: Adjust detection parameters
   - Lower `min_pulse_height` threshold
   - Adjust `min_pulse_width` and `max_pulse_width`
   - Check signal preprocessing parameters

4. **Poor Quality Results**: Optimize signal processing
   - Adjust filter cutoff frequency
   - Modify baseline estimation window
   - Tune plateau safety margin

### Parameter Tuning Guide

1. **Start with test data**: Use `test_analyzer.py` to validate setup
2. **Visualize raw data**: Check signal characteristics and noise levels
3. **Adjust thresholds**: Tune detection parameters based on signal amplitude
4. **Validate results**: Check pulse detection accuracy and quality scores
5. **Iterate**: Refine parameters based on analysis results

## Contributing

When implementing new approaches:

1. Create new analysis class inheriting from base functionality
2. Update `ANALYSIS_APPROACHES.md` with implementation status
3. Add comprehensive testing and validation
4. Document parameter sensitivity and tuning guidelines
5. Benchmark against success criteria

## Contact

For questions or issues with the LED signal processing analysis, please refer to the documentation or create detailed issue reports with sample data and parameter configurations.