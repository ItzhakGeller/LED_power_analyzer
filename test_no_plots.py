from led_reader_test import LEDAnalyzer
import sys

# Change to workspace directory where the bin file is
bin_file = (
    r"C:\Users\geller\OneDrive - HP Inc\data\W3\complete thermal test\led_data_06.bin"
)

print(f"Testing with file: {bin_file}")
analyzer = LEDAnalyzer(bin_file, 1, 15000, show_plots=False)
analyzer.run_analysis()
