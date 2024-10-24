import sys

import matplotlib.pyplot as plt
import numpy as np
from adi import ad4080
import argparse
import genalyzer_advanced as gn

# Config Parameters
# AD4080 Params
BUFFER_SIZE = 8192
SAMPLING_FREQUENCY = 40000000.0
# FFT Params
code_fmt = gn.CodeFormat.TWOS_COMPLEMENT
rfft_scale = gn.RfftScale.NATIVE # FFT Scale
window = gn.Window.BLACKMAN_HARRIS # FFT Window
npts = BUFFER_SIZE         # Receive buffer size
navg = 2             # No. of fft averages
nfft = npts // navg  # No. of points per FFT

parser = argparse.ArgumentParser(
        description='Hello AD4080 Python Script, can take URI as argument. ')
parser.add_argument('-u', "--uri", default="/dev/ttyACM0,230400,8n1",
                        help='URI of AD4080')
args = vars(parser.parse_args())

my_uri = args["uri"]

my_ad4080 = ad4080(uri=my_uri, device_name="ad4080")

print("Sampling frequency from IIO Context: ", my_ad4080.sampling_frequency)
print("Scale: ", my_ad4080.scale)

# set sampling frequency
my_ad4080.sampling_frequency = SAMPLING_FREQUENCY
# set buffer size
my_ad4080.rx_buffer_size(BUFFER_SIZE)

# grab initial chunk of data
data = my_ad4080.rx()

# scale from ADC codes to Volts
data = data * my_ad4080.scale / 1e6

# Create figure to plot results
fig, (ax1, ax2) = plt.subplots(nrows=2)
fig.set_figheight(6)
fig.set_figwidth(12)
fig.tight_layout()

# Set-up FFT Plot
ax1.set_title("Time Domain")
x = data
ax1.set_ylim([min(data) * 1.1, max(data) * 1.1])
line1, = ax1.plot(x, label="time domain")
ax1.legend()
ax2.set_title("FFT")
line2, = ax2.plot(x[:len(x)//2], label="freq. domain")
ax2.legend()

while True:

    # Collect data
    data = my_ad4080.rx()
    # ADC codes -> Volts
    data = data * my_ad4080.scale
    # Remove DC component
    data = data - np.average(data)

    # Compute FFT
    fft_cplx = gn.rfft(np.array(data), navg, nfft, window, code_fmt, rfft_scale)
    fft_db = gn.db(fft_cplx)
    # Compute frequency axis
    freq_axis = gn.freq_axis(nfft, gn.FreqAxisType.REAL, my_ad4080.sampling_frequency)

    line1.set_ydata(data)
    line2.set_xdata(freq_axis)
    line2.set_ydata(fft_db)
    plt.pause(2)
    # Exit loop and close AD4080 context
    if not plt.fignum_exists(1):
        del my_ad4080
        sys.exit()

