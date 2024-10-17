# Lifted from example:
# https://docs.obspy.org/tutorial/code_snippets/seismogram_envelopes.html
import libm2k
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import adi

import obspy
import obspy.signal
import obspy.signal.filter

# 0. Configuration
AD4080_URI = 'serial:/dev/ttyACM0,230400,8n1'

fs_out = 75000    # Transmit waveform sample rate
fs_in  = 40000000 # Received waveform sample rate. AD4080 fixed at 40Msps

# 1. Connect to M2K and AD4080
m2k = libm2k.m2kOpen('ip:192.168.2.1')
if m2k is None:
    print("Connection Error: No ADALM2000 device available/connected to your PC.")
    exit(1)

# Initialize DAC channel 0
aout = m2k.getAnalogOut()
aout.reset()
m2k.calibrateDAC()
aout.setSampleRate(0, fs_out)
aout.enableChannel(0, True)
aout.setCyclic(True) # Send buffer repeatedly, not just once

# Connect to AD4080
ad4080 = adi.ad4080(AD4080_URI)
if ad4080 is None:
    print("Connection Error: No AD4080 device available/connected to your PC.")
    exit(1)

# 2. Download waveform and resample to 7500sps
st = obspy.read("https://examples.obspy.org/RJOB_061005_072159.ehz.new")
data = st[0].data
npts = st[0].stats.npts
samprate = st[0].stats.sampling_rate

shortdata = data[int(80*samprate):int(120*samprate)]

# The entire 40 second waveform cannot be recorded in a single AD4080 buffer
# With the fixed 40Msps and maximum 1024 decimation rate, a full buffer of 16384 samples equates to 419ms
# Just for the sake of this demo, to fit the 40 second seismic waveform, we speed it up 100x
# to fit within the 419ms window
speedup_factor = 100

# Compute the number of samples after changing the sampling rate to fs_out and the 100x speedup
num_samples = int(len(shortdata) * fs_out / speedup_factor / samprate)

m2kdata = resample(shortdata, num_samples)

# Rescale data to roughly -1 .. 1 V. Should later multiply the received values by this
scaling_factor = 10000
m2kdata = m2kdata / scaling_factor

print(f'Seismograph data:    {len(shortdata)} samples @ {samprate} sps = {len(shortdata)/samprate} s')
print(f'm2k transmit buffer: {num_samples   } samples @ {fs_out  } sps = {num_samples/fs_out     } s')

# 3. Transmit waveform repeatedly
aout.push([m2kdata])

# 4. Receive waveform
ad4080.filter_sel = 'sinc1'
ad4080.sinc_dec_rate = 1024
ad4080.rx_buffer_size = 16384
ad4080.sample_rate = fs_in
print(f"Receiving {ad4080.rx_buffer_size} samples @ {fs_in / ad4080.sinc_dec_rate} sps = {ad4080.rx_buffer_size / fs_in * ad4080.sinc_dec_rate} s")
data_in = ad4080.rx()

# Remove DC bias
data_in = data_in - np.average(data_in)

# Convert ADC codes to Volts
# ioan: Why divide by 2**20 instead of 2**19, when we have 20 bit 2s complement?
data_in = data_in * ad4080.scale / 2**20

# Undo earlier scaling
data_in = data_in * scaling_factor

fs = fs_in / ad4080.sinc_dec_rate

# Envelope of filtered data
data_envelope = obspy.signal.filter.envelope(data_in)

# The plotting, plain matplotlib
plt.figure(1)
t = np.arange(0, len(data_in) / fs, 1 / fs)
plt.plot(t, data_in)
plt.plot(t, data_envelope, 'k:')
plt.title(st[0].stats.starttime)
plt.ylabel('Filtered Data w/ Envelope')
plt.xlabel('Time [s]')
#plt.xlim(80, 90)

plt.show()
