import libm2k
import adi
from sys import exit
import numpy as np
import genalyzer_advanced as gn
import workshop

# 0. Configuration
fs_out = 7500000  # Generated waveform sample rate
fs_in  = 40000000 # Received waveform sample rate. AD4080 fixed at 40Msps

# Tone parameters
phase = 0.0  # Tone phase
td = 0.0 # ASK MARK
tj = 0.0 # ASK MARK
fsr = 2.0  # Full-scale range in ?Volts?
fund_freq = 100000.0  # Hz

# This is a list of the amplitudes (in dBfs) of the fundamental (first element)
# and harmonics. You can add more harmonics to the list, but we'll start
# out with just the 2nd, 3rd, and 4th.
# Replace -200.0 with greater values to add harmonics
harm_dbfs = [-3.0, -23.0, -20.0, -20.0]

# These are lists of the frequencies (Hz) and amplitudes (in dBfs) of
# interfering tones or "noise tones". Genalyzer will interpret them as not
# harmonically related and add them to the total noise.
noise_freqs = [150000.0, 250000.0, 350000.0, 450000.0]
# Replace -200.0 with greater values to add noise tones
noise_dbfs = [-40, -60, -70, -50]

# FFT parameters
window = gn.Window.BLACKMAN_HARRIS  # FFT window
npts = 16384        # Receive buffer size
navg = 2            # No. of fft averages
nfft = npts // navg # No. of points per FFT

# 1. Connect to M2K and AD4080
m2k = libm2k.m2kOpen('usb:')
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
ad4080 = adi.ad4080('serial:/dev/ttyACM0,230400,8n1')
if ad4080 is None:
    print("Connection Error: No AD4080 device available/connected to your PC.")
    exit(1)

# 2. Generate waveform containing both the wanted signal and some noise

# Convert dBfs to amplitudes for both harmonics and noise
harm_ampl = [(fsr / 2) * 10 ** (x / 20) for x in harm_dbfs]
noise_ampl = [(fsr / 2) * 10 ** (x / 20) for x in noise_dbfs]

# Nudge fundamental to the closest coherent bin
fund_freq = gn.coherent(nfft, fs_out, fund_freq)

# Build up the signal from the fundamental, harmonics, and noise tones
awf = np.zeros(npts)

for harmonic in range(len(harm_dbfs)):
    freq = fund_freq * (harmonic + 1)
    print(f"Frequency: {freq} ({harm_dbfs[harmonic]} dBfs)")

    awf += gn.cos(npts, fs_out, harm_ampl[harmonic], freq, phase, td, tj)

for tone in range(len(noise_freqs)):
    freq = noise_freqs[tone]
    freq = gn.coherent(nfft, fs_out, noise_freqs[tone])

    print(f"Noise Frequency: {freq} ({noise_dbfs[tone]} dBfs)")
    awf += gn.cos(npts, fs_out, noise_ampl[tone], freq, phase, td, tj)

# 3. Transmit generated waveform
aout.push([awf]) # Would be [awf0, awf1] if sending data to multiple channels

# 4. Receive one buffer of samples
ad4080.rx_buffer_size = npts
ad4080.sample_rate = fs_in
data_in = ad4080.rx()

# Convert ADC codes to Volts
# ioan: Why divide by 2**20 instead of 2**19, when we have 20 bit 2s complement?
data_in = data_in * ad4080.scale / 2**20

# 5. Analyze recorded waveform
workshop.fourier_analysis(data_in, fundamental = fund_freq, sampling_rate = fs_in, window = window)
