import matplotlib.pyplot as pl
import libm2k
import adi
from sys import exit
import numpy as np
import genalyzer_advanced as gn
import workshop
import argparse

# 0. Configuration
fs_in  = 40e6    # Received waveform sample rate. AD4080ARDZ fixed at 40Msps
fs_out = 750000  # Generated waveform sample rate

plot_freq_range = 100000 # Plot all FFTs with the same frequency range for easy comparison

# FFT parameters
window = gn.Window.BLACKMAN_HARRIS  # FFT window
npts = 16384        # Receive buffer size - maximum for this board
navg = 1            # No. of fft averages
nfft = npts // navg # No. of points per FFT


parser = argparse.ArgumentParser(
    description='Generate wideband noise on the M2K, record it using the AD4080ARDZ, comparing it to the theoretical sinc1 response, taking into account frequency folding.')
parser.add_argument('-m', '--m2k_uri', default='ip:192.168.2.1',
    help='LibIIO context URI of the ADALM2000')
parser.add_argument('-a', '--ad4080_uri', default='serial:/dev/ttyACM0,230400,8n1',
    help='LibIIO context URI of the EVAL-AD4080ARDZ')
args = vars(parser.parse_args())

# 1. Connect to M2K and AD4080
m2k = libm2k.m2kOpen(args['m2k_uri'])
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
ad4080 = adi.ad4080(args['ad4080_uri'])
if ad4080 is None:
    print("Connection Error: No AD4080 device available/connected to your PC.")
    exit(1)

# 2. Generate waveform with "zebra" spectrum
spectrum = np.array(
    # Funky formula ahead makes it so that we have bands of noise arranged such that:
    # - 
    # - after folding, the bands don't overlap the unfolded ones (+7/8)
    [int((i // int(fs_in / 1024 / (8 + 7/8))) % 8 == 0) for i in range(160000)] +
    [0 for i in range(160000, fs_out//2)]
)
awf = workshop.time_points_from_freq(spectrum) * 1000 # scaling factor arbitrarily chosen
times = np.arange(len(awf)) / fs_out # Time of each sample

pl.figure(1, figsize=(10, 10))
pl.subplot(2, 1, 1)
pl.title(f"Generated waveform")
pl.plot(times, awf)
pl.ylim(-5, 5)
pl.grid(True)

# Compute and plot generated signal FFT
fft_cplx = np.fft.fft(awf)[:fs_out//2+1]
fft_db = gn.db(fft_cplx)
freq_axis = gn.freq_axis(fs_out, gn.FreqAxisType.REAL, fs_out)

pl.subplot(2, 1, 2)
pl.title(f"Generated FFT")
pl.plot(freq_axis, fft_db)
pl.xlim(0, plot_freq_range)
pl.ylim(-160, 0)
pl.grid(True)

# 3. Transmit generated waveform
aout.push([awf]) # Would be [awf0, awf1] if sending data to multiple channels

# 4. Receive one buffer of samples

for i, decimation in enumerate([ #1, 2, 
    #4, 8, 16, 32, 64, 128,
    256,
    #512, 1024
]):
    print(i, f'{decimation=}')
    pl.figure(i+2, figsize=(10, 10))

    fs_in_effective = fs_in / decimation
    times = np.arange(npts) / fs_in_effective

    if decimation == 1:
        ad4080.filter_sel = 'none'
    else:
        ad4080.filter_sel = 'sinc1'
        ad4080.sinc_dec_rate = decimation

    num_avg = 4
    fft_db = np.zeros(nfft // 2 + 1)

    ad4080.rx_buffer_size = npts
    ad4080.sample_rate = fs_in

    for _ in range(num_avg):
        data_in = ad4080.rx() * ad4080.scale / 1e6 # uV -> V
        print(len(data_in))

        # Remove DC component
        data_in = data_in - np.average(data_in)

        # Compute FFT
        code_fmt = gn.CodeFormat.TWOS_COMPLEMENT
        rfft_scale = gn.RfftScale.NATIVE
        fft_cplx = gn.rfft(np.array(data_in), navg, nfft, window, code_fmt, rfft_scale)
        fft_db += gn.db(fft_cplx)

    fft_db /= num_avg

    pl.subplot(2, 1, 1)
    pl.title(f"Recorded waveform, decimation {decimation}")
    pl.plot(times, data_in)
    pl.ylim(-5, 5)
    pl.grid(True)

    # Compute frequency axis
    freq_axis = gn.freq_axis(nfft, gn.FreqAxisType.REAL, fs_in_effective)

    # Plot FFT
    pl.subplot(2, 1, 2)
    pl.title(f"Recorded FFT, decimation {decimation}, {num_avg} averages")
    pl.plot(freq_axis, fft_db)
    pl.xlim(0, plot_freq_range)
    pl.ylim(-160, 0)
    pl.grid(True)

    # Compute expected sinc response
    for fold in range(2):
        sinc1 = np.sinc(fold + (-1)**fold * freq_axis / fs_in_effective)
        pl.plot(freq_axis, gn.db(np.complex128(sinc1))-40, 'k--')
    
pl.show()
