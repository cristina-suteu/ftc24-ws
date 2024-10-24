import matplotlib.pyplot as pl
import libm2k
import adi
import sys
import numpy as np
import genalyzer_advanced as gn
import workshop
import time
import argparse

# 0. Configuration
decimation = 256

fs_pre = 40e6                # Pre-digital filter sample rate, AD4080ARDZ fixed at 40Msps
fs_in  = fs_pre / decimation # Actual data rate we receive after decimation
fs_out = 750000              # Generated waveform sample rate

# Plot all FFTs with the same frequency range for easy comparison
plot_freq_range = int(fs_in * 2.5) # See two and a half sinc lobes

# FFT parameters
window = gn.Window.BLACKMAN_HARRIS  # FFT window
npts = 16384        # Receive buffer size - maximum for this board
navg = 1            # No. of fft averages
nfft = npts // navg # No. of points per FFT


parser = argparse.ArgumentParser(
    description='Sweep a band of noise using the M2K, record it using the AD4080ARDZ, comparing the results to the theoretical sinc1 response, taking into account frequency folding.')
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

ad4080.rx_buffer_size = npts
print(f'Sampling frequency: {ad4080.select_sampling_frequency}')
print(f'Available sampling frequencies: {ad4080.select_sampling_frequency_available}')
assert ad4080.select_sampling_frequency == fs_pre

if decimation == 1:
    ad4080.filter_sel = 'none'
else:
    ad4080.filter_sel = 'sinc1'
    ad4080.sinc_dec_rate = decimation

def generate_signal(center, width, fs):
    assert center+width/2 <= fs/2, "Noise band must fit below DAC nyquist frequency"

    spectrum = np.array([
        1 if center-width/2 <= f <= center+width/2 else 0
        for f in range(fs // 2)
    ])
    signal = workshop.time_points_from_freq(spectrum)
    signal = signal / np.std(signal) # Bring rms to 1V

    times = np.arange(fs) / fs # Time of each sample

    pl.figure(1, figsize=(10, 10))
    ax = pl.subplot(2, 1, 1)
    ax.clear()
    pl.title(f"Generated waveform")
    pl.plot(times, signal)
    pl.ylim(-5, 5)
    pl.grid(True)

    # Compute and plot generated signal FFT
    fft_cplx = gn.rfft(signal, 1, len(signal), gn.Window.BLACKMAN_HARRIS, gn.CodeFormat.TWOS_COMPLEMENT, gn.RfftScale.NATIVE)
    fft_db = gn.db(fft_cplx)
    freq_axis = gn.freq_axis(fs, gn.FreqAxisType.REAL, fs)

    ax = pl.subplot(2, 1, 2)
    ax.clear()
    pl.title(f"Generated FFT")
    pl.plot(freq_axis, fft_db)
    pl.xlim(0, plot_freq_range)
    pl.ylim(-160, 0)
    pl.grid(True)

    # ?? is this how I update the plot live?
    pl.draw()
    pl.pause(0.01)

    return signal

while True:
    for i, center in enumerate(range(0, plot_freq_range - 10000, 10000)):
        signal = generate_signal(center, 10000, fs_out)
        aout.push([signal])

        print(f'Sending: {center=}')

        pl.figure(2, figsize=(10, 10))

        times = np.arange(npts) / fs_in

        print(f'Receiving {npts} samples ({npts / fs_in:.3f} s, transmitted in {npts * 4 * 10 / 230400:.1f} s)...')
        t0 = time.time()
        data_in = ad4080.rx()
        t1 = time.time()
        print(f'Received in {t1-t0:.1f} s')

        # Scale to Volts
        data_in = data_in * ad4080.scale / 1e6 # scale is in uV

        # Remove DC component
        data_in -= np.average(data_in)

        # Compute FFT
        code_fmt = gn.CodeFormat.TWOS_COMPLEMENT
        rfft_scale = gn.RfftScale.NATIVE
        fft_cplx = gn.rfft(np.array(data_in), navg, nfft, window, code_fmt, rfft_scale)
        fft_db = gn.db(fft_cplx)

        ax = pl.subplot(2, 1, 1)
        ax.clear()
        pl.title(f"Recorded waveform, {decimation=}, noise band {center=}")
        pl.plot(times, data_in)
        pl.ylim(-5, 5)
        pl.grid(True)

        # Compute frequency axis
        freq_axis = gn.freq_axis(nfft, gn.FreqAxisType.REAL, fs_in)

        # Plot FFT
        ax = pl.subplot(2, 1, 2)
        ax.clear()
        pl.title(f"'Unfolded' FFT, {decimation=}, noise band {center=}")
        #pl.plot(freq_axis, fft_db)
        pl.xlim(0, plot_freq_range)
        pl.ylim(-160, 0)
        pl.grid(True)

        # Compute expected sinc response
        # for fold in range(2):
        #     sinc1 = np.sinc(fold + (-1)**fold * freq_axis / fs_in)
        #     pl.plot(freq_axis, gn.db(np.complex128(sinc1))-40, 'k--')

        # Plot unfolded signal spectrum and theoretical sinc
        for fold in range(5):
            freqs = freq_axis + fold * (fs_in // 2 + 1)
            sinc1 = gn.db(np.complex128(np.sinc(freqs / fs_in))) - 20

            pl.plot(freqs, sinc1,
                'r--' if fold > 0 else 'r',
                alpha = 0.75 ** fold,
                label = f'Theoretical sinc1 response'
            )
            pl.plot(freqs, fft_db[::(-1)**fold],
                'b--' if fold > 0 else 'b',
                alpha = 0.75 ** fold,
                label = f'Unfolded signal FFT' if fold > 0 else 'Signal FFT'
            )

        # x tick at nyquist
        pl.axvline(fs_in//2, linestyle='--', color='k')

        # arrow at generated tone
        pl.annotate('Generated', xy=(center, -20), xytext=(center, -10), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='center')

        if center > fs_in // 2:
            fold = center // (fs_in // 2)
            if fold % 2 == 0:
                center_aliased = center - (fs_in // 2) * fold
            else:
                center_aliased = (fs_in // 2) * (fold + 1) - center
            pl.annotate('Aliased', xy=(center_aliased, -20), xytext=(center_aliased, -10), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='center')

        # ?? is this how I update the plot live?
        pl.draw()
        pl.pause(0.01)
        pl.savefig(f'frames/{i:02}.png')
