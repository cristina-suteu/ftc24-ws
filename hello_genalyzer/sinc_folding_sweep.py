import libm2k
import adi
import sys
import numpy as np
import genalyzer_advanced as gn
import workshop
import time
import argparse

parser = argparse.ArgumentParser(
    description='Sweep a band of noise using the M2K, record it using the AD4080ARDZ, comparing the results to the theoretical sinc1 response, taking into account frequency folding.')
parser.add_argument('-m', '--m2k_uri', default='ip:192.168.2.1',
    help='LibIIO context URI of the ADALM2000')
parser.add_argument('-a', '--ad4080_uri', default='serial:/dev/ttyACM0,230400,8n1',
    help='LibIIO context URI of the EVAL-AD4080ARDZ')
args = vars(parser.parse_args())

# 0. Configuration
decimation = 256

fs_pre = 40e6                # Pre-digital filter sample rate, AD4080ARDZ fixed at 40Msps
fs_in  = fs_pre / decimation # Actual data rate we receive after decimation
fs_out = 750000              # Generated waveform sample rate

# Plot all FFTs with the same frequency range for easy comparison
plot_freq_range = int(fs_in * 2.5) # See two and a half sinc lobes

# FFT parameters
window = gn.Window.BLACKMAN_HARRIS  # FFT window
npts = 2048         # Receive buffer size - maximum for this board
navg = 4            # No. of fft averages
nfft = npts // navg # No. of points per FFT

# 1. Connect to M2K and AD4080
m2k = libm2k.m2kOpen(args['m2k_uri'])
if m2k is None:
    print("Connection Error: No ADALM2000 device available/connected to your PC.")
    sys.exit(1)

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
    sys.exit(1)

ad4080.filter_sel = 'sinc1'
ad4080.sinc_dec_rate = decimation
ad4080.rx_buffer_size = npts

print(f'Sampling frequency: {ad4080.select_sampling_frequency}')
print(f'Available sampling frequencies: {ad4080.select_sampling_frequency_available}')
assert ad4080.select_sampling_frequency == fs_pre # Check 40Msps assumption

for i, center in enumerate(range(0, fs_out // 2, 10000)):
    # Generate signal
    signal = workshop.generate_noise_band(center, 10000, fs_out)

    # Plot generated signal
    workshop.plot_waveform_and_fft('Generated', signal, fs_out, fignum=1)

    # Send generated signal through the m2k
    aout.push([signal])
    print(f'Sending: {center=}')

    # Receive a buffer of samples from the AD4080
    recorded_time = npts / fs_in
    transfer_time = npts * 4 * 10 / 230400 # serial link at 203400 baud
    print(f'Receiving {npts} samples ({recorded_time:.3f} s, transfer should take {transfer_time:.1f} s)...')
    t0 = time.time()
    data_in = ad4080.rx()
    t1 = time.time()
    print(f'Received in {t1-t0:.1f} s')

    # Scale to Volts
    data_in = data_in * ad4080.scale / 1e6 # scale is in uV

    # Remove DC component
    data_in -= np.average(data_in)

    # Compute FFT
    fft_cplx = gn.rfft(np.array(data_in), navg, nfft, window, gn.CodeFormat.TWOS_COMPLEMENT, gn.RfftScale.NATIVE)
    fft_db = gn.db(fft_cplx)

    # Nice plot
    workshop.plot_waveform_fft_sinc1_unfolded(data_in, fft_db, fs_in, center, decimation)
