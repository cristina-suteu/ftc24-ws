import matplotlib.widgets
import matplotlib.pyplot as pl
import genalyzer_advanced as gn
from threading import Thread
import libm2k
import adi
import sys
import numpy as np
import workshop
import time
import argparse

parser = argparse.ArgumentParser(
    description='Generate a band of noise (whose frequency is adjustable by a GUI slider) on the M2K, record it using the AD4080ARDZ, comparing it to the theoretical sinc1 response, taking into account frequency folding.')
parser.add_argument('-m', '--m2k_uri', default='ip:192.168.2.1',
    help='LibIIO context URI of the ADALM2000')
parser.add_argument('-a', '--ad4080_uri', default='serial:/dev/ttyACM0,230400,8n1',
    help='LibIIO context URI of the VAL-AD4080ARDZ')
parser.add_argument('-d', '--decimation', default='256',
    choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 1024],
    help='AD4080 digital filter (sinc1) decimation. Set to 1 for no filtering.')
args = vars(parser.parse_args())

# 0. Configuration
decimation = int(args['decimation'])

fs_pre = 40e6                # Pre-digital filter sample rate, AD4080ARDZ fixed at 40Msps
fs_in  = fs_pre / decimation # Actual data rate we receive after AD4080 filtering
fs_out = 750000              # Generated waveform sample rate (up to 75 Msps)

# Plot all FFTs with the same frequency range for easy comparison
plot_freq_range = int(fs_in * 2.5) # See two and a half sinc lobes

# FFT parameters
window = gn.Window.BLACKMAN_HARRIS  # FFT window
nfft = 1024        # no. of points per FFT
navg = 4            # No. of fft averages
npts = navg * nfft  # Receive buffer size - maximum for this board is 16384

times = np.arange(npts) / fs_in
freq_axis = gn.freq_axis(nfft, gn.FreqAxisType.REAL, fs_in)

received_center_frequency = None
selected_center_frequency = fs_in // 4 

data_in = np.zeros(npts)
fft_db = np.zeros(nfft // 2 + 1)

running = True

last_status_time = time.time()
status_msg = None
def status(msg):
    global last_status_time, status_msg
    now = time.time()
    delta = now - last_status_time
    last_status_time = now
    print(f'[{now: 8.3f}] (+{delta: 5.3f}) {msg}')
    status_msg = f'{msg} (+{delta: 5.3f})'

status('Initializing')

def slider_changed(value):
    global selected_center_frequency
    selected_center_frequency = value

def iio_thread():
    global received_center_frequency, data_in, fft_db
    
    # Connect to M2K and AD4080 and initialize them
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

    # Initialize ADC
    ad4080.rx_buffer_size = npts
    print(f'Sampling frequency: {ad4080.select_sampling_frequency}')
    print(f'Available sampling frequencies: {ad4080.select_sampling_frequency_available}')
    assert ad4080.select_sampling_frequency == fs_pre

    if decimation == 1:
        ad4080.filter_sel = 'none'
    else:
        ad4080.filter_sel = 'sinc1'
        ad4080.sinc_dec_rate = decimation

    transmitting_center_frequency = None

    while running:
        # If running for the first time or after a slider change, regenerate waveform
        if transmitting_center_frequency != selected_center_frequency:
            status(f"Generating new waveform with center frequency = {selected_center_frequency} Hz")
            noise_band_width = 10000
            noise_band_lo = int(max(selected_center_frequency - noise_band_width // 2, 1))
            noise_band_hi = int(min(selected_center_frequency + noise_band_width // 2, fs_out // 2))

            # FIXME: Generate a shorter waveform with the same spectral content!
            #        Currently generating one whole second, but, for example, noise
            #        around 10k, would only need a circa 200us waveform
            spectrum = np.concatenate((
                np.zeros(noise_band_lo),
                np.ones(noise_band_hi - noise_band_lo),
                np.zeros(fs_out // 2 - noise_band_hi)
            ))
            spectrum /= np.sqrt(noise_band_hi - noise_band_lo) # Normalize to 1V RMS regardless of width

            waveform = workshop.time_points_from_freq(spectrum, fs=fs_out, density=True)

            #waveform = gn.cos(fs_out, fs_out, 1, selected_center_frequency, 0, 0, 0) # Test single tone

            print(f"RMS of transmitted signal: {np.std(waveform):.3f} V")

            aout.push([waveform])
            # time.sleep(0.5) # offline UI testing
            transmitting_center_frequency = selected_center_frequency

        # Receive one buffer and process it
        status("Receiving")
        din = ad4080.rx()
        received_center_frequency = transmitting_center_frequency
        # time.sleep(3) # offline UI testing
        # din = np.random.rand(nfft) * 2 - 1 # offline UI testing

        # Scale to Volts
        din = din * ad4080.scale / 1e6 # scale is in uV

        # Remove DC component
        din -= np.average(din)

        print(f"RMS of received signal: {np.std(din):.3f} V")

        # If operating directly on data_in, the UI thread may display an intermediary unprocessed result
        # So we operate on din locally and only store the final processed waveform in data_in to be displayed
        data_in = din

        # Compute FFT
        code_fmt = gn.CodeFormat.TWOS_COMPLEMENT
        rfft_scale = gn.RfftScale.NATIVE
        fft_cplx = gn.rfft(np.array(din), navg, nfft, window, code_fmt, rfft_scale)
        fft_db = gn.db(fft_cplx) # FIXME: White noise offset depending on nfft

        print(f'{np.max(fft_db)=}') 

th = Thread(target=iio_thread)
th.start()

# Main thread does UI stuff. Everything that follows is just that.

# axw - AXes for received Waveform
# axf - AXes for received Fft
# axs - AXes for Slider widget
fig, (axw, axf, axs) = pl.subplots(3, 1, gridspec_kw={'height_ratios': [4, 5, 1]})

pl.pause(0.001)

axs.set_title('Transmit noise center frequency')
slider = matplotlib.widgets.Slider(axs, '', 0, plot_freq_range,
    valinit = selected_center_frequency, valstep = 1000, initcolor = 'none',
)
slider.on_changed(slider_changed)

fig.tight_layout()
fig.canvas.draw()

while pl.get_fignums(): # get_fignums will be falsey if window has been closed
    axw.clear()
    axw.set_title('Received waveform')
    axw.set_xlim(0, npts / fs_in)
    axw.set_ylim(-5, 5)
    axw.grid(True)
    axw.text(0, 4.5, f'Status: {status_msg}', horizontalalignment='left', verticalalignment='center')
    axw.plot(times, data_in)

    axf.clear()
    axf.set_title('Received FFT')
    axf.set_xlim(0, plot_freq_range)
    axf.set_ylim(-160, 20)
    axf.grid(True)

    for fold in range(5):
        freqs = freq_axis + fold * (fs_in // 2 + 1)
        sinc1 = np.sinc(freqs / fs_in)
        sinc1 = gn.db(np.complex128(sinc1)) # Convert to dB and adjust

        axf.plot(freqs, sinc1,
            'r--' if fold > 0 else 'r',
            alpha = 0.75 ** fold,
            label = f'Theoretical sinc1 response'
        )
        axf.plot(freqs, fft_db[::(-1)**fold],
            'b--' if fold > 0 else 'b',
            alpha = 0.75 ** fold
        )

        if fold == 0:
            axf.annotate('Actual FFT', xy=(fs_in // 4, -150), horizontalalignment='center')
        else:
            axf.annotate(f'"Unfolded" {fold}', xy=(fs_in * (2 * fold + 1) // 4, -150), horizontalalignment='center')

    # x tick at nyquist
    axf.axvline(fs_in//2, linestyle='--', color='k')

    if received_center_frequency is not None:
        fc = received_center_frequency

        # Draw arrow at generated frequency
        axf.annotate('Generated', xy=(fc, 0), xytext=(fc, 10), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='center')

        if fc > fs_in // 2:
            # Compute fa = aliased frequency
            fold = fc // (fs_in // 2)
            if fold % 2 == 0:
                fa = fc - (fs_in // 2) * fold
            else:
                fa = (fs_in // 2) * (fold + 1) - fc

            # Draw arrow at aliased frequency
            axf.annotate('Aliased', xy=(fa, 0), xytext=(fa, 10), arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='center')

    fig.canvas.draw_idle()
    pl.pause(0.001)

# Stop iio thread as well
running = False
th.join()
