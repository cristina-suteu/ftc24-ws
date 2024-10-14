import libm2k
import adi
from sys import exit
import matplotlib.pyplot as pl
import numpy as np
import genalyzer.advanced as gn
from matplotlib.patches import Rectangle as MPRect

fs_in  = 100000
fs_out = 750000

# Normally, npts would be computed as navg * nfft
npts = 16384 # AD4080 carrier firmware max limit
navg = 2  # No. of fft averages
# Get number of points
nfft = npts // navg


m2k = libm2k.m2kOpen('usb:')
if m2k is None:
	print("Connection Error: No ADALM2000 device available/connected to your PC.")
	exit(1)

aout = m2k.getAnalogOut()

aout.reset()
m2k.calibrateDAC()
aout.setSampleRate(0, fs_out)
aout.enableChannel(0, True)

ad4080 = adi.ad4080('serial:/dev/ttyACM0,230400,8n1')
if ad4080 is None:
	print("Connection Error: No AD4080 device available/connected to your PC.")
	exit(1)


phase = 0.0  # Tone phase
td = 0.0
tj = 0.0
code_fmt = gn.CodeFormat.TWOS_COMPLEMENT  # ADC codes format
rfft_scale = gn.RfftScale.NATIVE  # FFT scale
window = gn.Window.BLACKMAN_HARRIS  # FFT window
fsr = 2.0  # Full-scale range
fund_freq = 10000.0  # Hz
# This is a list of the amplitudes (in dBfs) of the fundamental (first element)
# and harmonics. You can add more harmonics to the list, but we'll start
# out with just the 2nd, 3rd, and 4th.
# Replace -200.0 with greater values to add harmonics
harm_dbfs = [-3.0, -23.0, -20.0, -20.0]

# These are lists of the frequencies (Hz) and amplitudes (in dBfs) of
# interfering tones or "noise tones". Genalyzer will interpret them as not
# harmonically related and add them to the total noise.
noise_freqs = [15000.0, 25000.0, 35000.0, 45000.0]
# Replace -200.0 with greater values to add noise tones
noise_dbfs = [-40, -50, -70, -60]
# Calculate absolute amplitudes from dBfs.
harm_ampl = []
for x in range(len(harm_dbfs)):
    harm_ampl.append((fsr / 2) * 10 ** (harm_dbfs[x] / 20))
noise_ampl = []
for x in range(len(noise_dbfs)):
    noise_ampl.append((fsr / 2) * 10 ** (noise_dbfs[x] / 20))
# Single side bin "width" around peaks
# If these values are too low, spectral leakage near signal peaks will be mistakenly detected as noise
ssb_fund = 1000 # Single side bins for fundamental and harmonics
ssb_rest = 1000 # Single side bins for DC and WorstOther

# If we are not windowing then choose the closest coherent bin for fundamental
if gn.Window.NO_WINDOW == window:
    fund_freq = gn.coherent(nfft, fs_out, fund_freq)
    ssb_fund = 0
    ssb_rest = 0

# Build up the signal from the fundamental, harmonics, and noise tones
awf = np.zeros(npts)

for harmonic in range(len(harm_dbfs)):
    freq = fund_freq * (harmonic + 1)
    print(f"Frequency: {freq} ({harm_dbfs[harmonic]} dBfs)")

    awf += gn.cos(npts, fs_out, harm_ampl[harmonic], freq, phase, td, tj)

for tone in range(len(noise_freqs)):
    freq = noise_freqs[tone]
    if gn.Window.NO_WINDOW == window:
        freq = gn.coherent(nfft, fs_out, noise_freqs[tone])
    print(f"Noise Frequency: {freq} ({noise_dbfs[tone]} dBfs)")
    awf += gn.cos(npts, fs_out, noise_ampl[tone], freq, phase, td, tj)

# push awf to M2K DAC
aout.setCyclic(True)
aout.push([awf])

# Config RX
ad4080.rx_buffer_size = npts
ad4080.sample_rate = fs_in

data_in = ad4080.rx()

# Remove DC component
data_in = data_in - np.average(data_in)

# Compute FFT
fft_cplx = gn.rfft(np.array(data_in),navg, nfft, window, code_fmt, rfft_scale)
# Compute frequency axis
freq_axis = gn.freq_axis(nfft, gn.FreqAxisType.REAL, fs_in)
# Compute FFT in db
fft_db = gn.db(fft_cplx)

# Fourier analysis configuration
key = 'fa'
gn.mgr_remove(key)
gn.fa_create(key)
gn.fa_analysis_band(key, "fdata*0.0", "fdata*1.0")
gn.fa_fixed_tone(key, 'A', gn.FaCompTag.SIGNAL, fund_freq, ssb_fund)
gn.fa_hd(key, 4)
gn.fa_ssb(key, gn.FaSsb.DEFAULT, ssb_rest)
gn.fa_ssb(key, gn.FaSsb.DC, -1)
gn.fa_ssb(key, gn.FaSsb.SIGNAL, -1)
gn.fa_ssb(key, gn.FaSsb.WO, -1)
gn.fa_fsample(key, fs_in)
print(gn.fa_preview(key, False))

# Fourier analysis results
fft_results = gn.fft_analysis(key, fft_cplx, nfft)
# compute THD
thd = 20 * np.log10(fft_results['thd_rss'] / harm_ampl[0])

print("\nFourier Analysis Results:\n")
print("\nFrequency, Phase and Amplitude for Harmonics:\n")
for k in ['A:freq', 'A:mag_dbfs', 'A:phase',
          '2A:freq', '2A:mag_dbfs', '2A:phase',
          '3A:freq', '3A:mag_dbfs', '3A:phase',
          '4A:freq', '4A:mag_dbfs', '4A:phase']:
    print("{:20s}{:20.6f}".format(k, fft_results[k]))
print("\nFrequency, Phase and Amplitude for Noise:\n")
for k in ['wo:freq','wo:mag_dbfs', 'wo:phase']:
    print("{:20s}{:20.6f}".format(k, fft_results[k]))
print("\nSNR and THD \n")
for k in ['snr', 'fsnr']:
    print("{:20s}{:20.6f}".format(k, fft_results[k]))
print("{:20s}{:20.6f}".format("thd", thd))

# Plot FFT
pl.figure(1)
fftax = pl.subplot2grid((1, 1), (0, 0), rowspan=2, colspan=2)
pl.title("FFT")
pl.plot(freq_axis, fft_db)
pl.grid(True)
#pl.xlim(freq_axis[0], 20)
pl.ylim(-160.0, 20.0)
annots = gn.fa_annotations(fft_results)

for x, y, label in annots["labels"]:
    pl.annotate(label, xy=(x, y), ha='center', va='bottom')
for box in annots["tone_boxes"]:
    fftax.add_patch(MPRect((box[0], box[1]), box[2], box[3],
                           ec='pink', fc='pink', fill=True, hatch='x'))

pl.tight_layout()
pl.show()
