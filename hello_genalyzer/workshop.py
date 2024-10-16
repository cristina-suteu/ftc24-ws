import matplotlib.pyplot as pl
import genalyzer.advanced as gn
from matplotlib.patches import Rectangle as MPRect

def time_points_from_freq(freq, fs=1, density=False):
	"""Generate time series from half-spectrum.

	Parameters
	----------
	freq : [float]
		Half-spectrum of signal to be generated. DC offset in zeroth element.
	fs : float
		Sampling frequency. Used if `density == True` to scale the resulting
		waveform.
	density : bool
		If true, scales the resulting waveform by $N \sqrt{fs / N}$, where $N$ is
		the half-spectrum size.

	Returns
	-------
	res : np.array
		Resulting waveform with specified spectrum. Its length is twice the length
		of the `freq` parameter.
	"""
	
    N = len(freq)
    
    # Random phases for each frequency component, expect for DC which gets 0 phase
    rnd_ph_pos = (np.ones(N-1, dtype=np.complex)*
                  np.exp(1j*np.random.uniform(0.0, 2.0*np.pi, N-1)))
    rnd_ph_neg = np.flip(np.conjugate(rnd_ph_pos))
    rnd_ph_full = np.concatenate(([1],rnd_ph_pos,[1], rnd_ph_neg))

    r_spectrum_full = np.concatenate((freq, np.roll(np.flip(freq), 1)))
    r_spectrum_rnd_ph = r_spectrum_full * rnd_ph_full
    
    r_time_full = np.fft.ifft(r_spectrum_rnd_ph) # This line does the "real work"

    # Sanity check: is the imaginary component close to nothing?
    rms_imag = np.std(np.imag(r_time_full))
    rms_real = np.std(np.real(r_time_full))
    assert(rms_imag < rms_real * 1e-6, "RMS imaginary component should be close to zero")

    if density:
        r_time_full *= N * np.sqrt(fs / N) # Note that this N is "predivided" by 2
    
    return np.real(r_time_full)

def fourier_analysis(
	waveform,
	sampling_rate,
	fundamental,
	ssb_fund = 100,
	ssb_rest = 100,
	navg = 2,
	nfft = len(waveform) // navg,
	window = gn.Window.BLACKMAN_HARRIS,
	code_fmt = gn.CodeFormat.TWOS_COMPLEMENT,
	rfft_scale = gn.RfftScale.NATIVE
):
	"""Do fixed tone fourier analysis using genalyzer and plot results.

	Parameters
	----------
	waveform : np.array
	    Received waveform.
	sampling_rate : int
	    Sampling rate of received waveform.
	fundamental : float
		Fundamental frequency expected in the received waveform.
	ssb_fund : int
		Number of Fourier analysis single side bins for the fundamental and its
		harmonics. If this value is too low, spectral leakage around signal peaks
		will be labeled as noise.
	ssb_rest : int
		Number of Fourier analysis single side bins for other components: DC,
		WorstOther.
	navg : int
		Number of FFT windows to be averaged.
	nfft : int
		Length in samples of an FFT window. The total length of the waveform MUST
		be equal to `navg * nfft`.
	window : genalyzer.advanced.Window
		Windowing used for FFT.
	code_fmt : genalyzer.advanced.CodeFormat
		Code format of received samples.
	rfft_scale : genalyzer.advanced.RfftScale
		Real FFT scale for analysis.

	Returns
	-------
	None
	"""
	assert(len(waveform) == navg * nfft)

	# Remove DC component
	waveform = waveform - np.average(waveform)

	# Compute FFT
	fft_cplx = gn.rfft(np.array(waveform), navg, nfft, window, code_fmt, rfft_scale)

	# Compute frequency axis
	freq_axis = gn.freq_axis(nfft, gn.FreqAxisType.REAL, sampling_rate)

	# Compute FFT in db
	fft_db = gn.db(fft_cplx)

	# Fourier analysis configuration
	key = 'fa'
	gn.mgr_remove(key)
	gn.fa_create(key)
	gn.fa_analysis_band(key, "fdata*0.0", "fdata*1.0")
	gn.fa_fixed_tone(key, 'A', gn.FaCompTag.SIGNAL, fundamental, ssb_fund)
	gn.fa_hd(key, 4)
	gn.fa_ssb(key, gn.FaSsb.DEFAULT, ssb_rest)
	gn.fa_ssb(key, gn.FaSsb.DC, -1)
	gn.fa_ssb(key, gn.FaSsb.SIGNAL, -1)
	gn.fa_ssb(key, gn.FaSsb.WO, -1)
	gn.fa_fsample(key, sampling_rate)
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


	pl.figure(2)
	pl.plot(np.array(waveform))

	pl.tight_layout()
	pl.show()
	pl.show()
