import librosa
import librosa.display
import numpy as np 
import matplotlib.pyplot as plt 

blu_file = "blu.wav"
hh_file = "hh.wav"

blu, sr = librosa.load(blu_file)
hh, _ = librosa.load(hh_file)

FRAME_SIZE = 2048
HOP_SIZE = 512

bluSpec = librosa.stft(blu, n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
hhSpec = librosa.stft(hh, n_fft = FRAME_SIZE, hop_length = HOP_SIZE)

bluSpec_trans = bluSpec.T

def calculte_split_freqbin(spectrogram, splitFre, sample_fre):
	frequency_range = sample_fre/2
	frequency_delta_perbin = frequency_range/spectrogram.shape[0]
	split_freqbin = np.floor(splitFre / frequency_delta_perbin)
	return int(split_freqbin)

split_freqbin = calculte_split_freqbin(bluSpec, 2080, 22050)
#print(split_freqbin)

def calculate_ber(spectrogram, splitFre, sample_ra):
	split_freqbin = calculte_split_freqbin(spectrogram, splitFre, 22050)

	#move to power spec
	power_spec = np.abs(spectrogram)**2
	power_spec = power_spec.T 

	band_energy_ratio = []

	for frequencies in power_spec:
		sum_power_lfreq = np.sum(frequencies[1:split_freqbin])
		sum_power_hfreq =np.sum(frequencies[split_freqbin:])
		ber_cur_frame = sum_power_lfreq / sum_power_hfreq
		band_energy_ratio.append(ber_cur_frame)

	return np.array(band_energy_ratio)

ber_blu = calculate_ber(bluSpec, 2000, sr)
#print(ber_blu.shape)

ber_hh = calculate_ber(hhSpec, 2000, sr)

#visualizing BER

frame = range(len(ber_blu))
t = librosa.frames_to_time(frame, hop_length = HOP_SIZE) 
plt.figure(figsize = (25,10))
plt.plot(t, ber_blu, color = 'b')
plt.plot(t, ber_hh, color = 'r')
plt.show()