import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#loading the file
blu_file = "blu.wav"
hh_file = "hh.wav"
ro_file = "ro.wav"

#loadig with lobrosa(with sr = 23500)
blu, sr = librosa.load(blu_file)
hh, _ = librosa.load(hh_file)
ro, _ = librosa.load(ro_file)

#stft
FRAME_SIZE = 2048
HOP_SIZE = 512

s_blu = librosa.stft(blu, n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
s_hh = librosa.stft(hh, n_fft = FRAME_SIZE, hop_length = HOP_SIZE)
s_ro = librosa.stft(ro, n_fft = FRAME_SIZE, hop_length = HOP_SIZE)

y_blu = np.abs(s_blu)**2
y_hh = np.abs(s_hh)**2
s_ro = np.abs(s_ro)**2

def plot_spectrogram(y, sr, hop_len,y_axis):
	plt.figure(figsize=(15,10))
	librosa.display.specshow(y,sr = sr, hop_length = hop_len, x_axis = "time", y_axis = y_axis)
	plt.colorbar(format="\+2.f")

y_log_blu = librosa.power_to_db(y_blu)
plot_spectrogram(y_log_blu, sr, HOP_SIZE,  "log")