#Analyzed with the audio files stored locally


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

#spectralCentroids
sc_blu = librosa.feature.spectral_centroid(y=blu, sr=sr,n_fft = FRAME_SIZE, hop_length = HOP_SIZE )[0]
sc_hh = librosa.feature.spectral_centroid(y=hh, sr=sr,n_fft = FRAME_SIZE, hop_length = HOP_SIZE )[0]

#visualize

frame = range(len(sc_blu))
t = librosa.frames_to_time(frame) 


plt.figure(figsize = (25,10))
plt.plot(t, sc_blu, color = 'b')
plt.plot(t, sc_hh, color = 'r')
plt.show()

bw_blu = librosa.feature.spectral_bandwidth(y=blu, sr=sr,n_fft = FRAME_SIZE, hop_length = HOP_SIZE )[0]
bw_hh = librosa.feature.spectral_bandwidth(y=hh, sr=sr,n_fft = FRAME_SIZE, hop_length = HOP_SIZE )[0]

plt.figure(figsize = (25,10))
plt.plot(t, bw_blu, color = 'b')
plt.plot(t, bw_hh, color = 'r')
plt.show()
