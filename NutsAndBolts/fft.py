import numpy as np 
import matplotlib.pyplot as plt 
import os
import librosa 

bluFile = "blu.wav"
hhFile = "hh.wav"
roFile = "ro.wav"

blu_data,sr = librosa.load(bluFile)
hh_data,sr = librosa.load(bluFile)
ro_data,sr = librosa.load(roFile)

#print(blu_data.shape)

bluFFt = np.fft.fft(blu_data)
#print(bluFFt[0])
magSpec = np.abs(bluFFt)
#print(magSpec[0])

def dis_mag_spec(signal,title,sr,freq_rat):
    ft = np.fft.fft(signal)

    mag_spec = np.abs(ft)
    plt.figure(figsize=(18,5))

    freq = np.linspace(0,sr,len(mag_spec))
    num_freq_sins = int(len(freq)*freq_rat)
    plt.plot(freq[:num_freq_sins], mag_spec[:num_freq_sins])
    plt.xlabel("frequency")
    plt.title(title)
    plt.show()

dis_mag_spec(blu_data,"blue", sr,0.5)