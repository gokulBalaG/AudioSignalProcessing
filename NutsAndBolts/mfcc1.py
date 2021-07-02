import librosa 
import librosa.display
import numpy as np 
import matplotlib.pyplot as plt 

blu_file = "blu.wav"

blu, sr = librosa.load(blu_file)

mfcc = librosa.feature.mfcc(blu, n_mfcc = 13, sr=sr)
 

plt.figure(figsize=(25,10))
librosa.display.specshow(mfcc, x_axis = "time", sr=sr)
plt.colorbar(format = "%+2f")
plt.show()

#delta and deltadelta
deltamfcc = librosa.feature.delta(mfcc)
delta2mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(25,10))
librosa.display.specshow(deltamfcc, x_axis = "time", sr=sr)
plt.colorbar(format = "%+2f")
plt.show()

plt.figure(figsize=(25,10))
librosa.display.specshow(delta2mfcc, x_axis = "time", sr=sr)
plt.colorbar(format = "%+2f")
plt.show()
