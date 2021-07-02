import librosa
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np

blu = "blu.wav" #any audio files
hh = "hh.wav"
ro = "ro.wav"

blu_data, sr = librosa.load(blu)
hh,  sr = librosa.load(hh)
ro, sr = librosa.load(ro)

sampDur = 1/sr 
#print("sample Duration : ", round(sampDur,5))
signalDur = sampDur*len(blu_data)
#print(signalDur)

plt.figure(figsize=(15,17))
plt.subplot(3,1,1)
librosa.display.waveplot(blu_data)
plt.title("Blues")
plt.ylim((-1,1))



plt.subplot(3,1,2)
librosa.display.waveplot(hh)
plt.title("HH")
plt.ylim((-1,1))



plt.subplot(3,1,3)
librosa.display.waveplot(ro)
plt.title("Rock")
plt.ylim((-1,1))
#plt.show()

FRAME_SIZE = 1024
HOP_LENGTH = 512

def amp_env(signal, fr_si,hop_len):
	amplitude_env = []

	for i in range(0,len(signal), hop_len):
		curr_fr_en = max(signal[1:i+fr_si])
		amplitude_env.append(curr_fr_en)

	return np.array(amplitude_env)

def fan_amp_env(signal,fr_si,hop_len):
	return np.array([max(signal[i:i+fr_si]) for i in range(0,signal.size,hop_len)])


ae_blu = amp_env(blu_data, FRAME_SIZE, HOP_LENGTH)
#print(len(ae_blu))



aee_blu = fan_amp_env(blu_data,FRAME_SIZE,HOP_LENGTH)
#print(len(aee_blu))

ae_hh = fan_amp_env(hh,FRAME_SIZE,HOP_LENGTH)

ae_ro = fan_amp_env(ro,FRAME_SIZE,HOP_LENGTH)
frames = range(0,ae_blu.size - 1) 
t = librosa.frames_to_time(frames, hop_length = HOP_LENGTH) 

plt.figure(figsize=(15,17))
plt.subplot(3,1,1)
librosa.display.waveplot(blu_data)
plt.plot(t,ae_blu,color = 'r')
plt.title("Blues")
plt.ylim((-1,1))



plt.subplot(3,1,2)
librosa.display.waveplot(hh)
plt.plot(t,ae_hh,color = 'r')
plt.title("HH")
plt.ylim((-1,1))



plt.subplot(3,1,3)
librosa.display.waveplot(ro)
plt.plot(t,ae_ro,color = 'r')
plt.title("Rock")
plt.ylim((-1,1))
plt.show()