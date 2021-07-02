import librosa
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np
import IPython as ipd

blu_file = "blu.wav"
hh_file = "hh.wav"
ro_file = "ro.wav"

blu, _ = librosa.load(blu_file)
hh, _= librosa.load(hh_file)
ro, _ = librosa.load(ro_file)

FRAME_LENGTH = 1024
HOP_LENGTH = 512

rmsBlu = librosa.feature.rms(blu, frame_length = FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
rmsHh = librosa.feature.rms(hh, frame_length = FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
rmsRo = librosa.feature.rms(ro, frame_length = FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

print(rmsBlu.size)


frames = range(0,rmsBlu.size ) 
t = librosa.frames_to_time(frames, hop_length = HOP_LENGTH) 
print(t.size)

print(rmsHh.size)
print(rmsRo.size)

plt.figure(figsize=(15,17))


plt.subplot(3,1,1)
librosa.display.waveplot(blu)
plt.plot(t,rmsBlu,color = 'r')
plt.title("Blues")
plt.ylim((-1,1))



plt.subplot(3,1,2)
librosa.display.waveplot(hh)
plt.plot(t,rmsHh,color = 'r')
plt.title("HH")
plt.ylim((-1,1))



plt.subplot(3,1,3)
librosa.display.waveplot(ro)
plt.plot(t,rmsRo,color = 'r')
plt.title("Rock")
plt.ylim((-1,1))
plt.show()

#RMS from scratch

def rms(signal,fr_si,ho_si):
    rms_val = []

    for i in range(0,len(signal),ho_si):
        rms_cur = np.sqrt(np.sum(signal[i:i+fr_si]**2)/fr_si)
        rms_val.append(rms_cur)

    return np.array(rms_val)


rmsBlu1 = rms(blu, FRAME_LENGTH, HOP_LENGTH)
rmsHh1 = rms(hh,  FRAME_LENGTH, HOP_LENGTH)
rmsRo1 = rms(ro,FRAME_LENGTH, HOP_LENGTH)

print(rmsBlu1.size)
print(rmsHh1.size)
print(rmsRo1.size)



plt.figure(figsize=(15,17))


plt.subplot(3,1,1)
librosa.display.waveplot(blu)
plt.plot(t,rmsBlu,color = 'r')
plt.plot(t,rmsBlu1,color = 'y')
plt.title("Blues")
plt.ylim((-1,1))



''''plt.subplot(3,1,2)
librosa.display.waveplot(hh)
plt.plot(t,rmsHh1,color = 'y')
plt.title("HH")
plt.ylim((-1,1))'''



plt.subplot(3,1,3)
librosa.display.waveplot(ro)
plt.plot(t,rmsRo,color = 'r')
plt.plot(t,rmsRo1,color = 'y')
plt.title("Rock")
plt.ylim((-1,1))
plt.show()



zcrBlu = librosa.feature.zero_crossing_rate(blu, frame_length = FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
zcrHh = librosa.feature.zero_crossing_rate(hh, frame_length = FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
zcrRo = librosa.feature.zero_crossing_rate(ro, frame_length = FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

plt.figure(figsize=(15,17))
plt.plot(t,zcrBlu,color = 'r')
plt.plot(t,zcrHh,color = 'b')
plt.plot(t,zcrRo,color = 'y')
plt.ylim((0,1))#Normalized value
plt.show()

