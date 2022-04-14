from re import A
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

sr,p=read("自分の声.wav")
wleng=int(sr*0.025)
sample=p[50*wleng:51*wleng]*np.hanning(wleng)
samplere=np.max(np.real(np.fft.fft(sample)))
sampleim=np.max(np.imag(np.fft.fft(sample)))
sampleabs=np.max(np.abs(np.fft.fft(sample)))
plt.plot(sample)
plt.show()
samplere/sampleabs
sampleim/sampleabs
samplere/sampleim
