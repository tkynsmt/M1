import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
if __name__=='__main__':
    #引数を解決
    sr,clean_amp=read('F://卒論プログラム３//自分の声.wav')
    if (clean_amp.ndim > 1):
        clean_amp = clean_amp[:,1] #音源が2ch以上の場合は1chのデータにする
    if (clean_amp.size % 2 != 0):
        clean_amp = np.append(clean_amp,[0]) #音源の長さを２の倍数にする

    clean_amp=clean_amp/32768
    t=np.arange(0.0,len(clean_amp)/sr,1/sr)
    start=int(sr*1.5)
    cuttime=0.04
    stop=int(start+sr*cuttime)
    wavdata=clean_amp[start:stop]
    hanningdata=wavdata*np.hanning(len(wavdata))
    n=4096
    dft=np.fft.fft(hanningdata,n)
    Adft=np.abs(dft)
    Pdft=np.abs(dft)**2
    fscale=np.fft.fftfreq(n,d=1.0/sr)
    Adftlog=20*np.log10(Adft)
    cps=np.real(np.fft.ifft(Adftlog))
    quefrency=np.linspace(0,cuttime,num=int(n/2))

    plt.plot(quefrency,(cps[0:int(n/2)]),color='orchid')
    plt.xlim(0,0.025)
    plt.ylim(-3,3)
    plt.title('騒音付加音声のケプストラム',fontname='MS Gothic',fontsize=20)
    plt.xlabel('ケフレンシー（s)',fontname='MS Gothic',fontsize=20)
    plt.ylabel('ケプストラム',fontname='MS Gothic',fontsize=20)
    plt.tight_layout()
    plt.savefig('騒音付加音声ケプストラム.png')
    plt.show()
    

