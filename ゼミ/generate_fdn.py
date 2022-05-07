import time,argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read,write
from scipy import signal

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tfn',type=str)
    parser.add_argument('mfn1',type=str)
    parser.add_argument('snr1',type=float)
    args=parser.parse_args()
    return args

def generate_VN(Nd,fs,leng):
    # Nd: パルス密度 [pulses/sec]
    # fs: サンプリング周波数
    # leng: 生成するVelvet noiseの長さ[sec]
    out = np.zeros(int(np.round(fs*leng)))
    Td = fs / Nd # the average distance between impulses [sample]
    loop_size = int(np.floor(out.size / Td))
    for i in range(loop_size):
        pos =  int( (i-1) * Td + np.random.rand() * (Td-1) )
        out[pos] = 2 * np.round(np.random.rand()) - 1
    return out

def apply_Aweighting(wave, fs):
    def calc_r(f):
        r = 12194**2 * f**4 / ( \
        (f**2 + 20.6**2) * \
        np.sqrt(f**2 + 107.7**2) * \
        np.sqrt(f**2 + 737.9**2) * \
        (f**2 + 12194**2) )
        return r
    #rfftした時の周波数軸を作る
    freqs = np.arange(4097) * fs / (4096 * 2)
    #4097個分の振幅を入れる箱を作っておく
    r_amp = np.zeros(4097)
    #周波数一つ一つを見ていく
    for i, f in enumerate(freqs):
        #f=0の時は除外する
        if f > 0:
            #i番目の振幅変化量＝周波数fにおけるcalc_r(f)
            r_amp[i] = calc_r(f)
    #周波数が1000の時のrを基準とする
    r_amp /= calc_r(1000)
    #r_ampをirfft
    fil_a = np.fft.irfft(r_amp)
    #irfft結果の前半と後半を入れ替える
    fil_a = np.append(fil_a[4096:8192],fil_a[0:4096])
    wave_a = np.convolve(wave, fil_a)
    return wave_a[4096 : 4096 + wave.size]

def normalize_wave(x):
    if x.dtype == 'int16':
        x = x /np.abs((np.iinfo('int16')).min)
    elif target.dtype == 'int32':
        x = x / np.abs((np.iinfo('int32')).min)
    return x

def calc_RMSs(t1,t2,f1,f2,num,sr,x):
    sigs = np.zeros((num, x.size))

    # 周波数軸上の分割
    cfreqs = 10**np.linspace(np.log10(f1),np.log10(f2),num=num)
    for i in range(num):
        b, a = signal.gammatone(cfreqs[i], 'fir', fs=sr)
        zi = signal.lfilter_zi(b, a)
        sigs[i,:], _ = signal.lfilter(b, a, x, zi=zi*x[0])

    # 時間軸上の分割とT-FセルのRMS計算
    w_leng = int(np.round(sr * t1))
    w_shift = int(np.round(sr * t2))
    loop_size = int(np.floor(x.size / w_shift))-1
    ret = np.zeros((num, loop_size))
    for i in range(num):
        for j in range(loop_size):
            sig_cell = sigs[i,j*w_shift:j*w_shift+w_leng]*np.hanning(w_leng)
            ret[i,j] = np.sqrt(np.mean(sig_cell**2))
    return ret

def apply_itfs(TF,t1,t2,f1,f2,filter,sr,x):
    out = np.zeros(x.size)
    filter_size = np.shape(filter)
    cfreqs = 10**np.linspace(np.log10(f1),np.log10(f2),num=filter_size[0])
    w_leng = int(np.round(sr * t1))
    w_shift = int(np.round(sr * t2))
    loop_size = int(np.floor(x.size / w_shift))-1
    for i in range(filter_size[0]):
        b, a = signal.gammatone(cfreqs[i], 'fir', fs=sr)
        y = signal.filtfilt(b, a, x)
        for j in range(loop_size):
            if filter[i,j] == TF:
                out[j*w_shift:j*w_shift+w_leng] += \
                y[j*w_shift:j*w_shift+w_leng]*np.hanning(w_leng)
    return out

if __name__=='__main__':
    args = get_args()
    tfn = args.tfn
    mfn1 = args.mfn1
    snr1 = args.snr1

    t1 = 0.02
    t2 = 0.01
    f1 = 80
    f2 = 8000
    freq_reso = 128

    #vn = generate_VN(2400,48000,5)
    #write('vn.wav',48000,vn)

    sample_rate, target = read(tfn)
    target = normalize_wave(target)
    sample_rate, masker = read(mfn1)
    masker = normalize_wave(masker)

    if target.ndim > 1:
        target = target[:,0]
    if masker.ndim > 1:
        masker = masker[:,0]
    if target.size > masker.size:
        target = target[0:masker.size]
    else:
        masker = masker[0:target.size]

    # SNR調整
    laeq_t = 10 * np.log10(np.mean(apply_Aweighting(target,sample_rate) ** 2))
    print(laeq_t)
    laeq_m = 10 * np.log10(np.mean(apply_Aweighting(masker,sample_rate) ** 2))
    masker *= 10 ** -((snr1 - (laeq_t - laeq_m)) / 20)
    laeq_m = 10 * np.log10(np.mean(apply_Aweighting(masker,sample_rate) ** 2))
    print(laeq_m)

    print('calc. RMSs of target+masker...')
    RMSs_tm = calc_RMSs(t1,t2,f1,f2,freq_reso,sample_rate,target+masker)

    print('calc. RMSs of masker...')
    RMSs_masker = calc_RMSs(t1,t2,f1,f2,freq_reso,sample_rate,masker)

    print('generate fill-dip noise...')
    bin_filter = RMSs_tm > RMSs_masker * 1.01
    filter_shape = np.shape(bin_filter)
    fdn = np.zeros(target.size)
    array_for_calc_factor = np.zeros(target.size)
    w_leng = int(np.round(sample_rate * t1))
    w_shift = int(np.round(sample_rate * t2))

    cfreqs = 10**np.linspace(np.log10(f1),np.log10(f2),num=filter_shape[0])
    maskers_eacn_band = np.zeros((filter_shape[0], target.size))
    for i in range(filter_shape[0]):
        b, a = signal.gammatone(cfreqs[i], 'fir', fs=sample_rate)
        maskers_eacn_band[i,:] = signal.filtfilt(b, a, masker)

    filtered_RMSs = RMSs_tm * bin_filter
    for i in range(filter_shape[1]):
        RMSs_this_segment = filtered_RMSs[:,i]
        peaks,_ = signal.find_peaks(RMSs_this_segment, distance=freq_reso/16)

        if peaks.size>1:
            RMSs_fdn_interp = np.interp(range(filter_shape[0]),peaks,RMSs_this_segment[peaks],RMSs_this_segment[0],RMSs_this_segment[-1])
        else:
            RMSs_fdn_interp = np.zeros(filter_shape[0])

        RMSs_fdn_interp = (RMSs_fdn_interp - RMSs_tm[:,i])*((RMSs_fdn_interp - RMSs_tm[:,i])>0)
        plt.clf()
        plt.plot(RMSs_tm[:,i])
        plt.plot(RMSs_tm[:,i]+RMSs_fdn_interp)
        plt.ylim(0,0.1)
        plt.pause(0.1)

        for j in range(filter_shape[0]):
            masker_cell = maskers_eacn_band[j,i*w_shift:i*w_shift+w_leng]*np.hanning(w_leng)
            array_for_calc_factor[i*w_shift:i*w_shift+w_leng] += masker_cell
            masker_cell *= RMSs_fdn_interp[j] / np.sqrt(np.mean(masker_cell**2))
            fdn[i*w_shift:i*w_shift+w_leng] += masker_cell

    fdn *= np.sqrt(np.mean(np.abs(masker**2))) / np.sqrt(np.mean(np.abs(array_for_calc_factor**2)))

    norm_factor = np.amax(np.abs(target+masker+fdn))
    write('mix.wav',sample_rate,(target+masker)/norm_factor)
    write('fdn.wav',sample_rate,fdn/norm_factor)

    RMSs_total = calc_RMSs(t1,t2,f1,f2,freq_reso,sample_rate,target+masker+fdn)

    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.imshow(RMSs_tm,origin='lower')
    ax2.imshow(RMSs_total,origin='lower')
    fig.tight_layout()
    plt.show()
