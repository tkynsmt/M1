import numpy as np
import argparse 
from scipy.io.wavfile import read

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('cleanfile',type=str)
    parser.add_argument('f0_low',type=float)
    parser.add_argument('f0_high',type=float)
    parser.add_argument('number_of_simulation',type=int,help='シミュレーション回数')
    parser.add_argument('Cep_degree',type=float)
    args=parser.parse_args()
    return args

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp),axis=-1))

def cal_adjusted_noise_rms(clean_rms,snr):
    anr=clean_rms/(10**(float(snr)/20))
    return anr

def generate_pink_noise(noise_length):
    wn = np.random.normal(size=noise_length)
    wn_f = np.fft.rfft(wn) #ホワイトノイズのフーリエスペクトル
    amp = np.abs(wn_f) #振幅スペクトル（-3dB/Octにする）
    phase = np.angle(wn_f) #位相スペクトル（保持する）
    for f in range(amp.size):
        if f > 0 : # f = 0は対数を取れないのでスキップ
            amp[f] *= 10 ** ( ( -3 * np.log2( f / 2 ) ) / 20 ) #振幅スペクトルを-3dB/Oct
    pn_f = amp * np.exp( 1j * phase ) #振幅スペクトルと位相スペクトルからフーリエスペクトルを作成
    return np.fft.irfft( pn_f ) #時間軸に戻す

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
    
def cal_fmax(sr, p):
    #サンプリング周波数をf0_lowで割る
    Cep_degree=args.Cep_degree
    #サンプリンぐ周波数を時間窓の半分の長さに分けたのを作る
    shift = int(sr * 0.025 / 2)
    #時間窓一つのサンプルを定義
    wleng = shift * 2
    #窓関数をかける回数を決める
    loop_size = int( np.floor( p.size / shift ) ) - 1
    #shs,f0の記録箱を作る
    shs_array = np.zeros(loop_size) #shs: Strength of Harmonic Structure
    f0_array = np.zeros(loop_size)

    #窓を1つずつ見ていく
    for k in range(loop_size):
        #窓一つに対してハニング窓をかける
        sample = p[ k * shift : k * shift + wleng ] * np.hanning(wleng)
        #fft,ifftをかけてケフレンシーを求める
        c_amp = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(sample)))))
        #ケプストラムについて次数を指定する
        c_amp[Cep_degree:]=0
        #c_ampの中の最も大きいサンプルのサンプル数を取り出し、サンプリング周波数で割ることで基本の音を求める
        f0 = 1 / ( ( np.argmax( np.abs( c_amp[ qrf_high : qrf_low ] ) ) + qrf_high ) / sr )
        shs_array[k] = shs
        f0_array[k] = f0
    #窓関数をかける回数分のarrayにshift一つあたりのサンプル数をかけサンプリング周波数で割って時間軸を作っておく
    t = np.arange(loop_size) * shift / sr
    return t,shs_array, f0_array


if __name__ =="__main__":
    args=get_args()
    cleanfile=args.cleanfile
    Cep_degree=args.Cep_degree