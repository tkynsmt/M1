import argparse
import numpy as np
from scipy.io.wavfile import read

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('cleanfile',type=str)
    parser.add_argument('f0_low',type=float)
    parser.add_argument('f0_high',type=float)
    parser.add_argument('number_of_simulation',type=int,help='シミュレーション回数')
    parser.add_argument('low_percentile',type=float)
    parser.add_argument('high_percentile',type=float)
    args=parser.parse_args()
    return args

#rms:root-mean-square:二乗平均平方根
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
    
def calc_f0_shs(sr, p):
    #サンプリング周波数をf0_lowで割る
    qrf_low = int(np.round(1 / float(args.f0_low) * sr))
    #サンプリング周波数をf0_highで割る
    qrf_high = int(np.round(1 / float(args.f0_high) * sr))
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
        #c_ampのqrf_high～qrf_lowサンプルの中の最大値を平均値で割る
        shs = np.max( np.abs( c_amp[ qrf_high : qrf_low ] ) ) / np.median( np.abs( c_amp[ qrf_high : qrf_low ] ) )
        #c_ampの中の最も大きいサンプルのサンプル数を取り出し、サンプリング周波数で割ることで基本の音を求める
        f0 = 1 / ( ( np.argmax( np.abs( c_amp[ qrf_high : qrf_low ] ) ) + qrf_high ) / sr )
        shs_array[k] = shs
        f0_array[k] = f0
    #窓関数をかける回数分のarrayにshift一つあたりのサンプル数をかけサンプリング周波数で割って時間軸を作っておく
    t = np.arange(loop_size) * shift / sr
    return t,shs_array, f0_array

def cal_f0_continuity(f0_array):
    continuity_var_box=np.zeros(np.size(f0_array)-4)
    for i in range(np.size(f0_array)-4):
        five_window_var=np.var(f0_array[i:i+5])
        continuity_var_box[i]=five_window_var
    return continuity_var_box    

if __name__=='__main__':
    #引数を解決
    args=get_args()
    cleanfile=args.cleanfile
    low=args.low_percentile
    high=args.high_percentile
    number_of_simulation=args.number_of_simulation
    
    #recordfileの作成　Rcmdrで多重比較、エクセルで分散分析表の作成
    f=open(f'record_shs_1-3_p={low}{high}.txt','a',encoding='UTF-8')
    f.write('shs_ratio '+'snr'+'\n')
    g=open(f'record_fovar_1-3_p={low}{high}.txt','a',encoding='UTF-8')
    g.write('fovar_ratio '+'snr'+'\n')
    h=open(f'record_total_1-3_p={low}{high}.txt','a',encoding='UTF-8')
    h.write('total_ratio '+'snr'+'\n')


    #音源を読み込む
    sample_rate,clean_amp=read(cleanfile)
    if (clean_amp.ndim > 1):
        clean_amp = clean_amp[:,1] #音源が2ch以上の場合は1chのデータにする
    if (clean_amp.size % 2 != 0):
        clean_amp = np.append(clean_amp,[0]) #音源の長さを２の倍数にする

    #音源にA特性補正をかけた時間波形を求める
    clean_amp_A = apply_Aweighting(clean_amp, sample_rate)

    #clean音源の振幅のrms(二乗平均平方根)を求める
    clean_rms_A=cal_rms(clean_amp_A)

    #ノイズの長さを決める
    noise_length=clean_amp.size

    snr_box=np.zeros(6)
    shs_ratio_mean_box=np.zeros(6)
    shs_ratio_sd_box=np.zeros(6)
    fovar_ratio_mean_box=np.zeros(6)
    fovar_ratio_sd_box=np.zeros(6)
    total_ratio_mean_box=np.zeros(6)
    total_ratio_sd_box=np.zeros(6)
    for a in range(6):
        #sn比の決定
        snr=(-5)*(a+1)
        snr_box[a]=snr

        #指標を試行回数分だけ計算
        shs_box=np.zeros(number_of_simulation)
        fovar_box=np.zeros(number_of_simulation)
        total_box=np.zeros(number_of_simulation)
        for i in range(total_box.size):
            #ピンクノイズの生成
            noise_amp = generate_pink_noise(noise_length)

            #ノイズにA特性補正をかけた時間波形を求める
            noise_amp_A = apply_Aweighting(noise_amp, sample_rate)

            #作成したノイズをSN比に合わせた倍率を掛ける
            adjusted_noise_rms=cal_adjusted_noise_rms(clean_rms_A,snr)
            noise_rms_A=cal_rms(noise_amp_A)
            adjusted_noise_amp=noise_amp*(adjusted_noise_rms/noise_rms_A)

            #Sと補正したNを足し合わせる
            mixed_amp=clean_amp+adjusted_noise_amp

            #最大振幅が32767になるよう基準化 書き出しの時に必要？
            m1 = np.max(np.abs(adjusted_noise_amp))
            m2 = np.max(np.abs(mixed_amp))
            adjusted_noise_amp = 32767 * adjusted_noise_amp / np.max([m1,m2])
            mixed_amp = 32767 * mixed_amp / np.max([m1,m2])

            #shsの計算
            t, shs_array, f0_array = calc_f0_shs(sample_rate,mixed_amp)
            t_n, shs_array_n, f0_array_n = calc_f0_shs(sample_rate,adjusted_noise_amp)

            #基本周波数分散の計算
            f0_var_array=cal_f0_continuity(f0_array)
            f0_var_array_n=cal_f0_continuity(f0_array_n)
            

            #得点計算
            #調波構造検出可能時間率の計算
            n_shs_percentile=np.percentile(shs_array_n,high)
            score1=0
            for j in range(np.size(shs_array)):
                if shs_array[j]>n_shs_percentile:
                    score1+=1
            harmonic_structure_ratio=score1/shs_array.size #調波構造検出可能時間率の算出
            shs_box[i]=harmonic_structure_ratio
            f.write(str(harmonic_structure_ratio)+' '+str(snr)+'\n')

            #基本周波数連続時間率の計算
            n_var_percentile=np.percentile(f0_var_array_n,low)
            score2=0
            for j in range(np.size(f0_var_array)):
                if f0_var_array[j]<n_var_percentile:
                    score2+=1
            f0_continuity_ratio=score2/np.size(f0_var_array) #基本周波数連続時間率の算出
            fovar_box[i]=f0_continuity_ratio
            g.write(str(f0_continuity_ratio)+' '+str(snr)+'\n')

            #周波数特性総合時間率の計算
            total_ratio=(harmonic_structure_ratio+f0_continuity_ratio)/2
            total_box[i]=total_ratio
            h.write(str(total_ratio)+' '+str(snr)+'\n')

        shs_ratio_mean_box[a]=np.mean(shs_box)
        shs_ratio_sd_box[a]=np.std(shs_box)
        fovar_ratio_mean_box[a]=np.mean(fovar_box)
        fovar_ratio_sd_box[a]=np.std(fovar_box)
        total_ratio_mean_box[a]=np.mean(total_box)
        total_ratio_sd_box[a]=np.std(total_box)
    f.close()
    g.close()    
    h.close()
    
    #グラフ作成用ファイル
    e=open(f'forgraph_shs_1-3_p={high}{low}.txt','a',encoding='UTF-8')
    for a in range(len(snr_box)):
        e.write(str(snr_box[a])+' ')
    e.write('\n')
    for a in range(len(shs_ratio_mean_box)):
        e.write(str(shs_ratio_mean_box[a])+' ')
    e.write('\n')
    for a in range(len(shs_ratio_sd_box)):
        e.write(str(shs_ratio_sd_box[a])+' ')
    e.close()

    e=open(f'forgraph_fovar_1-3_p={high}{low}.txt','a',encoding='UTF-8')
    for a in range(len(snr_box)):
        e.write(str(snr_box[a])+' ')
    e.write('\n')
    for a in range(len(fovar_ratio_mean_box)):
        e.write(str(fovar_ratio_mean_box[a])+' ')
    e.write('\n')
    for a in range(len(fovar_ratio_sd_box)):
        e.write(str(fovar_ratio_sd_box[a])+' ')
    e.close()
    
    e=open(f'forgraph_total_1-3_p={high}{low}.txt','a',encoding='UTF-8')
    for a in range(len(snr_box)):
        e.write(str(snr_box[a])+' ')
    e.write('\n')
    for a in range(len(total_ratio_mean_box)):
        e.write(str(total_ratio_mean_box[a])+' ')
    e.write('\n')
    for a in range(len(total_ratio_sd_box)):
        e.write(str(total_ratio_sd_box[a])+' ')
    e.close()
