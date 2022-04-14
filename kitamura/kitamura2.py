
#パーセンタイル値は予め50回試行して算出したものを用いる

import argparse
import numpy as np
from scipy.io.wavfile import read
from PyOctaveBand import octavefilter
import os
cwd=os.getcwd()

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('cleanfile',type=str)
    parser.add_argument('number_of_simulation',type=int,help='シミュレーション回数')
    parser.add_argument('percentile',type=float)
    args=parser.parse_args()
    return args

#rms:root-mean-square:二乗平均平方根　音圧平均を求める
def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp),axis=-1))

#任意のSN比となるように音源の音圧平均から騒音の音圧平均を求める
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

def Leq(t_list,data,wleng):
    #窓関数の数
    loop_size=int(np.floor(len(data)/wleng))
    #Leqを代入する箱
    leq_array=np.zeros([len(t_list),loop_size])
    idx=0
    for i in t_list:
        for j in range(loop_size):
            sample=i[j*wleng:j*wleng+wleng]
            leq=10*np.log10(np.mean(sample*sample))
            leq_array[idx,j]=leq
        idx+=1
    return leq_array

def Gradient(leq_array,data,wleng,sr):
    loop_size=int(np.floor(len(data)/wleng)) #窓関数の数
    gradient_array=np.zeros([len(leq_array),loop_size-4]) #代入用零行列の生成
    x=np.array(range(5))/(wleng/sr)
    x_mean=np.mean(x)
    x_var=np.sum((x-x_mean)**2)/5 #xの分散
    idx=0
    for i in leq_array:
        for j in range(loop_size-4):
            y=i[j:j+5]
            y_mean=np.mean(y)
            xy_var=np.sum((x-x_mean)*(y-y_mean))/5 #xyの共分散
            gradient=np.abs(xy_var/x_var) #傾きは絶対値を取る
            gradient_array[idx,j]=gradient
        idx+=1
    return gradient_array
    
            
                        

if __name__=='__main__':
    
    #引数を解決
    args=get_args()
    cleanfile=args.cleanfile
    percentile=args.percentile
    number_of_simulation=args.number_of_simulation
    
    #読み込み
    sr,s=read(cleanfile)
    if (s.ndim>1):
        s=s[:,1]
    if (s.size % 2 !=0):
        s=np.append(s,[0])

    #sにA特性で重みづけし、音圧二乗平均を求める
    sA=apply_Aweighting(s,sr)
    sA_rms=cal_rms(sA)


    '''
    A特性について整理
    まず、もと音源のSとNがそれぞれある。
    その次に、SとNそれぞれにA特性をかける（これが普段聞いている音と似た音になる）。
    A特性がかかったSとNの音圧の2乗の平均の比に対して対数を取って10を掛ける事でSN比（㏈）が求まる。

    '''    
    #1つのSN比での音声識別可能時間率の平均値と標準偏差を記録する箱
    total_time_ratio_mean=np.zeros(6)
    total_time_ratio_sd=np.zeros(6)
    SNR_box=np.zeros(6)

    
    #Rcmdr用のファイル
    f=open(f"{cwd}\\recordファイル\\record_gradient_2_p={str(percentile)}.txt","a",encoding="UTF-8")
    f.write("total_time_ratio SNR\n")
    
    #SN比の指定
    for a in range(6):
        SNR=(-5)*(a+1)
        SNR_box[a]=SNR
        nA_rms=cal_adjusted_noise_rms(sA_rms,SNR) #SNRから騒音の音圧2乗平均を求める
    

#-----------------------閾値を先に計算する-----------------------------------------------------------------------------------------------------------------------


        #1つのSN比につきnのパーセンタイル値を50回求める
        threshold_21x50=np.zeros([21,50])
        for i in range(50):

            #nの作成
            n1=generate_pink_noise(len(s))    #n1の作成
            n1A=apply_Aweighting(n1,sr)    #n1にA特性の重みづけ
            n1A_rms=cal_rms(n1A)    #n1Aのrmsを求める
            n=n1*(nA_rms/n1A_rms)    #nの算出
            mix=s+n
            nmax=np.max(np.abs(n))
            mixmax=np.max(np.abs(mix))
            n=32767*n/np.max([nmax,mixmax]) #正規化

            #nをバンドパスフィルタに通す>t_n_listの作成
            f_n_list,cf,t_n_list=octavefilter(n,sr,fraction=3,order=6,limits=[90,11000],show=0,sigbands=1)

            wleng=int(sr*0.025)
            #t_n_listからバンド毎、時間窓毎にleqを求める>n_leqの作成
            
            n_leq=Leq(t_n_list,n,wleng)

            #バンド毎で5つの時間窓のleqから回帰直線の傾きを求める>n_gradientの作成
            n_gradient=Gradient(n_leq,n,wleng,sr)

            #n_gradientからバンド毎でパーセンタイル値を作成
            n_threshold=np.zeros(len(n_gradient))
            idx=0
            for one_array in n_gradient:
                threshold_21x50[idx,i]=np.percentile(one_array,percentile)
                idx+=1
            

        #バンド毎でパーセンタイル値の平均値を算出
        threshold_mean=np.zeros(len(n_threshold))   #あるSN比でのバンド毎の閾値の平均値
        idx=0
        for threshold_50 in threshold_21x50:
            threshold_mean[idx]=np.mean(threshold_50)
            idx+=1


#-------------------------ここから比較に入る---------------------------------------------------------------------------------------------


        #number_of_simulationの数だけ時間率を計算
        total_time_ratio_record=np.zeros(number_of_simulation)
        for nth_time in range(number_of_simulation):

            #同じサンプル数のノイズを作る
            n1=generate_pink_noise(len(s))

            #n1にA特性を掛ける
            n1A=apply_Aweighting(n1,sr)

            #n1Aの音圧2乗平均を求める
            n1A_rms=cal_rms(n1A)

            #SNRに合う騒音の作成
            n=n1*(nA_rms/n1A_rms)
            
            #騒音付加音声
            mix=s+n

            #正規化
            nmax=np.max(np.abs(n))
            mixmax=np.max(np.abs(mix))
            n=32767*n/np.max([nmax,mixmax])
            mix=32767*mix/np.max([nmax,mixmax])

            #基準化した音源をバンドパスフィルタに通す
            f_mix_list,cf,t_mix_list=octavefilter(x=mix,fs=sr,fraction=3,order=6,limits=[90,11000],show=False,sigbands=True)

            #窓関数1つあたりのデータ数
            wleng=int(0.025*sr)

            #Leqの計算
            mix_leq=Leq(t_mix_list,mix,wleng)

            #回帰直線の計算
            mix_gradient=Gradient(mix_leq,mix,wleng,sr)

            
            
            #各帯域で閾値とmix_gradientの比較,点数加算
            time_ratio_box=np.zeros(len(mix_gradient))
            idx=0
            for one_array in mix_gradient: #バンドごとで比較
                score=0
                for one_gradient in one_array: #mixの回帰直線の傾きとnの傾きの閾値を比較
                    if one_gradient>threshold_mean[idx]: 
                        score+=1 #傾きが大きければ加点
                time_ratio_box[idx]=score/len(mix_gradient[0]) #合計得点を計算回数で割った時間率を記録
                idx+=1
            
            #各バンドの時間率の平均を算出（音声識別可能時間率を算出）
            total_time_ratio=np.mean(time_ratio_box)
            #音声識別可能時間率をひとつずつ書き出し
            f.write(f"{str(total_time_ratio)} {str(SNR)}\n")
            #1つのSN比で算出した音声識別可能時間率を全て記録
            total_time_ratio_record[nth_time]=total_time_ratio
        
        f.close

        #1つのSN比での音声識別可能時間率の平均値、標準偏差を計算し記録　図の作成に用いる
        total_time_ratio_mean[a]=np.mean(total_time_ratio_record)
        total_time_ratio_sd[a]=np.std(total_time_ratio_record)
        
    #グラフ用ファイルの作成
    g=open(f"{cwd}\\forgraph保存\\forgraph_gradient_2_p={percentile}.txt","a",encoding="UTF-8")
    for SNR in SNR_box:
        g.write(f"{str(SNR)} ")
    g.write("\n")
    for mean in total_time_ratio_mean:
        g.write(f"{str(mean)} ")
    g.write("\n")
    for sd in total_time_ratio_sd:
        g.write(f"{str(sd)} ")
    g.close()
    