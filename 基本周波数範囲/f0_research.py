import numpy as np
from scipy.io.wavfile import read
import argparse
import os
import matplotlib.pyplot as plt

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("wavfile")
    parser.add_argument('f0_low',type=float)
    parser.add_argument('f0_high',type=float)
    args=parser.parse_args()
    return args


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

args=get_args()
wavfile=args.wavfile
sr,s=read(wavfile)

t,shs_array,f0_array=calc_f0_shs(sr,s)

plt.plot(f0_array)
plt.show()