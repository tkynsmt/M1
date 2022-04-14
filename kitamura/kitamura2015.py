from scipy.io.wavfile import read, write
from PyOctaveBand import octavefilter
import argparse
import numpy as np

# 引数の取得
# args.f_noise: noiseのwaveファイル
# args.f_recorded: 測定点で録音した音のwaveファイル
# args.percentile: 閾値とするパーセンタイル値
parser = argparse.ArgumentParser(description="Calculates the ranged of possible speech identification")
parser.add_argument("f_noise", help = "N.wav")
parser.add_argument("f_recorded", help = "N.wav")
parser.add_argument("percentile", help = "95")
args = parser.parse_args()

# noiseの読み込み
# sr_n: noiseのサンプリング周波数，p_n: noiseの波形データ
sr_n, p_n = read(args.f_noise)

# バンドパスフィルタリング（PyOctaveBand.pyが必要）
# spl_n_list: noiseの1/3オクターブバンドレベル
# cf: 中心周波数
# p_n_b_list: バンドパスフィルタリングされたnoise（リスト）
# octavefilterで中心周波数ごとの音圧レベル(スペクトル)の対応リストと、バンド毎の時間ｰ音圧レベルのリストを作る。
#len(p_n_b_list):バンド数  len('リスト'):リストの数
spl_n_list, cf, p_n_b_list = octavefilter(p_n, sr_n, fraction=3, order=6, limits=[90,11000], show=0, sigbands =1)
# --------------------------------------------------------------------------------------------------------------------
# 25msごとのLeqを算出

# wleng: 25msの窓の長さ（サンプル）
# loop_size: noise全体を分析するのに必要な窓の数
# leq_array: 時間-周波数平面上のleqのデータ
# サンプリング周波数に0．025をかけることで時間窓1個あたりのサンプル数を求める(整数で定義) なぜ整数？
wleng = int(sr_n * 0.025)
# .sizeでデータの要素数を求め、データの要素数を時間窓1つあたりの要素数で割ることでnoise全体を分析するのに必要な窓数(loop_size)を求める(1つの値)
loop_size = int( np.floor( p_n.size / wleng ) )
# (p_n_b_listのバンド数:周波数)行(時間窓数:時間)列の行列を作る
leq_array = np.zeros( [len( p_n_b_list ), loop_size] )

#時間窓ごとの等価騒音レベルに1つずつ番号をつけるため、最初は0番としておく
idx = 0
#p_n_b_listの中のデータを一つ一つ最初から見ていく
for p_n_b in p_n_b_list :
    #ノイズ全体の時間窓を一つ一つ最初から見ていく
    for i in range(loop_size):
        #p_n_b_listの中の音圧データから時間窓1つ当たりのサンプル数分だけ順番に抽出する
        sample = p_n_b[ i * wleng : i * wleng + wleng ]
        #上で定義したサンプルデータで音圧二乗平均を求め、それによってLeq（等価騒音レベル?)とする　なぜ対数？
        leq = 10 * np.log10( np.sum( sample * sample ) / wleng )
        #時間窓ごとの等価騒音レベルに0から番号をつけ、零行列を埋めていく
        leq_array[ idx, i] = leq
    #1つの配列を作り終えたため、次の配列を作るための番号を作る
    idx += 1

print(type(sample))

# 連続する5つの時間窓のLeqの傾きの絶対値を求める
# slope_array: 時間-周波数平面上の傾きの絶対値データ
#(p_n_b_listのバンド数)行(時間窓数-4)列の零行列を作っておく
slope_array = np.zeros( [len( p_n_b_list ), loop_size - 4] )
#p_n_b_listのバンドごとに計算する
for i in range(len( p_n_b_list )) :
    #時間窓数-4の中で
    for j in range(loop_size - 4) :
        #y=時間-周波数Leqリストのi番目のバンド、j‐j+5番目までの音圧データ
        y = leq_array[i, j : j + 5]
        #x=0.1.2.3.4に、窓の長さをサンプリング周波数で割ったもの(=0.025)をかける
        x = np.arange(5) * wleng / sr_n
        #xの平均値を出す
        mean_x = np.sum(x) / x.size
        #yの平均値を出す
        mean_y = np.sum(y) / y.size
        #xの分散を求める
        v_x = np.sum(np.power(x - mean_x, 2)) / x.size
        #x,yの共分散を求める
        v_xy = np.sum((x - mean_x) * (y - mean_y)) / x.size
        #共分散をxの分散で割って回帰直線の傾きを求める
        slope = v_xy / v_x
        #i番目のバンドのj番目の回帰直線の傾きの絶対値を求める
        slope_array[i, j] = np.abs(slope)

# 各帯域の傾きの閾値を求める
# threshold_array: 各帯域の傾きの閾値
threshold_array = np.zeros(len( p_n_b_list ))
#p_n_b_listの長さ分について
for i in range(len( p_n_b_list )) :
    #回帰直線の傾きの中から、入力した数値に相当するパーセンタイル値を閾値として算出する
    threshold = np.percentile(slope_array[i, 0:], float(args.percentile))
    #i番目のバンドごとの閾値の表を作る
    threshold_array[i] = threshold


#閾値リストを表示
print("thresholds for 21 bands: {}".format(threshold_array))
# (MEMO) noiseのslope_arrayはもう使用しないので，recordedのデータで上書きしても良い

# recordedの読み込み
# sr_r: recordedのサンプリング周波数
# p_r: recordedの波形データ
sr_r, p_r = read(args.f_recorded)

# バンドパスフィルタリング（PyOctaveBand.pyが必要）
# spl_r_list: recordedの1/3オクターブバンドレベル
# cf: 中心周波数
# p_r_b_list: バンドパスフィルタリングされたrecorded（リスト）
spl_r_list, cf, p_r_b_list = octavefilter(p_r, sr_r, fraction=3, order=6, limits=[90,11000], show=0, sigbands =1)

# 25msごとのLeqを算出
# wleng: 25msの窓の長さ（サンプル）
# loop_size: recorded全体を分析するのに必要な窓の数
# leq_array: 時間-周波数平面上のleqのデータ
wleng = int(sr_r * 0.025)
loop_size = int( np.floor( p_r.size / wleng ) )
leq_array = np.zeros( [len( p_r_b_list ), loop_size] )
idx = 0
for p_r_b in p_r_b_list :
    for i in range(loop_size):
        sample = p_r_b[ i * wleng : i * wleng + wleng ]
        leq = 10 * np.log10( np.sum( sample * sample ) / wleng )
        leq_array[ idx, i] = leq
    idx += 1

# 連続する5つの時間窓のLeqの傾きを求める
# slope_array: 時間-周波数平面上の傾きのデータ
slope_array = np.zeros( [len( p_r_b_list ), loop_size - 4] )
for i in range(len( p_r_b_list )) :
    for j in range(loop_size - 4) :
        y = leq_array[i, j : j + 5]
        x = np.arange(5) * wleng / sr_r
        mean_x = np.sum(x) / x.size
        mean_y = np.sum(y) / y.size
        v_x = np.sum(np.power(x - mean_x, 2)) / x.size
        v_xy = np.sum((x - mean_x) * (y - mean_y)) / x.size
        slope = v_xy / v_x
        slope_array[i, j] = np.abs(slope)

# scoreの計算

#バンド数分のスコアを表示する箱を作る
score_array = np.zeros(len( p_r_b_list ))
#バンドごとでスコアを計算する
for i in range(len( p_r_b_list )) :
    #スコアの合計点を最初に0点とし、その後順に足していく
    score = 0
    #0or1を順に決めていく
    for j in range(loop_size - 4) :
        #回帰直線の傾きが閾値よりも大きければ
        if threshold_array[i] < slope_array[i, j] :
            #+1点する
            score += 1
    #i番目のバンドについて、合計スコアを窓関数-4で割る        
    score_array[i] = score / (loop_size - 4)
#バンドごとの音声識別可能時間率を表示    
print("ratios for 21 bands: {}".format(score_array))
#上の21個のデータを平均して音声識別可能時間率を求める
print("ratio: {}".format(np.mean( score_array )))


