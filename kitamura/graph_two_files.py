import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

parser=argparse.ArgumentParser()
parser.add_argument('textfile1')
parser.add_argument('textfile2')
args=parser.parse_args()

textfile1=args.textfile1
textfile2=args.textfile2

fovar=re.compile(r'fovar')
shs=re.compile(r'shs')
total=re.compile(r'total')
two=re.compile(r'2-3')
three=re.compile(r'3-3')

f=open(textfile1,'r',encoding='UTF-8')
all_list1=np.array(f.readlines())
snr_list1=np.array(all_list1[0].split(' ')[:-1]).astype(float)-0.5
mean_list1=np.array(all_list1[1].split(' ')[:-1]).astype(float)
sd_list1=np.array(all_list1[2].split(' ')[:-1]).astype(float)
if fovar.search(textfile1):
    ylabel='基本周波数連続時間率±SD'
    name1='fovar'
elif shs.search(textfile1):
    ylabel='調波構造検出可能時間率±SD'
    name1='shs'
elif total.search(textfile1):
    ylabel='周波数特性総合時間率±SD'
    name1='total'

g=open(textfile2,'r',encoding='UTF-8')
all_list2=np.array(g.readlines())
snr_list2=np.array(all_list2[0].split(' ')[:-1]).astype(float)+0.5
mean_list2=np.array(all_list2[1].split(' ')[:-1]).astype(float)
sd_list2=np.array(all_list2[2].split(' ')[:-1]).astype(float)
if two.search(textfile2):
    label2='19-24'
    marker2='^'
    name2='1_2'
    
elif three.search(textfile2):
    label2='25-30'
    marker2='x'
    name2='1_3'


fig,ax=plt.subplots(figsize=(7.5,4))
ax.errorbar(snr_list1,mean_list1,yerr=sd_list1,\
            label='7-12',\
            marker='o',markersize=8,\
            fillstyle='none',\
            ecolor='black',color='black',\
            capsize=5,linestyle='None')
ax.errorbar(snr_list2,mean_list2,yerr=sd_list2,\
            label=label2,\
            marker=marker2,markersize=8,\
            fillstyle='full',\
            ecolor='black',color='black',\
            capsize=5,linestyle='None')
ax.tick_params(direction='in',labelsize=10,length=5)
ax.set_xlabel('SN比(dB)',fontname='Yu Mincho',fontsize=13)
ax.set_ylabel(ylabel,fontname='Yu Mincho',fontsize=13)
plt.legend()
pathout='F:\\卒論プログラム３\\グラフ保存先\\'
filename=f'{name1}{name2}.png'
plt.savefig(pathout+filename)
plt.show()



