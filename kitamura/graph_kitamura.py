import os
import numpy as np
import matplotlib.pyplot as plt
import re

cwd=os.getcwd()
path=f'{cwd}\\forgraph保存\\'
forgraph=re.compile(r'forgraph')
gradient=re.compile(r'gradient')
two=re.compile(r'2')
three=re.compile(r'3')
p90=re.compile(r'90.0')
p95=re.compile(r'95.0')
p99=re.compile(r'99.0')

for filename in os.listdir(path):
    if forgraph.search(filename):
        textfile=filename
        ylabel="音声識別可能時間率"

        if two.search(filename):
            labelname='閾値固定'
            marker='^'
            fillstyle='full'   
        elif three.search(filename):
            labelname='white'
            marker='x'
            fillstyle='none'
        elif p90.search(filename):
            labelname='p=90'
            marker='o'
            fillstyle='full'
        elif p95.search(filename):
            labelname='p=95'
            marker='o'
            fillstyle='none'
        elif p99.search(filename):
            labelname='p=99'
            marker='^'
            fillstyle='none'


        #ファイル読み込み、リストの取得
        z=open(path+textfile,'r',encoding='UTF-8')
        all_list=np.array(z.readlines())
        snr_list=np.array(all_list[0].split(' ')[:-1]).astype(float)
        mean_list=np.array(all_list[1].split(' ')[:-1]).astype(float)
        sd_list=np.array(all_list[2].split(' ')[:-1]).astype(float)

        fig,ax=plt.subplots(figsize=(9,5),dpi=300)
        if two.search(filename):
            ax.set_ylim([0.04,0.13])
        elif three.search(filename):
            ax.set_ylim([0.04,0.13])
        elif p95.search(filename):
            ax.set_ylim([0.04,0.13])

        ax.errorbar(snr_list,mean_list,yerr=sd_list,\
                    label=labelname,\
                    marker=marker,markersize=13,\
                    fillstyle=fillstyle,\
                    ecolor='black',color='black',\
                    capsize=8,linestyle='None',)
        
        ax.tick_params(direction='in',labelsize=25,length=8)
        ax.set_xlabel('SN比(dB)',fontname='Yu Gothic',fontsize=25)
        ax.set_ylabel(ylabel,fontname='Yu Gothic',fontsize=25)

        plt.tight_layout()
        pngfile=textfile.replace('.txt','.jpg').replace('forgraph_','')
        dirname='グラフ保存先/'
        os.makedirs(dirname,exist_ok=True)
        filename=dirname+f'{pngfile}'
        plt.savefig(filename)