import numpy as np
import matplotlib.pyplot as plt
import re
import os

pathin='F:\\卒論プログラム３\\forgraph保存\\'
forgraph=re.compile(r'forgraph')
shs=re.compile(r'shs')
fovar=re.compile(r'fovar')
total=re.compile(r'total')
two=re.compile(r'2-3')
three=re.compile(r'3-3')
p95=re.compile(r'95.05.0')

for filename1 in os.listdir(pathin):
    if p95.search(filename1):
        f=open(pathin+filename1,'r',encoding='UTF-8')
        all_list1=np.array(f.readlines())
        snr_list1=np.array(all_list1[0].split(' ')[:-1]).astype(float)-0.5
        mean_list1=np.array(all_list1[1].split(' ')[:-1]).astype(float)
        sd_list1=np.array(all_list1[2].split(' ')[:-1]).astype(float)
        if shs.search(filename1):
            ylabel='調波構造検出可能時間率±SD'
            name1='shs'
        elif fovar.search(filename1):
            ylabel='基本周波数連続時間率±SD'
            name1='fovar'
        elif total.search(filename1):
            ylabel='周波数特性総合時間率±SD'
            name1='total'
        fig,ax=plt.subplots(figsize=(7.5,4))
        ax.errorbar(snr_list1,mean_list1,yerr=sd_list1,\
                    label='7-12',\
                    marker='o',markersize=8,\
                    fillstyle='none',\
                    ecolor='black',color='black',\
                    capsize=5,linestyle='None')
    
        for filename2 in os.listdir(pathin):
            if two.search(filename2):
                textfile2=filename2
                g=open(pathin+textfile2,'r',encoding='UTF-8')
                all_list2=np.array(g.readlines())
                snr_list2=np.array(all_list2[0].split(' ')[:-1]).astype(float)+0.5
                mean_list2=np.array(all_list2[1].split(' ')[:-1]).astype(float)
                sd_list2=np.array(all_list2[2].split(' ')[:-1]).astype(float)

                label2='19-24'
                marker2='^'
                fillstyle2='full'
                name2='1_2'

                ax.errorbar(snr_list2,mean_list2,yerr=sd_list2,\
                            label=label2,\
                            marker=marker2,markersize=8,\
                            fillstyle=fillstyle2,\
                            ecolor='black',color='black',\
                            capsize=5,linestyle='None')
                ax.tick_params(direction='in',labelsize=10,length=5)
                ax.set_xlabel('SN比(dB)',fontname='Yu Gothic',fontsize=13)
                ax.set_ylabel(ylabel,fontname='Yu Gothic',fontsize=13)
                plt.legend()
                pathout='F:\\卒論プログラム３\\グラフ保存先\\'
                newfile=f'{name1}+{name2}'
                plt.savefig(pathout+newfile)


            elif three.search(filename2):
                g=open(pathin+filename2,'r',encoding='UTF-8')
                all_list2=np.array(g.readlines())
                snr_list2=np.array(all_list2[0].split(' ')[:-1]).astype(float)+0.5
                mean_list2=np.array(all_list2[1].split(' ')[:-1]).astype(float)
                sd_list2=np.array(all_list2[2].split(' ')[:-1]).astype(float)

                label2='25-30'
                marker2='x'
                fillstyle2='none'
                name2='1_3'

                ax.errorbar(snr_list2,mean_list2,yerr=sd_list2,\
                            label=label2,\
                            marker=marker2,markersize=8,\
                            fillstyle=fillstyle2,\
                            ecolor='black',color='black',\
                            capsize=5,linestyle='None')
                ax.tick_params(direction='in',labelsize=10,length=5)
                ax.set_xlabel('SN比(dB)',fontname='Yu Gothic',fontsize=13)
                ax.set_ylabel(ylabel,fontname='Yu Gothic',fontsize=13)
                plt.legend()
                pathout='F:\\卒論プログラム３\\グラフ保存先\\'
                newfile=f'{name1}+{name2}'
                plt.savefig(pathout+newfile)
            else:
                continue
    else:
        continue
