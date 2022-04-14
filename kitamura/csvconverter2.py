import sys
import csv
path='F:\\卒論プログラム３\\tukey\\'
txtfile=sys.argv[1]
csvfile=txtfile.replace('.txt','.csv')
with open(path+txtfile,newline='') as fin,open(path+csvfile,mode='w',newline='') as fout :

    reader=csv.reader(fin,delimiter=' ',skipinitialspace=True)
    writer=csv.writer(fout)

    writer.writerows(reader)
        