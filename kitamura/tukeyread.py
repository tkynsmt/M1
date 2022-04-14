from concurrent.futures import process
import numpy as np
from scipy.io.wavfile import read
import os 
from csv import writer,reader

a=np.array([[1,2,3,4,5][6,7,8,9,10]])
f=open("test.csv","w",encoding="UTF-8")
csv_writer=writer(f)
for i in a:
    csv_writer.writerow(i)
f.close()

g=open("test.csv","r",encoding="UTF-8")
csv_reader=reader(g)
for i in csv_reader:
    print(i)
import csv
  
word = "Good morning, Hello, Good night"
words = word.split(',')

with open('start.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(words)