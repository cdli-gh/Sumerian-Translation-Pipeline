import pandas as pd
import re
import glob
import os
import codecs
import numpy as np
import json


# the path to your csv file directory
mycsvdir = 'Dataset/ETCSL_conll/'

# get all the conll files in that directory (assuming they have the extension .conll)
conllfiles = glob.glob(os.path.join(mycsvdir, '*.conll'))

# loop through the files and read them in with codes UTF-8
n=1
dataframes = {} # a list to hold all the individual pandas DataFrames
ID=[]
FORM=[]
SEGM=[]
XPOSTAG=[]

for conllfile in conllfiles:
    with codecs.open(conllfile, 'r', 'utf-8') as CDLICoNLLFile:
        print(n)
        n+=1
        inputLines = list()
        for line in CDLICoNLLFile:
            line = line.strip()
            if(len(line)==0):
                continue
            if line[0] != '#':
                line = line.split("\t")
                a=""
                b=""
                c=""
                d=""
                a=line[0]
                b=line[1].replace('[', '').replace(']', '').replace('<', '').replace('>','').replace('!', '').replace('?', '').replace('@c','').replace('@t','').replace('_','').replace(',','').replace('\\','').replace('/','').replace('c','sz').replace('C','SZ').replace('j','g')
                c=line[2]+"["+line[4]+"]"
                d=line[3].split(':')[-1].replace('C','CNJ').replace('PD','DET').replace('I','J').replace('ADP','').replace('AUX','V').replace('NEG','').replace('ideophone','N').replace('cardinal','NU').replace('fraction','NU').replace('ordinal','NU').replace('demonstrative','DET').replace('indefinite','DET').replace('interrogative','DET').replace('nominal-relative','DET').replace('personal','DET').replace('reflexive','DET').replace('_','O').replace('X','O')
                print(a+"\t"+b+"\t"+c+"\t"+d)
                ID.append(a)
                FORM.append(b)
                SEGM.append(c)
                XPOSTAG.append(d)
            else:
                print(line)
    print("\n")
    
    
    
    
    
dataframes['ID']=ID
dataframes['FORM']=FORM
dataframes['SEGM']=SEGM
dataframes['XPOSTAG']=XPOSTAG
df = pd.DataFrame(dataframes)

df.drop(df[df['XPOSTAG']==""].index, inplace = True)
df = df.dropna()
df = df.reset_index(drop=True)
df.to_csv('Dataset/ETCSL_RAW_NER_POS.csv',index=False)



