import pandas as pd
import re
import glob
import os
import codecs
import json


# the path to your csv file directory
mycsvdir = 'Dataset/Raw/to_dict'

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
                b=line[1]
                c=line[2]
                d=line[3]
                print(a+"\t"+b+"\t"+c+"\t"+d)
                ID.append(a)
                FORM.append(b)
                SEGM.append(c)
                XPOSTAG.append(d)
            else:
                print(line)
    print("\n")
    
#Creating dataframe

dataframes['ID']=ID
dataframes['FORM']=FORM
dataframes['SEGM']=SEGM
dataframes['XPOSTAG']=XPOSTAG
df = pd.DataFrame(dataframes)
    

#for i in range(len(df)):
#    l=df["ID"][i].split(".")
#    if (len(l)<3):
#        print(l)
#        print(i)
#['_']
#7
#['_']
#14565
#removed index 7 and corrected index 14565 by inputing r.1.4 manually     
   
#df.to_csv('Dataset/Raw_NER_POS_data.csv',index=False)
    
    
    
    