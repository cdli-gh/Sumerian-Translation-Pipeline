import pandas as pd
import re
import glob
import os
import codecs
import json
import pprint, time
import tqdm



l1=pd.read_csv('Named Entities Sumerian ORACC.csv',header=None)
l2=pd.read_csv('Part of Speech (POS) tags Sumerian ORACC.csv',header=None)
l1=list(l1[0])
l2=list(l2[0])
Entities=l1+l2
print("Named and POS Entities for the Sumerian Language")
print(Entities)
print()
#Entities=['DN','EN','GN','MN','ON','PN','RN','SN','TN','WN','AN','CN','FN','LN','QN','YN','AJ','AV','NU','CNJ','DET','J','N','PP','V','IP','DP','MOD','PRP','QP','RP','REL','SBJ','XP']




def processing(df):
    index=[]
    for i in range(len(df)):
        l=df["ID"][i].split(".")
        ind=int(l[-1])
        index.append(ind)
    df=df[['FORM','XPOSTAG']]
    df.insert(0,"ID",index)
    
    for i in range(len(df)):
        k=df['XPOSTAG'][i].split(".")
    #    print(k)
        POS="O"
        for j in k:
            if j in Entities:
                POS=j
        df['XPOSTAG'][i]=POS
    
    
    return df



def pndict_processing(pndict):
    l=[]
    l1=[]
    for i in range(len(pndict)):
        s=pndict['NER'][i].split("|")
        pndict['Text'][i]=pndict['Text'][i].strip().replace("\'","")
        if len(s)>2:
            continue
        if (len(s)==1):
            l.append(pndict['Text'][i])
            l1.append(s[0])
        else:
            l.append(pndict['Text'][i])
            l1.append(s[0])
            l.append(pndict['Text'][i])
            l1.append(s[1])
    d={'Text':l,"NER":l1}
    data=pd.DataFrame(d)
    return data


def preparing_DICT(data):
    DICT={'PN':[],
      'DN':[],
      'FN':[],
      'GN':[],
      'RN':[],
      'TN':[],
      'WN':[]
      }
    for i in range(len(data)):
        DICT[data['NER'][i]].append(data['Text'][i])
        
    print(DICT.keys())
    return DICT




def text_augmenting(df_cpy,DICT):
    p=0
    d=0
    f=0
    g=0
    r=0
    t=0
    w=0
    c=1
    for i in range(len(df_cpy)):
        c+=1
        if (p==len(DICT['PN']) and d==len(DICT['DN']) and f==len(DICT['FN']) and g==len(DICT['GN']) and r==len(DICT['WN']) and t==len(DICT['TN']) and w==len(DICT['WN'])):
            break
        if df_cpy['XPOSTAG'][i]=='PN':
            if(p<len(DICT['PN'])):
                df_cpy['FORM'][i]=DICT['PN'][p]
                p+=1
        elif df_cpy['XPOSTAG'][i]=='DN':
            if(d<len(DICT['DN'])):
                df_cpy['FORM'][i]=DICT['DN'][d]
                d+=1
        elif df_cpy['XPOSTAG'][i]=='FN':
            if(f<len(DICT['FN'])):
                df_cpy['FORM'][i]=DICT['FN'][f]
                f+=1
        elif df_cpy['XPOSTAG'][i]=='GN':
            if(g<len(DICT['GN'])):
                df_cpy['FORM'][i]=DICT['GN'][g]
                g+=1
        elif df_cpy['XPOSTAG'][i]=='RN':
            if(r<len(DICT['RN'])):
                df_cpy['FORM'][i]=DICT['RN'][r]
                r+=1
        elif df_cpy['XPOSTAG'][i]=='TN':
            if(t<len(DICT['TN'])):
                df_cpy['FORM'][i]=DICT['TN'][t]
                t+=1
        elif df_cpy['XPOSTAG'][i]=='WN':
            if(w<len(DICT['WN'])):
                df_cpy['FORM'][i]=DICT['WN'][w]
                w+=1

    print("Total number of phrases",c)
    return df_cpy


def process_training(data):
    c=0
    for i in range(len(data)):
        if int(data['ID'][i])==1:
            c+=1
            data['ID'][i]=c
        else:
            data['ID'][i]=c
    return data

def Preparing_tagged_data(df):
    tagged_sentence_string=[]
    tagged_sentence=[]
    c=1
    temp=[]
    for i in range(len(df)):
        if df['ID'][i]==c:
            temp.append((df['FORM'][i],df['XPOSTAG'][i]))
        else:
            tagged_sentence.append(temp)
            temp=[]
            temp.append((df['FORM'][i],df['XPOSTAG'][i]))
            c+=1
    tagged_sentence.append(temp)
    return tagged_sentence



def Unique_sentences(tagged_sentence):
    mat = [tuple(t) for t in tagged_sentence]
    print("Number of phrases in Original Data ",len(mat))
    matset = set(mat)
    print("Number of Unique phrase ",len(matset))
    unique_tagged_sentence=[tuple(t) for t in matset]
    return  unique_tagged_sentence


def creating_uniqe_df(tagged_sentence_uniq):
    c=1
    tagged_data=[]
    for sent in tagged_sentence_uniq:
        tagged_data_temp=[]
        for l in sent:
            l=list(l)
            l.insert(0,c)
            l=tuple(l)
            tagged_data_temp.append(l)
        tagged_data.append(tagged_data_temp)
        c+=1
        
    return tagged_data


def Creat_POS_data(df):
    for i in range(len(df)):
        k=df['XPOSTAG'][i].split(".")
        POS="O"
    #    print(k)
        for j in k:
            if j in l2:
                POS=j
            if j in l1:
                POS="NE"
        df['XPOSTAG'][i]=POS
    
    return df 

def CheckNULL(df):
    for i in range(len(df)):
        if df["FORM"][i]=="":
            df["FORM"][i]="..."
            df["XPOSTAG"][i]="O"
    return df






def main():
    df=pd.read_csv('TextAugmentation/Raw/Raw_NER_POS_data.csv')
    df=processing(df)
    print("Dataset Before Augmentation, Human annotated data")
    print(df.groupby('XPOSTAG').count())
    print()
    
    
    print("Preparing Dictionary Data to be used")
    print("\n")
    pndict=pd.read_csv('TextAugmentation/pndictioanry_processed.csv',usecols = ['Text','NER'])
    data=pndict_processing(pndict)
    print(data.groupby('NER').count())
    print("\n")
    DICT=preparing_DICT(data)
    
    df_cpy=df.copy()
    df_cpy=df_cpy.append(df_cpy)
    df_cpy=df_cpy.append(df_cpy)
    df_cpy=df_cpy.append(df_cpy)
    df_cpy=df_cpy.append(df_cpy)
    df_cpy=df_cpy.append(df_cpy)
    df_cpy=df_cpy.append(df_cpy)
    df_cpy=df_cpy.reset_index(drop=True)
    
    print("Applying Text Augmentation \n")
    start = time.time()
    
    df_cpy=text_augmenting(df_cpy, DICT)
    df=df.append(df_cpy)
    df=df.reset_index(drop=True)
    
    end = time.time()
    difference = end-start
    
    print("Time taken in seconds: ", difference)    
    print("\n")
    
    print("Processing training Data")
    df=process_training(df)
    tagged_sentence=Preparing_tagged_data(df)
    print("Training data processed \n")
    
    
    tagged_sentence_uniq=Unique_sentences(tagged_sentence)
    tagged_data=creating_uniqe_df(tagged_sentence_uniq)
    li=[tup for sent in tagged_data for tup in sent]
    df=pd.DataFrame(li)
    df=df.rename(columns={0:'ID',1:'FORM',2:'XPOSTAG'})
    df=CheckNULL(df)
    
    
    print("\n")
    print("Dataset After Augmentation")      
    print(df.groupby('XPOSTAG').count())
    print("\n")
    
    print("Saving Augmented RAW NER POS data in Dataset/Augmented_RAW_NER_POS.csv .......")
    df.to_csv('Dataset/Augmented_RAW_NER_POS.csv',index=False)
    print("\n")    
    
    print("Creating and Saving Augmented POS TAG training data in Dataset/Augmented_POSTAG_training_ml.csv .......")
    df=Creat_POS_data(df)
    print("POS Data description")
    print(df.groupby('XPOSTAG').count())
    df.to_csv('Dataset/Augmented_POSTAG_training_ml.csv',index=False)
    
    
    
if __name__=='__main__':
    main()
    
