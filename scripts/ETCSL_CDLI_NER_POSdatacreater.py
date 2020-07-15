import pandas as pd


def process_training(data):
    c=0
    for i in range(len(data)):
        if int(data['ID'][i])==1:
            c+=1
            data['ID'][i]=c
        else:
            data['ID'][i]=c
    return data


def Unique_sentences(tagged_sentence):
    mat = [tuple(t) for t in tagged_sentence]
    print("Number of phrases in Original Data ",len(mat))
    matset = set(mat)
    print("Number of Unique phrase ",len(matset))
    unique_tagged_sentence=[tuple(t) for t in matset]
    return  unique_tagged_sentence



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



df=pd.read_csv('Dataset/ETCSL_RAW_NER_POS.csv')

df=process_training(df)
tagged_sentence=Preparing_tagged_data(df)
print("Training data processed \n")

tagged_sentence_uniq=Unique_sentences(tagged_sentence)
tagged_data=creating_uniqe_df(tagged_sentence_uniq)
li=[tup for sent in tagged_data for tup in sent]
df=pd.DataFrame(li)
df=df.rename(columns={0:'ID',1:'FORM',2:'XPOSTAG'})

df_old=pd.read_csv('Dataset/Augmented_RAW_NER_POS.csv')

# total phrases in df_old are 25478
n=25479
c=1
for i in range(len(df_old)):
    if(df_old['ID'][i]==c):
        df_old['ID'][i]=n
    else:
        c+=1
        n+=1
        df_old['ID'][i]=n
        
df_new=pd.concat([df, df_old], ignore_index=True, sort=False)
l1=pd.read_csv('Named Entities Sumerian ORACC.csv',header=None)
l2=pd.read_csv('Part of Speech (POS) tags Sumerian ORACC.csv',header=None)
l1=list(l1[0])
l2=list(l2[0])

df1=df_new.copy(deep=True)

for i in range(len(df1)):
    k=df1['XPOSTAG'][i]
    POS="O"
    print(k)
    if k in l2:
        POS=k
    if k in l1:
        POS="NE"
    df1['XPOSTAG'][i]=POS
    
df1.to_csv('Dataset/ETCSL_ORACC_POS.csv',index=False)

df2=df_new.copy(deep=True)
for i in range(len(df2)):
    k=df2['XPOSTAG'][i]
    if k in l1:
        df2['XPOSTAG'][i]=k
    else:
        df2['XPOSTAG'][i]="O"
df2.to_csv('Dataset/ETCSL_ORACC_NER.csv',index=False)

df =pd.read_csv('Dataset/ETCSL_ORACC_POS.csv')
df=df.dropna()
df = df.reset_index(drop=True)
df.to_csv('Dataset/ETCSL_ORACC_POS.csv',index=False)

df =pd.read_csv('Dataset/ETCSL_ORACC_NER.csv')
df=df.dropna()
df = df.reset_index(drop=True)
df.to_csv('Dataset/ETCSL_ORACC_NER.csv',index=False)
