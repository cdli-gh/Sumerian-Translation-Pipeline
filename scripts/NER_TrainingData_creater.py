import pandas as pd
Input='Dataset/Augmented_RAW_NER_POS.csv'
Output='Dataset/Augmented_NER_training_ml.csv'

df=pd.read_csv(Input)
print(df.head())

l1=pd.read_csv('Named Entities Sumerian ORACC.csv',header=None)
l2=pd.read_csv('Part of Speech (POS) tags Sumerian ORACC.csv',header=None)
l1=list(l1[0])
l2=list(l2[0])
print(l1)
print(l2)


l=[]
for i in range(len(df)):
    k=df['XPOSTAG'][i]
    NE="O"
    if k in l1:
        NE=k
        df['XPOSTAG'][i]=NE
    l.append(NE)
    
        
df.insert(3,'NER',l)
df.groupby('NER').count()
df.to_csv(Output,index=False)