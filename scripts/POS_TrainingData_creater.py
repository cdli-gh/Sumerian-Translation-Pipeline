import pandas as pd
Input='Dataset/Raw_NER_POS_data.csv'
Output='Dataset/POSTAG_training_ml.csv'


df=pd.read_csv(Input)
index=[]
for i in range(len(df)):
    l=df["ID"][i].split(".")
    ind=int(l[-1])
    index.append(ind)
df=df[['FORM','XPOSTAG']]
df.insert(0,"ID",index)
df.head()


l1=pd.read_csv('Named Entities Sumerian ORACC.csv',header=None)
l2=pd.read_csv('Part of Speech (POS) tags Sumerian ORACC.csv',header=None)
l1=list(l1[0])
l2=list(l2[0])


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
    
    
def process_training(data):
    c=0
    for i in range(len(data)):
        if int(data['ID'][i])==1:
            c+=1
            data['ID'][i]=c
        else:
            data['ID'][i]=c
    return data

df=process_training(df)
df=df.rename(columns={'FORM':'FORM','XPOSTAG':'XPOSTAG'})

print('final dataset is \n')
print(df.head(10))
print()
print('Final dataset contains \n')
print(df.groupby('POS').count())
print()
print('Saving POSTAG_training_ml.csv')
df.to_csv(Output,index=False)
    
