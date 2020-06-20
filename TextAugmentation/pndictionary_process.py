data2=pd.read_csv("Raw/pndictionary.csv",error_bad_lines=False,header=None)
data2=data2.drop([0,3,4,5],axis=1)
print(data2.head())
data2=data2.drop_duplicates(subset=[1],keep='first')
data2=data2.reset_index(drop=True)


l=[]
for i in range(len(data2)):
    s=data2[2][i]
    if s==' `value`' or s==" 'GN ara[times]'":
        l.append("NULL")
        continue
    s=re.sub('\'',"",s)
    s=re.sub(" ","",s)
    s=s.split('|')
    k=list()
    for j in s:
        if len(j)==2:
            k.append(j)
    s="|".join(k)
    l.append(s)
    
data2[3]=l
data2=data2[data2[3]!="NULL"]
data2=data2.reset_index(drop=True)
data2=data2.drop([2],axis=1)
data2.rename(columns = {1:'Text',3:'NER'}, inplace = True)
data2.to_csv("pndictioanry_processed.csv")