from flair.data import Sentence
from flair.models import SequenceTagger
import re

class Predictions():
    
    


    def __init__(self,INPUT,OUTPUT,pos=False,ner=False):
        self.IN=INPUT
        self.OUT=OUTPUT
        self.pos=pos
        self.ner=ner
        if self.pos==True:
            self.model = SequenceTagger.load('FLAIR/resources/taggers/flairpos/best-model.pt')
        if self.ner==True:
            self.model = SequenceTagger.load('FLAIR/resources/taggers/flairNer/best-model.pt')

    def Savefile(self,Monolingual_sumerian,POS_list):
        with open(self.OUT, 'w') as f:
            for i in range(len(POS_list)):
                f.write("%s\n" %str(i+1))
                f.write("sentence: %s\n" %Monolingual_sumerian[i])
                if(self.pos==True):
                    f.write("POS:%s\n" % POS_list[i])
                else:
                    f.write("NER:%s\n" % POS_list[i])
        print()
        
    def OPEN(self,filename):
        lines=[]
        with open(filename, "r") as f:
            for line in f:
                line=line.strip()
                lines.append(line)

        return lines
    

    def process(self,result):
        LIST=[]
        for i in range(len(result)):
            l=result[i].split()
            n=len(l)
            j=0;
            POS=""
            while(j<n):
                if(j+1==n):
                    POS=POS+"("+l[j]+","+"O"+")"+" "
                    j+=1
                elif(re.search(r'<.+>',l[j+1])):
                    prediction=l[j+1].replace('<',"").replace('>',"")
                    POS=POS+"("+l[j]+","+prediction+")"+" "
                    j+=2
                else:
                    POS=POS+"("+l[j]+","+"O"+")"+" "
                    j+=1
            LIST.append(POS)
            
        return LIST


    def predict(self):
        result=[]
        lines=self.OPEN(self.IN)
        
        for i in range(len(lines)):
            sentence = Sentence(lines[i])
            self.model.evaluate(sentence)
            l=sentence.to_tagged_string()
            result.append(l)

        predictions=self.process(result)   
        self.Savefile(lines,predictions)   
	

if __name__=='__main__':
    inp='FLAIR/demo.txt'
    out='FLAIR/prediction.txt'
    obj=Predictions(inp,out,False,True)
    obj.predict()

                                    
