
import json

class TAGCLASS:
    
    def __init__(self,pos_input,ner_input,pipeline):
        self.pos_input=pos_input
        self.ner_input=ner_input
        self.pipeline=pipeline
        self.tagdict=dict()
    
    def OPEN(self,filename):
        lines=[]
        with open(filename, "r") as f:
            for line in f:
                line=line.strip()
                lines.append(line)
        return lines

    
    def tag2list(self):
        POS=self.OPEN(self.pos_input)
        POS=[POS[i] for i in range(2,len(POS),3)]
        NER=self.OPEN(self.ner_input)
        NER=[NER[i] for i in range(2,len(NER),3)]
        pipe=self.OPEN(self.pipeline)
        for i in range(len(POS)):
            text=pipe[i].split(" ")
            print(text)
            print(POS[i])
            print(NER[i])
            pos_text=POS[i].lstrip('POS:').replace('(','').replace(')','').split(" ")
            ner_text=NER[i].lstrip('NER:').replace('(','').replace(')','').split(" ")
            for j in range(len(pos_text)):
                tag=" "
                word=text[j]
                pos_tag=pos_text[j].split(',')[1]
                ner_tag=ner_text[j].split(',')[1]
                if (pos_tag=='NE' and ner_tag=='O'):
                	tag='PN'
                elif (pos_tag=='NE'):
                    tag=ner_tag
                elif(pos_tag=='O'):
                    tag=" "
                else:
                    tag=pos_tag
                    
                temp=self.tagdict.get(word)
                if(temp==None):
                    self.tagdict[word]=tag
                elif temp==" ":
                    self.tagdict[word]=tag
                    
        return self.tagdict
        
        
        
        
        
        
if __name__=='__main__':
    
    POS_INPUT='ATF_OUTPUT/pos_pipeline.txt'
    NER_INPUT='ATF_OUTPUT/ner_pipeline.txt'
    pipeline='ATF_OUTPUT/pipeline.txt'
    Obj=TAGCLASS(POS_INPUT,NER_INPUT,pipeline)
    taglist=Obj.tag2list()
    with open('ATF_2_Conll/tag_dict.json', 'w', encoding='utf-8') as f:
        json.dump(taglist, f, ensure_ascii=False, indent=4)
    #taglist=Obj.tag2list()
    #print(taglist)
    
    
