
class TAGCLASS:
    
    def __init__(self,pos_input,ner_input):
        self.pos_input=pos_input
        self.ner_input=ner_input
        
    
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
        taglist=[]
        for i in range(len(POS)):
            pos_text=POS[i].lstrip('POS:').replace('(','').replace(')','').split(" ")
            ner_text=NER[i].lstrip('NER:').replace('(','').replace(')','').split(" ")
            for j in range(len(pos_text)):
                pos_tag=pos_text[j].split(',')[1]
                ner_tag=ner_text[j].split(',')[1]
                if(pos_tag=='NE' and ner_tag!='O'):
                    taglist.append(ner_tag)
                elif(pos_tag=='O'):
                    taglist.append(" ")
                else:
                    taglist.append(pos_tag)
                    
        return taglist
        
        
        
        
        
        
if __name__=='__main__':
    
    POS_INPUT='ATF_OUTPUT/pos_pipeline.txt'
    NER_INPUT='ATF_OUTPUT/ner_pipeline.txt'
    Obj=TAGCLASS(POS_INPUT,NER_INPUT)
    taglist=Obj.tag2list()
    print(taglist)
    
    