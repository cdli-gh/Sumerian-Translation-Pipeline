import re
import os
import subprocess
import argparse
stopping_chars=["@", "#", "&", "$"]




def OPEN(filename):
    lines=[]
    with open(filename, "r") as f:
        for line in f:
            line=line.strip()
            lines.append(line)

    return lines


def savefile(filename,LIST):
    with open(filename, 'w') as f:
        for line in LIST:
            f.write("%s\n" % line)




def processing_1(text_line):
    #x = re.sub(r"\[\.+\]","unk",text_line)
    #x = re.sub(r"...","unk",x)
    x = re.sub(r'\#', '', text_line)
    x = re.sub(r"\_", "", x)
    x = re.sub(r"\[", "", x)
    x = re.sub(r"\]", "", x)
    x = re.sub(r"\<", "", x)
    x = re.sub(r"\>", "", x)
    x = re.sub(r"\!", "", x)
    x = re.sub(r"@c", "", x)
    x = re.sub(r"@t", "", x)
    #x=re.sub(r"(x)+","x",x)
    x = re.sub(r"\?", "", x)
    x = x.split()
    x = " ".join(x)
    k = re.search(r"[a-wyzA-Z]+",x)
    if k:
        return x
    else:
        return ""  
                    
            
            
def Pipeline_start(lines):            
    Pipeline=[]
    for i,line in enumerate(lines):
        if len(line)>0 and line[0] not in stopping_chars:
            index=line.find(".")
            line=line[index+1:].strip()
            text=processing_1(line)
            Pipeline.append(text)
    return Pipeline
        
        
      


    
def Pipeline_end(lines):
    pipeline_result=[]
    
    POS=OPEN(output_dir+'pos_pipeline.txt')
    NER=OPEN(output_dir+'ner_pipeline.txt')
    tr_en=OPEN(output_dir+'trans_pipeline.txt')
    index=0
    for i,line in enumerate(lines):
        pipeline_result.append(line)
        if len(line)>0 and line[0] not in stopping_chars:
            #pipeline_result.append(POS[index])
            #pipeline_result.append(NER[index])
            pipeline_result.append('#tr.en:'+tr_en[index])
            index+=1
            
    return pipeline_result




def main():
    
    lines=OPEN(input_path)
    Pipeline=Pipeline_start(lines)
    savefile(output_dir+'pipeline.txt',Pipeline)
    
    subprocess.run('python3 POS_CRF_Model/prediction.py -i ATF_OUTPUT/pipeline.txt -o ATF_OUTPUT/pipeline1.txt',shell=True)
    
    
    
    #POS MODEL
    print("Running Part of speech Model for Sumerian Language")
    os.system(f'python3 {pos_path} -i {output_dir}pipeline.txt -o {output_dir}pos_pipeline.txt')
    
    #NER MODEL
    print("Running Named entity recognation Model for Sumerian Language")
    os.system(f'python3 {ner_path} -i {output_dir}pipeline.txt -o {output_dir}ner_pipeline.txt')
    
    #Translation MODEL
    print("Running Translation Model for Sumerian Language")
    os.system(f'onmt_translate -model {trans_path} -src {output_dir}pipeline.txt -output {output_dir}trans_pipeline.txt -replace_unk -verbose')
    
    
        
    
    pipeline_result=Pipeline_end(lines)
    savefile(output_dir+'pipeline_output.atf',pipeline_result)
    

    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--input",help="Location of the Input ATF File ", default="ATF_INPUT/demo.atf")
    parser.add_argument("-p","--pos",help="POS Model to be used from ['POS_CRF','POS_HMM','POS_Bi_LSTM','POS_Bi_LSTM_CRF'] (Case sensitive)", choices=['POS_CRF','POS_HMM','POS_Bi_LSTM','POS_Bi_LSTM_CRF'],default="POS_CRF" )
    parser.add_argument("-n","--ner",help="NER Model to be used from ['NER_CRF','NER_Bi_LSTM','NER_Bi_LSTM_CRF'] (Case_sensitive)", choices=['NER_CRF','NER_Bi_LSTM','NER_Bi_LSTM_CRF'],default="NER_CRF" )
    parser.add_argument("-t","--trans",help="Machine Translation Model to be used",choices=['Transformer'], default="Transformer" )
    parser.add_argument("-o","--output",help="Location of output Directory", default='ATF_OUTPUT/')
    
    args=parser.parse_args()
    
    input_path=args.input
    pos_path='POS_Models/'+args.pos+'/prediction.py'
    ner_path='NER_Models/'+args.ner+'/prediction.py'
    trans_path='Translation_Models/'+args.trans+'.pt'
    output_dir=args.output
    
    print("\n")
    print("Input file is ", input_path)
    print("POS Model path is ", pos_path)
    print("NER Model path is ", ner_path)
    print("Translation Model path is ", trans_path)
    print("Output directory is", output_dir)
    print("\n")
    
    main()
    
    
    
    
    
    
    #IF WE WANT TO USE CLASS 
    #pipeline=OPEN('ATF_OUTPUT/pipeline.txt')
    #POS=POS_tag(Pipeline)
    #savefile('ATF_OUTPUT/pipeline1.txt',POS)
    #NER=NER_tag(Pipeline)
    #savefile('ATF_OUTPUT/pipeline2.txt',NER)
    #translations=Translation_tag(Pipeline)
    #savefile('ATF_OUTPUT/pipeline3.txt',translations)
