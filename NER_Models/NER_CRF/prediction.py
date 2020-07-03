import pickle
import re
import pandas
from tqdm import tqdm
import argparse
from NER_CRF_features import features


#Default
# Txt file containing sumerian sentences
#Input = 'CDLI_Data/sumerian_demo_mono.txt'
#Output = 'Output/Pos_tagged_sentences_crf.txt'
# load saved crf model
#Pkl_Filename = "Saved_Models/POS_CRF_Model.pkl" 


        
        
        
        

def Openfile(filename):
    Monolingual_sumerian=[]
    with open(filename) as f:
        for line in f:
            line=line.strip()
            Monolingual_sumerian.append(line)
    return Monolingual_sumerian


def test_word_list(sentence):
    list_of_words=sentence.split()
    return list_of_words



def prepare_test_Data(all_sentences):
    X=[]
    for sentence in all_sentences:
        single_sentence_feature=[]
        #word list of sentence
        list_of_words=test_word_list(sentence)
        # Preparing features of all words of a single sentence/phrase
        for i in range(len(sentence.split())):
            #feature of word at index i
            d=features(list_of_words,i)
            single_sentence_feature.append(d)

        X.append(single_sentence_feature)  
    return X



def POSLIST(Monolingual_sumerian,Prediction):
    my_list=[]
    for i in tqdm(range(len(Monolingual_sumerian))):
        print(i+1)
        print("sentence: "+Monolingual_sumerian[i])
        l=Monolingual_sumerian[i].split()
        POS=""
        for j in range(len(l)):
            POS=POS+"("+l[j]+","+Prediction[i][j]+")"+" "
        print('NER:'+POS)
        my_list.append(POS)
        print()

    return my_list





def Savefile(Monolingual_sumerian,POS_list):
    with open(args.output, 'w') as f:
        for i in range(len(POS_list)):
            f.write("%s\n" %str(i+1))
            f.write("sentence: %s\n" %Monolingual_sumerian[i])
            f.write("NER:%s\n" % POS_list[i])
    print()




def main():

    Monolingual_sumerian= Openfile(args.input)

    with open(args.saved, 'rb') as file:  
        crf = pickle.load(file)

    Processed_sumerian_monolingual=prepare_test_Data(Monolingual_sumerian)

    Prediction=crf.predict(Processed_sumerian_monolingual)

    POS_list=POSLIST(Monolingual_sumerian,Prediction)

    print("Saving_file "+args.output)
    Savefile(Monolingual_sumerian,POS_list)


    
    
    
    
    

        
if __name__=='__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--input",help="Location of the Input text file to be  predicted", default="Dataset/sumerian_demo.txt")
    parser.add_argument("-s","--saved",help="Location of saved CRF weights in .pkl format", default="Saved_Models/NER/NER_CRF.pkl" )
    parser.add_argument("-o","--output",help="Location of output text file(Result)", default='Output/NER_CRF.txt')
    
    args=parser.parse_args()
    
    print("\n")
    print("Input file is ", args.input)
    print("Saved model is ", args.saved)
    print("Output file will be ", args.output)
    print("\n")
    
    main()
    
   
  

