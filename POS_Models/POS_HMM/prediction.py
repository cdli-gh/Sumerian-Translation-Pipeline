from HMMs import *
import pickle
import re
import pandas
from tqdm import tqdm
import argparse


def POSLIST(Monolingual_sumerian,Prediction):
    my_list=[]
    for i in tqdm(range(len(Monolingual_sumerian))):
        print(i+1)
        print("sentence: "+Monolingual_sumerian[i])
        l=Monolingual_sumerian[i].split()
        POS=""
        for j in range(len(l)):
            if(re.search(r'\d+\(.+\)',l[j])):
                POS=POS+"("+l[j]+","+"NU"+")"+" "
            else:    
                POS=POS+"("+l[j]+","+Prediction[i][j]+")"+" "
        print('POS:'+POS)
        my_list.append(POS)
        print()
    
    return my_list



def Predict_Data(Monolingual_sumerian,train_bag,tags_df):
    start = time.time()
    y_pred=[]
    for i in range(len(Monolingual_sumerian)):
        l=Monolingual_sumerian[i].split()
        tagged_seq = HMM_Viterbi(l,train_bag,tags_df)
        y_pred.append(tagged_seq)

    end = time.time()
    difference = end-start
    print("Time taken in seconds: ", difference)
    
    return y_pred

def Openfile(filename):
    Monolingual_sumerian=[]
    with open(filename) as f:
        for line in f:
            line=line.strip()
            Monolingual_sumerian.append(line)
    return Monolingual_sumerian


def Savefile(Monolingual_sumerian,POS_list):
    with open(args.output, 'w') as f:
        for i in range(len(POS_list)):
            f.write("%s\n" %str(i+1))
            f.write("sentence: %s\n" %Monolingual_sumerian[i])
            f.write("POS:%s\n\n" % POS_list[i])
    print()
    
    
def main():
    Monolingual_sumerian=Openfile(args.input)
    
    df=pd.read_csv('Dataset/Augmented_POSTAG_training_ml.csv')
    tagged_sentence=Preparing_tagged_data(df)
    
    train_set, test_set = train_test_split(tagged_sentence,train_size=0.90,test_size=0.10,random_state=42)
    
    
    train_tagged_words = [ tup for sent in train_set for tup in sent ]
    test_tagged_words = [ tup for sent in test_set for tup in sent ]
    
    
    vocab = {word for word,tag in train_tagged_words}
    tags = {tag for word,tag in train_tagged_words}
    
    # creating t x t transition matrix of tags, t= no of tags
    # Matrix(i, j) represents P(jth tag after the ith tag)
    tags_df=TrainsitionMatrix(tags,train_tagged_words)
    
    Prediction=Predict_Data(Monolingual_sumerian,train_tagged_words,tags_df)
    POS_list=POSLIST(Monolingual_sumerian,Prediction)
    print("Saving_file "+args.output)
    Savefile(Monolingual_sumerian,POS_list)
    
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--input",help="Location of the Input text file to be  predicted", default="CDLI_Data/sumerian_demo.txt")
    parser.add_argument("-o","--output",help="Location of output text file(Result)", default='Output/POS_HMM.txt')
    args=parser.parse_args()
    
    print("\n")
    print("Input file is ", args.input)
    print("Output file will be ", args.output)
    print("\n")
    
    main()


