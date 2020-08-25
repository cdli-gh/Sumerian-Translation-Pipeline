import nltk
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter
import pickle
import re
#importing features
from NER_CRF_features import features


def Preparing_tagged_data(df):
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


def printing_details(tagged_sentence):
    print("\n")
    print('Example of tagged_sentence')
    print(tagged_sentence[2])
    print("\n")
    print('Dataset_information')
    print("Number of Tagged Sentences ",len(tagged_sentence))
    tagged_words=[tup for sent in tagged_sentence for tup in sent]
    print("Total Number of Tagged words", len(tagged_words))
    vocab=set([word for word,tag in tagged_words])
    print("Vocabulary of the Corpus",len(vocab))
    tags=set([tag for word,tag in tagged_words])
    print("Number of Tags in the Corpus ",len(tags))
    print("\n")
    
    
            
def word_list(sentence):
    list_of_words=[]
    for word,tag in sentence:
        list_of_words.append(word)
    return list_of_words


def prepareData(tagged_sentences):
    X,y=[],[]
    for index,sentence in enumerate(tagged_sentences):
        single_sentence_feature=[]
        # Preparing features of all words of a single sentence/phrase
        for i in range(len(sentence)):
            #word list of sentence
            list_of_words=word_list(sentence)
            #feature of word at index i
            d=features(list_of_words,i)
            single_sentence_feature.append(d)
            
        X.append(single_sentence_feature)
        # append list of tags for the associated sentence
        y.append([tag for word,tag in sentence])
    return X,y


def TestData(crf, X_train,y_train,X_test,y_test):
    y_pred=crf.predict(X_test)
    y_pred_train=crf.predict(X_train)
    print("training accuracy is %f \n" % metrics.flat_f1_score(y_train, y_pred_train,average='weighted',labels=crf.classes_))
    print("test accuracy is %f \n" % metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=crf.classes_))
    print("Test classification report is \n")
    print(metrics.flat_classification_report(y_test, y_pred, labels=crf.classes_, digits=3))





def main():

    df=pd.read_csv(args.input)
    tagged_sentence=Preparing_tagged_data(df)
    df=df[['ID','FORM','XPOSTAG']]
    #printing details
    printing_details(tagged_sentence)
    
    train_set, test_set = train_test_split(tagged_sentence,test_size=0.05,random_state=7)
    
    #print("Number of Sentences in Training Data ",len(train_set))
    #print("Number of Sentences in Testing Data ",len(test_set))
    X_train,y_train=prepareData(tagged_sentence)
    X_test,y_test=prepareData(test_set)
    
    crf = CRF(
    algorithm='l2sgd',
    c2=0.1,
    max_iterations=1000,
    all_possible_transitions=True)
    
    crf.fit(X_train, y_train)
    print(crf)
    
    print("Saving Model .....")
    # Save the Model to file in the current working directory
    Pkl_Filename = args.output
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(crf, file)
        
    print("Model Saved at "+ Pkl_Filename)
    print()    
    print("Checking the Algoritham's Performance \n")
    TestData(crf, X_train,y_train,X_test,y_test)
    
    
if __name__=='__main__':
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--input",help="Location of the Input training file in the specific format (csv file with columns ID FORM XPOSTAG NER)",default="Dataset/ETCSL_ORACC_NER.csv")
    parser.add_argument("-o","--output",help="Location of model weights to be saved",default="Saved_Models/NER/NER_CRF.pkl")
    
    args=parser.parse_args()
    
    print("\n")
    print("Input file is ", args.input)
    print("Output file is ", args.output)
    print("\n")
    
    main()
    
