import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter
import pickle
import re
#importing features
from Sumerian_CRF_features import features


def Preparing_tagged_data(df):
    tagged_sentence=[]
    c=1
    temp=[]
    for i in range(len(df)):
        if df['ID'][i]==c:
            temp.append((df['WORD'][i],df['POS'][i]))
        else:
            tagged_sentence.append(temp)
            temp=[]
            temp.append((df['WORD'][i],df['POS'][i]))
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
    for sentence in tagged_sentences:
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
    df=pd.read_csv('Dataset/POSTAG_training_ml.csv')
    tagged_sentence=Preparing_tagged_data(df)
    #printing details
    printing_details(tagged_sentence)
    
    train_set, test_set = train_test_split(tagged_sentence,test_size=0.1,random_state=42)
    
    #print("Number of Sentences in Training Data ",len(train_set))
    #print("Number of Sentences in Testing Data ",len(test_set))
    X_train,y_train=prepareData(tagged_sentence)
    X_test,y_test=prepareData(test_set)
    
    crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True)
    
    crf.fit(X_train, y_train)
    print(crf)
    
    print("Saving Model .....")
    # Save the Model to file in the current working directory
    Pkl_Filename = "Saved_Models/POS_CRF_Model.pkl"  
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(crf, file)
        
    print("Model Saved at "+ Pkl_Filename)
    print()    
    print("Checking the Algoritham's Performance \n")
    TestData(crf, X_train,y_train,X_test,y_test)
    
    
if __name__=='__main__':
    main()
    
