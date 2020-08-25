import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter
import argparse
import pprint, time
import re


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
    for i in range(len(tagged_sentence)):
        tagged_sentence[i].insert(len(tagged_sentence[i]),(('<end>','<e>')))
    
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
    
    
            
# compute Emission Probability
def word_given_tag(word, tag, train_bag):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    #now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)




# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


def TrainsitionMatrix(tags, train_bag):
    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)): 
            tags_matrix[i, j] = t2_given_t1(t2, t1, train_bag)[0]/t2_given_t1(t2, t1, train_bag)[1]    
    tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
    
    return tags_df

    

def rules_key(word):
    if(re.search(r'{d}',word) or re.search(r'{ki}',word) or re.search(r'lugal',word)):
        return 'NE'
    elif (re.search(r'.*\d+\(.*\)',word)):
        return 'NU'
    return 'N'
    
    
    
def HMM_Viterbi(words, train_bag,tags_df):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['<e>', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag, train_bag)[0]/word_given_tag(words[key], tag, train_bag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
            
        pmax = max(p)
        if pmax==0:
            state_max=rules_key(words[key])
        else:
            # getting state for which probability is maximum
            state_max = T[p.index(pmax)] 
        state.append(state_max)
        
    return list(state)



def TestData(test_set,train_bag,tags_df):
    #Code to test all the test sentences
    #(takes alot of time to run s0 we wont run it here)
    # tagging the test sentences()
    start = time.time()
    y_test=[]
    y_pred=[]
    for sent in test_set:
        y_test_untagged = [word[0] for word in sent]

        y_test.append([word[1] for word in sent])

        tagged_seq = HMM_Viterbi(y_test_untagged,train_bag,tags_df)

        y_pred.append(tagged_seq)

    end = time.time()
    difference = end-start
    
    print("Time taken in seconds: ", difference)


    print("test accuracy is %f \n" % np.multiply(metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=None),100))
    print("Test classification report is \n")
    print(metrics.flat_classification_report(y_test, y_pred, digits=3,labels=None))

    
    

def main():
    df=pd.read_csv(args.input)
    tagged_sentence=Preparing_tagged_data(df)
    #printing details
    printing_details(tagged_sentence)
    
    train_set, test_set = train_test_split(tagged_sentence,train_size=0.95,test_size=0.05,random_state=7)
    
    
    train_tagged_words = [ tup for sent in train_set for tup in sent ]
    test_tagged_words = [ tup for sent in test_set for tup in sent ]
    
    
    vocab = {word for word,tag in train_tagged_words}
    tags = {tag for word,tag in train_tagged_words}
    
    
    # creating t x t transition matrix of tags, t= no of tags
    # Matrix(i, j) represents P(jth tag after the ith tag)
    tags_df=TrainsitionMatrix(tags,train_tagged_words)
    print(tags_df)
    
    print("Saving Transition Matrix.....")
    # Save the Model to file in the current working directory
    
    tags_df.to_csv('Saved_Models/POS/TrainsitionMatrix_HMM.csv')    
    print("File Saved at " + 'Saved_Models/POS/TrainsitionMatrix_HMM.csv')
    print()    
    
    print("Checking the Algoritham's Performance \n")
    TestData(test_set,train_tagged_words,tags_df)
    
    
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",help="Location of the Input training file in the specific format (csv file with columns ID FORM XPOSTAG)",default="Dataset/Augmented_POSTAG_training_ml.csv")
    
    args=parser.parse_args()
    
    print("\n")
    print("Input file is ", args.input)
    print("\n")

    main()
