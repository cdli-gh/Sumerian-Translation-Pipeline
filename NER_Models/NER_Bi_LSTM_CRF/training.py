import numpy as np
import pandas as pd
import matplotlib
import numpy as np
import pandas as pd
import pickle
from sklearn_crfsuite import metrics
from collections import OrderedDict 
#import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model, Input
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional,  TimeDistributed
from keras.optimizers import Adam
from keras_contrib.layers import CRF
from tqdm.keras import TqdmCallback
import argparse



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



def prepareData(sentences,word2idx,tag2idx):
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=MAX, sequences=X, padding="post",value=word2idx["<end>"])
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=MAX, sequences=y, padding="post", value=tag2idx["<e>"])
    y = to_categorical(y, num_classes=len(tag2idx))
    return X,y




def preparedicts(df):
    vocab=list(df["FORM"].values)
    f = open(args.embedding)
    for line in f:
        values = line.split()
        word = values[0]
        vocab.append(word)
    vocab=sorted(list(set(vocab)))
    vocab.append("<end>")
    vocab.append("UNK")
    
    tags = sorted(list(set(df["XPOSTAG"].values)))
    tags.append("<e>")
    
    word2idx=OrderedDict() 
    idx2word=OrderedDict() 
    tag2idx=OrderedDict() 
    idx2tag=OrderedDict() 
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: w for w, i in tag2idx.items()}
    
    print("\n Saving Vocab as a list of josn files \n")
    vocabulary=[word2idx,idx2word,tag2idx,idx2tag]
    with open('NER_Models/NER_Bi_LSTM_CRF/Sumerian_Vocab.pkl', 'wb') as f:
    	pickle.dump(vocabulary,f)
    	
    return word2idx,idx2word,tag2idx,idx2tag



def embeddings(word2idx):
    embeddings_index = dict()
    f = open(args.embedding)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    print("dimention is",len(coefs))
    embedding_matrix = np.zeros((len(word2idx), len(coefs)))
    for i,word in enumerate(word2idx.keys()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector        
    
    return embedding_matrix




def BUILD_MODEL(X,MAX,n_words,n_tags,embedding_matrix):
    input_word = Input(shape = (MAX,))
    model=Embedding(input_dim=n_words,input_length=X.shape[1], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix],trainable=False)(input_word)
    model=Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(model)
    model=TimeDistributed(Dense(32, activation ='relu'))(model)
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output
    model = Model(input_word, out)
    model.summary()
    model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
    return model

    
def evaluate_model(history):    
    plt.title('Accuracy')
    plt.plot(history.history['crf_viterbi_accuracy'], label='train')
    plt.plot(history.history['val_crf_viterbi_accuracy'], label='Val')
    plt.legend()
    plt.show();
  
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.legend()
    plt.show();
    
def pred2label(pred,idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out


def TestData(model,X_test,y_test,idx2tag,label=None):
    test_pred = model.predict(X_test, verbose=1)
    y_pred = pred2label(test_pred,idx2tag)
    y_test = pred2label(y_test,idx2tag)
    print("test accuracy is %f \n" % np.multiply(metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=label),100))
    print("Test classification report is \n")
    print(metrics.flat_classification_report(y_test, y_pred, digits=3,labels=label))
    
    

    

def main():
    df= pd.read_csv(args.input)
    df=df[['ID','FORM','XPOSTAG']]
    print(df.head())
    sentences = Preparing_tagged_data(df)
    print ('Maximum sequence length:', MAX)
    word2idx,idx2word,tag2idx,idx2tag= preparedicts(df)
    X,y=prepareData(sentences,word2idx,tag2idx)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,random_state=7)
    
    print("Dataset dimentions \n")
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)

    embedding_matrix=embeddings(word2idx)
    
    model=BUILD_MODEL(X, MAX,len(word2idx),len(tag2idx),embedding_matrix)
    
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=32,validation_split=0.1,verbose=0,callbacks=[TqdmCallback(verbose=2)])
    
    #evaluate_model(history)
    print("Saving model at ",args.output)
    model.save(args.output)
    TestData(model,X_test,y_test,idx2tag)
    
    
if __name__=='__main__':
    MAX=50
    #Input_path='Dataset/Augmented_NER_training_ml.csv'
    #Embedding_path='Word_Embeddings/sumerian_word2vec_50.txt'
    #saved_path='Saved_Models/NER_Bi_LSTM_CRF.h5'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--input",help="Location of the Input training file in the specific format (csv file with columns ID FORM XPOSTAG)",default="Dataset/Augmented_NER_training_ml.csv")
    parser.add_argument("-e","--embedding",help="Location of sumerian word embeddings",default='Word_Embeddings/sumerian_word2vec_50.txt')
    parser.add_argument("-o","--output",help="Location of model weights to be saved",default="Saved_Models/NER/NER_Bi_LSTM_CRF.h5")
    
    args=parser.parse_args()
    
    print("\n")
    print("Input file is ", args.input)
    print("embedding file is ", args.embedding)
    print("Output file is ", args.output)
    print("\n")
    

    main()
