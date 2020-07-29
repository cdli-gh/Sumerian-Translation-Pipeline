# Translating the whole Ur III corpus (GSoC - 2020)
The project aims to translate and provide detailed information(POS tagging, Named Entity Recognation, English Translation) about 1.5M Raw Sumerian Text. The project aims to build a full translation pipeline, which will integrate NER (Named Entity Recognation), POS (Part of Speech Tagging) and machine translation of URIII Sumerian language, using Neural Network Techniques and Rule-Based approaches.

The tags/symbols(ORACC Version) for NER and POS can be observed from here - https://cdli-gh.github.io/guides/guide_tagsets.html

<details>
<summary> Requirements </summary> 
- Python 3.5.2 or higher <br/>
- numPy  <br/>
- pandas <br/>
- sklearn <br/>
- nltk <br/>
- sklearn_crfsuite <br/>
- matplotlib <br/>
- tqdm>=4.46.1 <br/>
- keras <br/>
- tensorflow <br/>
- CRF keras (pip3 install git+https://www.github.com/keras-team/keras-contrib.git) <br/>
- click <br/>
- OpenNMT-py (to use Machine Translation Models) <br/>

</details>

<p align="center">
  <img src="https://github.com/cdli-gh/Sumerian-Translation-Pipeline/blob/master/src/1.png" alt="Example image"/>
</p>
<p align="center">
  <img src="https://github.com/cdli-gh/Sumerian-Translation-Pipeline/blob/master/src/3.png" alt="Example image"/>
</p>
<p align="center">
  <img src="https://github.com/cdli-gh/Sumerian-Translation-Pipeline/blob/master/src/2.png" alt="Example image"/>
</p>




## Usage
Clone the Repo https://github.com/cdli-gh/Sumerian-Translation-Pipeline.git \
Install requirments by simply running requirments.sh file- \
Run pipeline.py file with the ATF input file, the results will be in ATF_OUTPUT folder 
```
git clone https://github.com/cdli-gh/Sumerian-Translation-Pipeline.git
cd Sumerian-Translation-Pipeline
sh requirments.sh
python3 pipeline.py -i ATF_INPUT/demo.atf -o ATF_OUTPUT
```




## Pipeline
Run Sumerian Translation Pipeline to extract information about Sumerian Text using POS, NER and Machine Translation. Since the weights are already saved, any model can be used directly without training.    
```
usage: pipeline.py [-h] [-i INPUT]
                   [-p {POS_CRF,POS_HMM,POS_Bi_LSTM,POS_Bi_LSTM_CRF}]
                   [-n {NER_CRF,NER_Bi_LSTM,NER_Bi_LSTM_CRF}]
                   [-t {Transformer}] [-o OUTPUT]
                   
  -i INPUT,
          Location of the Input ATF File
  -p {POS_CRF,POS_HMM,POS_Bi_LSTM,POS_Bi_LSTM_CRF}
                        POS Model to be used out of the above choices 
  -n {NER_CRF,NER_Bi_LSTM,NER_Bi_LSTM_CRF}
                        NER Model to be used from above the choices
  -t {Transformer}
                        Machine Translation Model to be used from above choices 
  -o OUTPUT
                        Location of output Directory/Folder
```


## POS and NER MODELS

### Training
#### 1 Hidden Markov Model (POS_HMM)
No need to train. To evaluate/run use HMM.py. It calculates probability without saving weights.
```
$ Python3 POS_Models/POS_HMM/training.py
  -i INPUT,     Location of the Input training file in the specific
                format (csv file with columns ID FORM XPOSTAG)
```
#### 2. Conditional Random Field (CRF_Model)
Uses Sumerian Features, can be modified with the subject knowledge. Weights of CRF models are saved in Saved_Model. There are different files for sumerian POS features and sumerian NER features
```
$ Python3 {POS_Models/NER_Models}/{POS_CRF/NER_CRF}/training.py
  -i INPUT,      Location of the Input training file in the specific
                 format (csv file with columns ID FORM XPOSTAG/NER)
  -o OUTPUT,     Location of the model weights to be saved
```
#### 3. Bidirectional LSTM Neural Network (Bi_LSTM_Model)
Deep learning model, used word2vec/fasttext word-embeddings, weights are saved in .h5 format in Saved_Model.  
```
$ Python3 {POS_Models/NER_Models}/POS_Bi_LSTM/training.py
  -i INPUT,      Location of the Input training file in the specific
                 format (csv file with columns ID FORM XPOSTAG)
  -e EMBEDDING,  Location of sumerian word embeddings
  -o OUTPUT,     Location of model weights to be saved
```
#### 4. Bidirectional LSTM Neural Network CRF (Bi_LSTM_CRF_Model)
Integrated Deep learning and conditional random field model, uses word2vec/fasttext word-embeddings, weights are saved in .h5 format in Saved_Model..  
```
$ Python3 {POS_Models/NER_Models}/{POS_Bi_LSTM_CRF/NER_Bi_LSTM_CRF}/training.py
  -i INPUT,      Location of the Input training file in the specific
                 format (csv file with columns ID FORM XPOSTAG)
  -e EMBEDDING,  Location of sumerian word embeddings
  -o OUTPUT,     Location of model weights to be saved
```


### Predictions
Since weights are saved, we can use all models directly for predictions.   
```
$ Python3 {POS_Models/NER_Models}/{Choice from the above models}/prediction.py
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Location of the Input text file to be predicted
  -s SAVED, --saved SAVED
                        Location of saved weights in .pkl or .h5 format
  -o OUTPUT, --output OUTPUT
                        Location of output text file(Result)

Any Model can be used for the predictions for any txt file. Here we used Dataset/sumerian_demo.txt as input file. which contains 150 random sentences from 1.5M sumerian text.
```


## Project structure

```
|__ Basic_Named_Entites/ --> collection of some sumerian named entities such as City name, Month names, etc.

|__ CDLI_data/
        |__ Monolingual_sumerian_processing.py/ --> Python Code for Processing Sumerian Monolingual dataset
        |__ extract.py/ --> Python Code for Extracting sumerian dataset from CDLI/data github
        |__ Sumerian_monolingual_original.txt/ --> Unprocessed Sumerian Monolingual dataset extracted from sumerian untranslated.txt
        |__ Sumerian_monolingual_processed.txt/ --> Processed Sumerian Monolingual dataset extracted from sumerian untranslated.txt and processed using Monolingual_sumerian_processing.py
        |__ Sumerian_translated.txt/ --> Sumerian Parallel En-Su data extracted from CDLI raw data using extract.py
        |__ Sumerian_untranslated.txt/ --> Sumerian Monolingual data extracted from CDLI raw data using extract.py
        
|__ Dataset/
        |__ Raw/
            |__ CDLI conll files, human annotated data, taken from CDLI/mtacc_gold_corpus
        |__ Augmented_POSTAG_training_ml.csv/ --> Augmented POS tag training Data using Named dictionary and POS_training_ml.csv, generated after applying TextAugmentation
        |__ Augmented_RAW_NER_POS.csv/ --> Augmented Raw training Data include POS and NER using Named dictionary and POS_training_ml.csv generated after applying TextAugmentation
        |__ POSTAG_training_ml.csv/ --> POS tagging dataset created from 'Dataset/Raw_NER_POS_data.csv' using scripts/POS_TrainingData_creater.py
        |__ Raw_NER_POS_data.csv/ --> Extracted and processed sumerian conll files using scripts/CDLI_conll_extracter.py 
        |__ sumerian_demo.txt/ --> Randomly extracted 150 sentences for the manual testing  of modeles from 1.5M sumerian text, code used - scripts/sumerian_random.py 

|__NER_Models/         
        |__ NER_CRF_Model
                |__ NER_CRF_features.py --> set of rules/features to identify NER tags for sumerian languages 
                |__ training.py --> conditional random field model including abouve feature set to identify NER taggs for sumerian language 
                |__ prediction.py --> python file use to predict output of CRF model

|__ Output/ --> Results of POS using different models (CRF,HMM,Bi_LSTM) on 150 randomly selected sumerian sentences

|__POS_Models/
        |__ POS_Bi_LSTM
                |__ training.py --> Bidirectional LSTM Neural network model trained with word2vec embeddings 
                |__ prediction.py --> python file use to predict output of deep neural network
                
        |__ POS_Bi_LSTM_CRF
                |__ training.py --> Bidirectional LSTM Neural network with CRF integrated, trained with word2vec embeddings 
                |__ prediction.py --> python file use to predict output of deep neural network
                
        |__ POS_CRF_Model
                |__ POS_CRF_features.py --> set of rules/features to identify POS tags for sumerian languages 
                |__ training.py --> conditional random field model including abouve feature set to identify pos taggs for sumerian language 
                |__ prediction.py --> python file use to predict output of CRF model

        |__ POS_HMM_Model
                |__ training.py --> Hidden markov model based on emission and transition probabilities   
                |__ prediction.py --> python file use to predict output of HMM model
        
|__ Saved_Model/ --> Saved weights of above models, output can be predicted using these without training the models 
        |__ POS/
        |__ NER/
        
|__ TextAugmentation/
        |__ Raw/
            |__ Raw_NER_POS_data.csv/ --> Extracted and processed sumerian conll files using
            |__ pndictionary.csv/ --> raw dataset which contains sumerian names and associated named entities 
        |__ pndictioanry_processed.csv / --> processed pndictionary.csv using emission pndictionary_process.py 
        |__ pndictionary_process.py / --> python code used to process raw dictionary and convert in usable form
        |__ textdata_augmentation.py/ --> Python code for text augmentation, used raw human annotated dataset of 3500 phrase and converted to 22500 phrases using pndictioanry_processed.csv and Raw_NER_POS_data.csv

|__ Word_Embeddings/ --> Sumerian word embeddings(word2vec,fasttext) and code to train the word embeddings on Sumerian_monolingual_processed.txt

|__scripts/
   |__CDLI_conll_extracter.py --> code to extract POS and Raw_POS_NER datset from CDLI conll files 
   |__POS_TrainingData_creater.py --> code to creat POS_Training dataset 
   |__sumerian_random.py --> code to extract 150 random sentences from 1.5M Sumerian_monolingual_processed.txt

```


### Mentor:

1. Ravneet Punia
