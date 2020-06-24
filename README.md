# Translating the whole Ur III corpus (GSoC - 2020)
The project aims to translate and provide detailed information(POS tagging, Named Entity Recognation, English Translation) about 1.5M Raw Sumerian Text. The project aims to build a full translation pipeline, which will integrate NER (Named Entity Recognation), POS (Part of Speech Tagging) and machine translation of URIII Sumerian language, using Neural Network Techniques and Rule-Based approaches.

The tags/symbols(ORACC Version) for NER and POS can be observed from here - https://cdli-gh.github.io/guides/guide_tagsets.html


## Requirements

- Python 3.5.2 or higher
- numPy
- pandas
- sklearn
- matplotlib
- tqdm
- keras
- tensorflow
- CRF keras (pip3 install git+https://www.github.com/keras-team/keras-contrib.git)


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
        
|__ Output/ --> Results of POS using different models (CRF,HMM,Bi_LSTM) on 150 randomly selected sumerian sentences

|__ POS_Bi_LSTM
        |__ POS_Deep_learning.py --> Bidirectional LSTM Neural network model trained with fasttext word embeddings 
        |__ prediction.py --> python file use to predict output of deep neural network

|__ POS_CRF_Model
        |__ Sumerian_CRF_features.py --> set of rules/features to identify POS tags for sumerian languages 
        |__ training.py --> conditional random field model including abouve feature set to identify pos taggs for sumerian language 
        |__ prediction.py --> python file use to predict output of CRF model

|__ POS_HMM_Model
        |__ POS_Deep_learning.py --> Hidden markov model based on emission and transition probabilities   
        |__ prediction.py --> python file use to predict output of HMM model
        
|__ Saved_Model/ --> Saved weights of above three models, output can be predicted using these without training the models 

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

## Usage
Clone the Repo https://github.com/cdli-gh/Sumerian-NER.git
Install requirments - $ sh requirments.sh 

### 1. Hidden Markov Model (POS_HMM)
No need to train. To evaluate/run use HMM.py. It calculates probability without saving weights.
```
$ Python3 POS_HMM_Model/HMMs.py
HMMs.py [-h] [-i INPUT]
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Location of the Input training file in the specific
                        format (csv file with columns ID FORM XPOSTAG)
```
### 2. Conditional Random Field (POS_CRF_Model)
Uses Sumerian Features, can be modified with the subject knowledge. Weights of CRF models are saved in Saved_Model.
```
$ python3 POS_CRF_Model/training.py
training.py [-h] [-i INPUT] [-o OUTPUT]
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Location of the Input training file in the specific
                        format (csv file with columns ID FORM XPOSTAG)
  -o OUTPUT, --output OUTPUT
                        Location of model weights to be saved
```
### 3. Bidirectional LSTM Neural Network (Bi_LSTM_Model)
Deep learning model, uses fasttext word-embeddings, weights are saved as model.h5 in Saved_Model.  
```
$ Python3 POS_Bi_LSTM/POS_Deep_learning.py
POS_Deep_learning.py [-h] [-i INPUT] [-e EMBEDDING] [-o OUTPUT]
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Location of the Input training file in the specific
                        format (csv file with columns ID FORM XPOSTAG)
  -e EMBEDDING, --embedding EMBEDDING
                        Location of sumerian word embeddings
  -o OUTPUT, --output OUTPUT
                        Location of model weights to be saved
```

## Predictions
Since weights are saved, we can directly use all three for models directly for predictions.   
```
prediction.py [-h] [-i INPUT] [-s SAVED] [-o OUTPUT]
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Location of the Input text file to be predicted
  -s SAVED, --saved SAVED
                        Location of saved CRF weights in .pkl format
  -o OUTPUT, --output OUTPUT
                        Location of output text file(Result)

Any Model can be used for the predictions for any txt file. Here we used Dataset/sumerian_demo.txt as input file.  

$ python3 POS_HMM_Model/prediction.py
$ python3 POS_CRF_Model/prediction.py
$ python3 POS_Bi_LSTM/prediction.py
```

### Mentor:

1. Ravneet Punia
