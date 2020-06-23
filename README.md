# Translating the whole Ur III corpus (GSoC - 2020)
The project aims to translate and provide information about 1.5M Raw Sumerian Text. The project aims to build a full translation pipeline, which will integrate NER (Named Entity Recognation) and POS (Part of Speech Tagging) of URIII Sumerian language, using various Neural Network Techniques and Rule-Based approaches to get perfect results and provide a full information of input Sumerian text.

The tags/symbols(ORACC Version) for NER and POS can be observed from here - https://cdli-gh.github.io/guides/guide_tagsets.html

## Data
### 1. CDLI_DATA
It contains Raw Sumerian Data taken from CDLI Dataset https://github.com/cdli-gh/data which is processed and converted to Monolingual Sumerian sentences. Further description is provided in the CDLI_DATA Folder.
### 2. Dataset  
The training data is extracted from https://github.com/cdli-gh/mtaac_gold_corpus. The conll files are extracted to be used for Named Entity Recgonation and POS tagging. This Repository contains the final data in csv format containg columns (Text, NER tag and/or POS tag) which can be used directly for the model. 


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

### Possible Mentors:

1. Ravneet Punia
2. Niko Schenk
