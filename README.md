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
        
        
        
        
|__ docs/ --> documentation
|__ embeddings
        |__ get_glove_embeddings.sh --> script for downloading GloVe6B 100-dimensional word embeddings
        |__ get_fasttext_embeddings.sh --> script for downloading Fasttext word embeddings
|__ pretrained/
        |__ tagger_NER.hdf5 --> tagger for NER, BiLSTM+CNN+CRF trained on NER-2003 shared task, English
src/
|__utils/
   |__generate_tree_description.py --> import os
   |__generate_ft_emb.py --> generate predefined FastText embeddings for dataset
|__models/
   |__tagger_base.py --> abstract base class for all types of taggers
   |__tagger_birnn.py --> Vanilla recurrent network model for sequences tagging.
   |__tagger_birnn_crf.py --> BiLSTM/BiGRU + CRF tagger model
   |__tagger_birnn_cnn.py --> BiLSTM/BiGRU + char-level CNN tagger model
   |__tagger_birnn_cnn_crf.py --> BiLSTM/BiGRU + char-level CNN  + CRF tagger model   
|__data_io/   
   |__data_io_connl_ner_2003.py --> input/output data wrapper for CoNNL file format used in  NER-2003 Shared Task dataset
   |__data_io_connl_pe.py --> input/output data wrapper for CoNNL file format used in Persuassive Essays dataset
   |__data_io_connl_wd.py --> input/output data wrapper for CoNNL file format used in Web Discourse dataset


### Possible Mentors:

1. Ravneet Punia
2. Niko Schenk
