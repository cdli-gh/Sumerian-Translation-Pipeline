# Translating the whole Ur III corpus (GSoC - 2020)
This Repo contains the raw and final Dataset for the Named Entity recgonation and POS tagging for sumerian language.

The tags/symbols(ORACC Version) for NER and POS can be observed here - https://cdli-gh.github.io/guides/guide_tagsets.html

# Data
Contains Two folders


### 1. CDLI_DATA
It contains Raw Sumerian Data taken from CDLI Dataset https://github.com/cdli-gh/data which is processed and converted to Monolingual Sumerian sentences. Further description is provided in the CDLI_DATA Folder.
### 2. Dataset  
The training data is extracted from https://github.com/cdli-gh/mtaac_gold_corpus. The conll files are extracted to be used for Named Entity Recgonation and POS tagging. This Repository contains the final data in csv format containg columns (Text, NER tag and/or POS tag) which can be used directly for the model. 


# AIM
The organization build an NMT model for Sumerian to English Translation, using a cleaned parallel dataset. But still around 1.5M raw untranslated data is available. The project aims to build a full translation pipeline, which will integrate NER (Named Entity Recognation) and POS (Part of Speech Tagging) of URIII Sumerian language, using various Neural Network Techniques and Rule-Based approaches to get perfect results and provide a full information of input Sumerian text.

## Possible Mentors:

1. Ravneet Punia
2. Niko Schenk
