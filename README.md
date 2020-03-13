# Translating the whole Ur III corpus (GSoC - 2020)
This Repo contains the raw and final Dataset for the Named Entity recgonation and POS tagging for sumerian language.

The tags/symbols(ORACC Version) for NER and POS can be observed here - https://cdli-gh.github.io/guides/guide_tagsets.html

# Cleaned_Data
Since the Named Entity Recgonation and POS tagging depends on different symbols of raw data (extracted text), it does not require a deep cleaning. Cleaned Data repo contains the final data in csv format containg columns (Text, NER tag and/or POS tag) which can be used directly for the model. 

# AIM
The organization build an NMT model for Sumerian to English Translation, using a cleaned parallel dataset. But still around 1.5M raw untranslated data is available. The project aims a full translation pipeline, which will integrate NER and POS tagging of URIII languages, either using Neural Networks of Rule-Based approach to perfect the results and provide a full set of translations for the texts.

## Possible Mentors:

1. Ravneet Punia
2. Niko Schenk


