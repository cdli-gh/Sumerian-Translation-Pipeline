# UrIII-names
Name authority for Ur III period administrative texts
# Goal
Goal of this project is to prepare a full list of names of Ur III period (ca. 2100 - 2000 BCE) administrative texts. The name list will have side by side the name form as it appears in the text and the normalized form of the name, following the conventions as used in the Open Richly Annotated Cuneiform Corpus ([ORACC](http://oracc.org)). 

The Ur III period yielded very large numbers of adminstrative texts in Sumerian cuneiform. Current estimates put the number of individual tablets from this time at about 150,000, some 100,000 of which are available in digitized transliterations. Analysis of this dataset depends on the ability to recognize people, deities and places reliably.

# Normalization
Normalized names ignore spelling differences (such as `U3-ta-mi-ca-ra-am` vs. `U2-ta2-mi-car-ra-am`) and leave off morphological suffixes (such as ablative `-ta`). 

The normalized name is always followed by square brackets and a name category, using the following categories:
- DN    Deity Name
- RN    Royal Name
- PN    Personal Name
- SN    Settlement Name

Examples of fully entries are:
- {d}Šul-gi               Šulgir[]RN
- {d}Nin-gir-su-ke4       Ningirsuk[]DN
- U3-ta2-mi-ca-ra-am-ta   Utamišaram[]PN
- Uri5{ki}-ce3            Urim[]SN

# Process
The current list (in the directory Output) is produced by the Python 3.5 Notebook included in this repository. It takes the entirety of the data set in [BDTNS](http://bdtns.filol.csic.es/), extracts names (characterized by an initial capital) and attempts to normalize these names on the basis of rules. The result has many errors and infelicities, but is a good start for a researcher who is knowledgeable about this period. All normalizations will be reviewed and, where necessary, corrected. The current list, therefore, is only stage 1 in the process.

# Usage
The list and the Notebook can be used and adapted by anyone interested under a Creative Commons Share Alike license [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
