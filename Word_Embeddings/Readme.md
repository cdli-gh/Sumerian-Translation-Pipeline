Fast text embeddings can be trained after cloning (https://github.com/facebookresearch/fastText/)[https://github.com/facebookresearch/fastText/]


$ git clone https://github.com/facebookresearch/fastText.git

$ cd fastText

$ pip install .



For Training


Word_Embeddings/fasttext skipgram -input CDLI_Data/Sumerian_monolingual_processed.txt -output Word_Embeddings/sumerian_fasttext_100.txt
