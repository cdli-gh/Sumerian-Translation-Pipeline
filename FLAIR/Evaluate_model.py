from flair.data import Corpus
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
columns = {0: 'text', 1: 'tags'}

#Name the data folder
data_folder = 'FLAIR/NER_corpus'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')
                              
                              
# load the model you trained
model = SequenceTagger.load('FLAIR/resources/taggers/flairNer1/best-model.pt')

#if you want to save test results in predcitions.txt
#result,val= model.evaluate(corpus.test,out_path=f"predictions.txt")
#else

result,val= model.evaluate(corpus.test)
print(result.detailed_results)
