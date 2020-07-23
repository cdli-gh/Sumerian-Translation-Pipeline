from flair.data import Sentence
from flair.models import SequenceTagger

# load the model you trained
model = SequenceTagger.load('resources/taggers/flairpos/final-model.pt')

# create example sentence
sentence = Sentence('5(iku) GAN2 bala-a 1(asz) 2(barig) 3(ban2)-ta')

# predict tags and print
model.predict(sentence)

print(sentence.to_tagged_string())
