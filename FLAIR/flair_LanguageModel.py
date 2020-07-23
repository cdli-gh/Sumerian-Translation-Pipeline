from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
is_forward_lm = True
dictionary: Dictionary = Dictionary.load('chars')
corpus = TextCorpus('corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)
                    
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=512,
                               nlayers=2)

trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('resources/taggers/language_model',
              sequence_length=30,
              mini_batch_size=50,
              max_epochs=20)

