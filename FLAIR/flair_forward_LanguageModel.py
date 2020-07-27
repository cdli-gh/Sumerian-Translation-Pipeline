from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
is_forward_lm = True
dictionary: Dictionary = Dictionary.load('chars')
corpus = TextCorpus('FLAIR/corpus',
                    dictionary,
                    is_forward_lm,
                    character_level=True)
                    
language_model = LanguageModel(dictionary,
                               is_forward_lm,
                               hidden_size=1024,
                               nlayers=2)

trainer = LanguageModelTrainer(language_model, corpus)

trainer.train('FLAIR/resources/taggers/language_model_forward',
              sequence_length=50,
              mini_batch_size=50,
              learning_rate=10,
              patience=3,
              max_epochs=50,
              checkpoint=True)

