from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments



# Data Preprocessing

paths = "CDLI_Data/Sumerian_monolingual_processed.txt"
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=1, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer.save_model("BERT/sumerianBERTo")

tokenizer = ByteLevelBPETokenizer(
    "BERT/sumerianBERTo/vocab.json",
    "BERT/sumerianBERTo/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
tokenizer.encode("dumu a-li2-wa-aq-rum")
print(tokenizer.encode("dumu a-li2-wa-aq-rum").tokens)




# Configuration

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
print(config)
tokenizer = RobertaTokenizerFast.from_pretrained("BERT/sumerianBERTo", max_len=512)
model = RobertaForMaskedLM(config=config)

print("Number of parameters \n")
print(model.num_parameters())


# Preparing Dataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="CDLI_Data/Sumerian_monolingual_processed.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)




# Training Model
training_args = TrainingArguments(
    output_dir="BERT/sumerianBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()




#save Model
trainer.save_model("BERT/sumerianBERTo")
print("Model Saved ..")



