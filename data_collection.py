import pandas as pd
import transformers
from transformers import AutoModelWithLMHead, AutoTokenizer, TrainingArguments, Trainer

# Instantiate a model
model = AutoModelWithLMHead.from_pretrained("distilgpt2")

#Load the data
data = pd.read_csv('data.csv')

#Preprocess the data
data = data.dropna()	#remove missing values
data = data.drop_duplicates()	#remove duplicates
data = data.sample(frac=1)	#shuffle the data


# Instantiate a tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Tokenize the text
text = data['text'].values
tokenized_text = tokenizer(text, padding=True, truncation=True)


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    eval_steps = 1000,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    save_steps=1000,
    save_total_limit=2
)

# Instantiate a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_text,
    eval_dataset=tokenized_text
)

# Train the model
trainer.train()
