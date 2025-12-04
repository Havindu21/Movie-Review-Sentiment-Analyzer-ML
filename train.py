import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch

# 1. Load data
df = pd.read_csv("data/reviews.csv")

# Convert labels to numbers
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 2. Split train & test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to HuggingFace dataset
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# 3. Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch['review'], padding='max_length', truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 4. Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 5. Training setup
args = TrainingArguments(
    output_dir="model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

# 6. Train the model
trainer.train()

# 7. Save trained model
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Training complete! Model saved to /model")
