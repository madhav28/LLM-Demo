# Import necessary libraries
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding, 
    EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import json
import joblib
import os
import shutil

# Specify the pre-trained model ID and directories for saving outputs and logs
model_id = "FacebookAI/roberta-base"
output_dir="/mnt/scratch/lellaom/LLM Fine-tuning Codes/Encoder Fine-tuning/models"
logging_dir="/mnt/scratch/lellaom/LLM Fine-tuning Codes/Encoder Fine-tuning/logs"

# Remove existing output directory if it exists
try:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Directory '{output_dir}' and its contents removed successfully.")
except Exception as e:
    print(f"Error removing directory '{output_dir}': {e}")

# Remove existing logging directory if it exists
try:
    if os.path.exists(logging_dir):
        shutil.rmtree(logging_dir)
        print(f"Directory '{logging_dir}' and its contents removed successfully.")
except Exception as e:
    print(f"Error removing directory '{logging_dir}': {e}")

# Load the dataset from a CSV file
df = pd.read_csv("categorized_tweets.csv")

# Split data into train, eval, and test sets (70/15/15 split) with stratified sampling based on the label
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=10)
eval_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=10)

# Convert pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer corresponding to the model
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/mnt/scratch/lellaom/models")

# Tokenization function to tokenize and pad/truncate text inputs
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")

# Tokenize and shuffle the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True).shuffle(seed=10)
eval_dataset = eval_dataset.map(tokenize_function, batched=True).shuffle(seed=10)
test_dataset = test_dataset.map(tokenize_function, batched=True).shuffle(seed=10)

# Define function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get predicted class
    
    # Calculate precision, recall, F1-score, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Load the pre-trained model with the number of classification labels
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    cache_dir="/mnt/scratch/lellaom/models", 
    num_labels=2
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    eval_strategy="epoch",              # Evaluate at the end of each epoch
    logging_strategy='epoch',           # Log metrics at the end of each epoch
    save_strategy='epoch',              # Save model at the end of each epoch
    learning_rate=1e-5,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    load_best_model_at_end=True,        # Restore the best model (based on eval metric)
    metric_for_best_model="f1",         # Use F1-score to determine the best model
    greater_is_better=True
)

# Pad batches dynamically during training
data_collator = DataCollatorWithPadding(tokenizer)

# Initialize the Trainer API with model, data, and configurations
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Optional early stopping
)

# Train the model
trainer.train()

# Alternative: Resume training from checkpoint if needed
# trainer.train(resume_from_checkpoint=True)

# Output the path to the best model
print(f"Best model path: {trainer.state.best_model_checkpoint}")

# Evaluate model on the test set and print results
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(test_results)