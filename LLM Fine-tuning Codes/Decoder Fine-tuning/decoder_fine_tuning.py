# Import required libraries
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

# Function to check if a directory exists and delete it if it does
def check_and_delete(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)
        print(f"Directory '{dir}' has been deleted.")
    else:
        print(f"Directory '{dir}' does not exist.")

# Function to prepare the data in chat format for the model
def data_preparation(df):
    # Example titles and responses for few-shot learning
    title1 = "The Role of AI in Shaping Public Discourse"
    title2 = "Media Representations of Racial Identity: Challenges and Opportunities"
    res1 = "<code>0</code><reason>The title focuses on AI and its impact on public discourse, without explicit reference to race, racial identity, or related topics.</reason>"
    res2 = "<code>1</code><reason>The title explicitly addresses 'racial identity,' directly connecting to discussions of race and its representation in the media.</reason>"
    
    # Instructions for the classification task
    info = ["Carefully analyze the following research title and determine whether the research topic is race-related or not.",
            "Research is considered race-related if it explicitly mentions race or racial groups, racism, stereotypes, racial privilege, racial bias, or marginalization, or interactual perspectives in the context of race (e.g., diversity, culture, intercultural relations, or intercultural research), ethnic groups or ethnicity-based experiences in the context of citizenship (e.g., immigrants, migrants), and people (e.g., Egyptian, African, Chinese).",
            "Race-related does not include geographic locations (e.g., Africa) unless tied to people or identity, religion (e.g., Muslims, Hindu) unless race/ethnicity is also explicitly mentioned, or individuals (e.g., Obama) unless their race or racial identity is explicitly mentioned.",
            "Return 0 in code tags if the research title is not related to race, and 1 if it is related to race.",
            "Additionally, provide a brief explanation in the reason tags for the classification."
            ]
    info = ' '.join(info)
    info += '\n\nResearch Title:\n'

    # Create different versions of the prompt
    info1 = info+title1
    info2 = info+title2
    info3 = info
    
    # Create the initial chat template with examples
    chat = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to classify whether a title is related to race or not."},
        {"role": "user", "content": info1},
        {"role": "assistant", "content": res1},
        {"role": "user", "content": info2},
        {"role": "assistant", "content": res2},
        ]
    response_template = " <code>[CODE]</code><reason>[REASON]</reason>"

    # Process each row in the dataframe to create chat examples
    chat_list = []
    for idx, row in df.iterrows():
        title = row['Presentation Title']
        info_dict = {}
        info_dict["role"] = "user"
        info = info3+str(title)
        info_dict["content"] = info
        response = response_template.replace('[CODE]', str(row['Code']))
        response = response.replace('[REASON]', row['Reason'])
        response_dict = {"role": "assistant", "content": response}
        new_chat = chat.copy()
        new_chat.append(info_dict)
        new_chat.append(response_dict)
        chat_list.append(tokenizer.apply_chat_template(new_chat, tokenize=False))
    
    df['Chat'] = chat_list
    return df

# Function to tokenize the chat examples
def tokenize_function(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    max_length = 1024

    for full_text in examples["Chat"]:
        # Split the text into prompt and response parts
        split_text = full_text.rsplit("[/INST]", maxsplit=1)
        prompt_part = split_text[0] + "[/INST]"  
        tokenized_full = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)

        # Tokenize just the prompt part to calculate its length
        tokenized_prompt = tokenizer(prompt_part, add_special_tokens=False, truncation=False)
        prompt_length = len(tokenized_prompt["input_ids"])
        
        # Create labels, masking the prompt part
        labels = tokenized_full["input_ids"].copy()
        if prompt_length < max_length:
            labels[:prompt_length] = [-100] * prompt_length  # -100 is ignored in loss calculation
        else:
            pass
        
        # Helper function to pad or truncate sequences
        def pad_and_truncate(seq, pad_value, length):
            if len(seq) < length:
                return seq + [pad_value] * (length - len(seq))
            return seq[:length]
        
        # Process each component
        input_ids = pad_and_truncate(tokenized_full["input_ids"], tokenizer.pad_token_id, max_length)
        attention_mask = pad_and_truncate(tokenized_full["attention_mask"], 0, max_length)
        labels = pad_and_truncate(labels, -100, max_length)
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

# Set output directories
output_dir = "/mnt/scratch/lellaom/LLM Fine-tuning Codes/Decoder Fine-tuning/checkpoints"
logging_dir = "/mnt/scratch/lellaom/LLM Fine-tuning Codes/Decoder Fine-tuning/logs"

# Clean up directories if they exist
check_and_delete(output_dir)
check_and_delete(logging_dir)

# Load tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          cache_dir='/mnt/scratch/lellaom/models',
                                          padding_side="right",
                                          add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

# Configure quantization for memory-efficient training
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Use nested quantization
    bnb_4bit_quant_type="nf4",  # Use normalized float 4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype
)

# Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj"],  # Modules to apply LoRA to
    lora_dropout=0.2,  # Dropout rate
    bias="none",  # No bias
    task_type="CAUSAL_LM",  # Task type
)

# Load the pre-trained model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically distribute model across devices
    cache_dir='/mnt/scratch/lellaom/models',  # Cache directory
)

# Prepare model for k-bit training and apply LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Load and prepare the dataset
df = pd.read_csv("Coded Sample with Reason.csv")
df = data_preparation(df)

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df.iloc[0:200], test_size=0.24, stratify=df.iloc[0:200]['Code'], random_state=10)
train_df = pd.concat([train_df, df.iloc[200:]])  # Add remaining data to training set

# Save datasets to CSV
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

# Drop unnecessary columns
train_df = train_df.drop(columns=['Presentation Title', 'Code', 'Reason'])
test_df = test_df.drop(columns=['Presentation Title', 'Code', 'Reason'])

# Convert pandas DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  # Output directory
    per_device_train_batch_size=1,  # Batch size per device during training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=4,  # Number of steps before performing optimization
    num_train_epochs=3,  # Number of training epochs
    eval_strategy="epoch",  # Evaluation strategy
    save_strategy="epoch",  # Model saving strategy
    logging_dir=logging_dir,  # Directory for logs
    learning_rate=3e-5,  # Learning rate
    lr_scheduler_type="cosine",  # Learning rate scheduler
    bf16=True,  # Use bfloat16 precision
    push_to_hub=False,  # Don't push to Hugging Face Hub
    load_best_model_at_end=True,  # Load the best model at the end
    metric_for_best_model="eval_loss",  # Metric for best model selection
    greater_is_better=False,  # Lower eval_loss is better
    warmup_ratio=0.1,  # Warmup ratio
    weight_decay=0.001,  # Weight decay
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    processing_class=tokenizer,
)

# Start training
trainer.train()

# Save the best model
output_dir = "/mnt/scratch/lellaom/LLM Fine-tuning Codes/Decoder Fine-tuning/best_model"
check_and_delete(output_dir)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)