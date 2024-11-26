from datasets import Dataset
import pandas as pd
from transformers import MarianTokenizer
from torch.utils.data import DataLoader

import sys
sys.path.append('.')
from config import *

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")

# Filter function to remove long sentences
def filter_long_sentences(example, max_length=MAX_SEQ_LENGTH):
    source_length = len(tokenizer.tokenize(example["source"]))
    target_length = len(tokenizer.tokenize(example["target"]))
    return source_length <= max_length and target_length <= max_length

# Preprocessing function for tokenization
def preprocess_function(batch):
    inputs = tokenizer(batch["source"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    targets = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs["labels"] = targets["input_ids"]
    return inputs

def get_dataset(dirs: str) -> Dataset:
    print("Creating dataset...\n")
    
    # Load the CSV file into a DataFrame
    train_df = pd.read_csv(dirs[0])
    val_df = pd.read_csv(dirs[1])
    
    # Drop 'source' column
    train_df = train_df.drop(columns=["source"])
    val_df = val_df.drop(columns=["source"])
    
    # Ensure columns are named correctly
    train_df = train_df.rename(columns={"en": "source", "vi": "target"})
    val_df = val_df.rename(columns={"en": "source", "vi": "target"})
    
    # Convert DataFrame to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Filter out long sentences
    print("Filtering out long sentences...")
    train_dataset = train_dataset.filter(filter_long_sentences)
    val_dataset = val_dataset.filter(filter_long_sentences)
    print("Filtered dataset successfully!\n")
    
    # Apply tokenization to the dataset
    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    print("Tokenized dataset successfully!\n")
    
    # Set the format of the dataset
    print("Setting dataset format...")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print("Set dataset format successfully!\n")
    
    # Print information about the dataset
    print("Created dataset successfully!")
    print("Train dataset:", len(train_dataset))
    print("Validation dataset:", len(val_dataset), '\n')
    
    # Return the Dataset
    return train_dataset, val_dataset, tokenizer
