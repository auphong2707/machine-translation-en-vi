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
    dataframes = [pd.read_csv(dir) for dir in dirs]
    
    # Drop 'source' column
    dataframes = [df.drop(columns=["source"]) for df in dataframes]
    
    # Ensure columns are named correctly
    dataframes = [df.rename(columns={"en": "source", "vi": "target"}) for df in dataframes]
    
    # Convert DataFrame to Hugging Face Dataset
    datasets = [Dataset.from_pandas(df) for df in dataframes]
    
    # Filter out long sentences
    print("Filtering out long sentences...")
    datasets = [dataset.filter(filter_long_sentences) for dataset in datasets]
    print("Filtered dataset successfully!\n")
    
    # Apply tokenization to the dataset
    print("Tokenizing dataset...")
    datasets = [dataset.map(preprocess_function, batched=True) for dataset in datasets]
    print("Tokenized dataset successfully!\n")
    
    # Set the format of the dataset
    print("Setting dataset format...")
    train_dataset, val_dataset, test_dataset = datasets
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print("Set dataset format successfully!\n")
    
    # Print information about the dataset
    print("Created dataset successfully!")
    print("Train dataset:", len(train_dataset))
    print("Validation dataset:", len(val_dataset))
    print("Test dataset:", len(test_dataset))
    
    # Return the Dataset
    return train_dataset, val_dataset, test_dataset, tokenizer

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset, tokenizer = get_dataset([TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR])
    for batch in DataLoader(train_dataset, batch_size=2, shuffle=True):
        print(batch)
        break
    
    for batch in DataLoader(val_dataset, batch_size=2, shuffle=True):
        print(batch)
        break
    
    for batch in DataLoader(test_dataset, batch_size=2, shuffle=True):
        print(batch)
        break