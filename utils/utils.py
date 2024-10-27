import os
import torch
import numpy as np
import random
import re
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def set_seed(seed=42):
    """Set random seed for reproducibility across numpy, random, and torch.

    Args:
        seed (int): Seed value to set for reproduction. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    filepath='checkpoint.pth'):
    """Save model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): Optmizer state to save. 
        epoch (int): Current epoch.
        filepath (str): Path to save the checkpoint. Defaults to 'checkpoint.pth'.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    filepath='checkpoint.pth'):
    """Load model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        filepath (str): Path to load the checkpoint from. Defaults to 'checkpoint.pth'.
        
    Returns:
        int: The epoch to resume training from.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filepath}, resuming training from epoch {epoch}")
    return epoch

def count_parameters(model: torch.nn.Module):
    """Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): The model to count parameters for.
        
    Returns:
        int: The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_predictions(predictions, filepath='predictions.txt'):
    """Save model predictions to a file

    Args:
        predictions (list): List of predictions to save.
        filepath (str): Path to save predictions. Defaults to 'predictions.txt'.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {filepath}")
    
def compute_bleu(reference, hypothesis):
    """Compute the BLEU score between a reference and a hypothesis sentence.

    Args:
        reference (list of list of str): The reference sentences.
        hypothesis (list of str): The hypothesis sentence.

    Returns:
        float: The BLEU score.
    """
    smoothie = SmoothingFunction().method
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)

# [TEXT PREPROCESSING]
def normalize_string(string: str) -> str:
    """Normalize a string by converting to lowercase and removing non-letter characters.

    Args:
        string (str): Input string to normalize.

    Returns:
        str: Normalized string.
    """
    string = string.lower().strip()
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-ZàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+", r" ", string)
    return string.strip()

def preprocess_data_csv(dir: str) -> list:
    """Reads and preprocesses data from a CSV file.
    
    Args:
        dir (str): Path to the CSV file.
        
    Returns:
        list: A list of sentence pairs.
    """
    data = pd.read_csv(dir, encoding='utf-8')
    data = data.applymap(lambda x: normalize_string(x) if isinstance(x, str) else x)
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.split().str.join('_')
    pairs = data.values.tolist()
    return pairs