import torch
import torch
from models.seq2seq import Seq2SeqGRU
from data.dataloader import get_dataloader, Lang
from torch.nn.functional import softmax
import pickle
import numpy as np

import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from collections import Counter
import math

def calculate_bleu(reference, candidate, max_n=4):
    """
    Calculate the BLEU score between two ordered lists of words.
    
    Args:
        reference (list): List of words in the reference sentence.
        candidate (list): List of words in the candidate (generated) sentence.
        max_n (int): Maximum n-gram order to use (usually 4 for BLEU-4).
        
    Returns:
        float: The BLEU score.
    """
    
    def n_gram_precision(reference, candidate, n):
        ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)])
        cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)])
        
        match_count = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items())
        total_count = sum(cand_ngrams.values())
        
        return match_count / total_count if total_count > 0 else 0
    
    precisions = [n_gram_precision(reference, candidate, n) for n in range(1, max_n + 1)]
    
    ref_len = len(reference)
    cand_len = len(candidate)
    brevity_penalty = math.exp(1 - ref_len / cand_len) if cand_len < ref_len else 1
    
    bleu_score = brevity_penalty * math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n)
    
    return bleu_score

def test_model(model, test_loader, output_lang):
    all_outputs = []  
    batch_cnt = 0
    with torch.no_grad():  
        for batch in test_loader:
            inputs, targets = batch 
            inputs = inputs.to(DEVICE)  

            # Forward pass through the model
            outputs = model(inputs)

            probabilities = softmax(outputs, dim=-1)
            predicted_token_id = torch.argmax(probabilities, dim=-1)
            for corpus in predicted_token_id:
                decoded_words = [output_lang.get_word(token_id) for token_id in corpus.tolist()]
                all_outputs.append(decoded_words) 
    return all_outputs

def tester(dir, output_lang, test_loader):
    model = Seq2SeqGRU().to(DEVICE)  

    checkpoint_path = 'results/{}/checkpoint.pth'.format(dir) 
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict']) 

    model.eval()
    test_outputs = test_model(model, test_loader, output_lang)

    ground_truth = []

    for batch in test_loader:
        for corpus in batch:
            for sentence in corpus:
                decoded_words = [output_lang.get_word(token_id) for token_id in sentence.tolist()]
                ground_truth.append(decoded_words) 

    bleu_score = []
    for i, (output, test) in enumerate(zip(test_outputs, ground_truth)):
        bleu_score.append(calculate_bleu(output, test))

    mean_bleu = np.mean(bleu_score)
    variance_bleu = np.var(bleu_score)
    num_elements = len(bleu_score)

    with open(r"results/{}/bleu_score_stats.txt".format(dir), "w") as f:
        f.write(f"Mean BLEU score: {mean_bleu}\n")
        f.write(f"Variance of BLEU score: {variance_bleu}\n")
        f.write(f"Number of elements in BLEU score array: {num_elements}\n")
