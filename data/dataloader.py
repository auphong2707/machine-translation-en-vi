import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import json

import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from helper import preprocess_data_csv

# [LANGUAGE VOCABULARY]
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_TOKEN: "<PAD>",
            SOS_TOKEN: "<SOS>",
            EOS_TOKEN: "<EOS>",
            UNK_TOKEN: "<UNK>"
        }
        self.n_words = 4  # Count PAD, SOS, EOS, UNK

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def trim_lang(self, max_vocab_size=VOCAB_SIZE):
        """Trims the vocabulary to retain only the most frequent words up to max_vocab_size."""
        # Sort words by frequency (high to low)
        sorted_words = sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)

        # Create a new list of words to keep
        top_words = sorted_words[:max_vocab_size - 4]

        # Create new dictionaries to store trimmed vocabulary
        new_word2index = {}
        new_word2count = {}
        new_index2word = {PAD_TOKEN: "<PAD>", SOS_TOKEN: "<SOS>", EOS_TOKEN: "<EOS>", UNK_TOKEN: "<UNK>"}
        
        # Reset n_words
        n_words = 4  # Count for PAD, SOS, EOS, UNK
        
        # Add the top words to the new dictionaries
        for word, count in top_words:
            new_word2index[word] = n_words
            new_word2count[word] = count
            new_index2word[n_words] = word
            n_words += 1
        
        # Update instance variables
        self.word2index = new_word2index
        self.word2count = new_word2count
        self.index2word = new_index2word
        self.n_words = n_words  # Update n_words
        
    def get_word(self, index):
        return self.index2word[index]
    
    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return UNK_TOKEN
    
    def save(self, filepath):
        """Saves the Lang object to a file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(self.__dict__, file)
    
    def load(self, filepath):
        """Loads the Lang object from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        with open(filepath, 'rb') as file:
            self.__dict__ = pickle.load(file)

def filter_pairs(pairs, max_length):
    """
    Filters a list of sentence pairs based on a maximum length criterion.
    Args:
        pairs (list of tuple): A list of tuples where each tuple contains two sentences (str).
    Returns:
        list of tuple: A list of tuples containing sentence pairs that meet the length criterion.
    """
    
    def filter_pair(p, max_length):
        return len(p[0].split(' ')) <= max_length and len(p[1].split(' ')) <= max_length
    
    return [pair for pair in pairs if filter_pair(pair, max_length)]

def prepare_data(dirs, lang1='en', lang2='vi', reverse=False):
    # [READ DATA]
    print("Reading lines...\n")
    
    # Split every line into pairs and normalize
    pairs_list = [preprocess_data_csv(dir) for dir in dirs]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    # [FILTER PAIRS]
    for i in range(len(pairs_list)):
        print("Read %s sentence pairs" % len(pairs_list[i]))
        pairs_list[i] = filter_pairs(pairs_list[i], MAX_SEQ_LENGTH - 1)
        print("Trimmed to %s sentence pairs\n" % len(pairs_list[i]))
        
    
    print("Counting words...")
    for pairs in pairs_list:
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print()
    
    print("Trimmed words to VOCAB_SIZE...")
    input_lang.trim_lang()
    print("Trimmed Input Lang successfully:", input_lang.name, input_lang.n_words)
    output_lang.trim_lang()
    print("Trimmed Output Lang successfully:", output_lang.name, output_lang.n_words)
    print()

    return input_lang, output_lang, pairs_list

# [DATALOADER]

def indexes_from_sentence(lang: Lang, sentence: str):
    return [lang.get_index(word) for word in sentence.split(' ')]

def get_dataloader(dirs, batch_size):
    input_lang, output_lang, pairs_list = prepare_data(dirs)
    dataloaders = []
    
    for pairs in pairs_list:
        n = len(pairs)
        input_ids = np.zeros((n, MAX_SEQ_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, MAX_SEQ_LENGTH), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(pairs):
            inp_ids = indexes_from_sentence(input_lang, inp) + [EOS_TOKEN]
            tgt_ids = indexes_from_sentence(output_lang, tgt) + [EOS_TOKEN]
            
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        data = TensorDataset(torch.LongTensor(input_ids).to(DEVICE),
                            torch.LongTensor(target_ids).to(DEVICE))

        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=True)
        dataloaders.append(dataloader)
    
    return input_lang, output_lang, dataloaders

if __name__ == "__main__":
    from pprint import pprint
    input_lang, output_lang, (train_dataloader, val_loader, test_loader) \
    = get_dataloader([TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR], 2)
    
    for batch in train_dataloader:
        print(type(batch[0]), type(batch[1]))
        pprint(batch)
        break
    
    for batch in train_dataloader:
        input_batch, target_batch = batch
        for i in range(input_batch.size(0)):
            input_sentence = ' '.join([input_lang.index2word[idx.item()] for idx in input_batch[i] if idx.item() in input_lang.index2word])
            target_sentence = ' '.join([output_lang.index2word[idx.item()] for idx in target_batch[i] if idx.item() in output_lang.index2word])
            print(f"Input: {input_sentence}")
            print(f"Target: {target_sentence}")
        break
