import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, VOCAB_SIZE, MAX_SEQ_LENGTH, DEVICE
from utils import preprocess_data_csv
import json

# [LANGUAGE VOCABULARY]
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_TOKEN: "<PAD>",
            SOS_TOKEN: "<SOS>",
            EOS_TOKEN: "<EOS>"
        }
        self.n_words = 3  # Count PAD, SOS and EOS

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
    
    def get_vocabulary(self):
        # Sort words by frequency (high to low) and take the top max_words
        sorted_words = sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)
        limited_vocab = sorted_words[:VOCAB_SIZE]

        # Create a dictionary with the token as key and index as value
        vocab = {word: self.word2index[word] for word, _ in limited_vocab}
        return vocab
    
    def save_lang(self, filepath):
        vocab_data = {
            'word2index': self.word2index,
            'word2count': self.word2count,
            'index2word': self.index2word,
            'n_words': self.n_words
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)

    def load_vocabulary(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.word2index = vocab_data['word2index']
        self.word2count = vocab_data['word2count']
        self.index2word = vocab_data['index2word']
        self.n_words = vocab_data['n_words']

def read_langs(data, lang1='en', lang2='vi', reverse=False):
    """
    Reads and processes language data, returning language objects and pairs of sentences.
    Args:
        data (str): The raw data containing sentences in both languages.
        lang1 (str, optional): The code for the first language. Defaults to 'en'.
        lang2 (str, optional): The code for the second language. Defaults to 'vi'.
        reverse (bool, optional): If True, reverses the language pairs. Defaults to False.
    Returns:
        tuple: A tuple containing the input language object, output language object, and a list of sentence pairs.
    """
    print("Reading lines...")
    
    # Split every line into pairs and normalize
    pairs = preprocess_data_csv(data)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

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

def prepare_data(dir, lang1='en', lang2='vi', reverse=False):
    input_lang, output_lang, pairs = read_langs(dir, lang1, lang2, reverse)
    
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs, MAX_SEQ_LENGTH)
    
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
        
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs

# [DATALOADER]

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def get_dataloader(dir, batch_size):
    input_lang, output_lang, pairs = prepare_data(dir)
    
    n = len(pairs)
    input_ids = np.zeros((n, MAX_SEQ_LENGTH + 2), dtype=np.int32)
    target_ids = np.zeros((n, MAX_SEQ_LENGTH + 2), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = [SOS_TOKEN] + indexes_from_sentence(input_lang, inp) + [EOS_TOKEN]
        tgt_ids = [SOS_TOKEN] + indexes_from_sentence(output_lang, tgt) + [EOS_TOKEN]
        
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(DEVICE),
                               torch.LongTensor(target_ids).to(DEVICE))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

if __name__ == "__main__":
    from pprint import pprint
    input_lang, output_lang, train_dataloader = get_dataloader('data/raw/test.csv', 2)
    for batch in train_dataloader:
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
