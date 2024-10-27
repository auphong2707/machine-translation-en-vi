from ..utils.config import VOCAB_SIZE, SOS_TOKEN, EOS_TOKEN, MAX_SEQ_LENGTH
from ..utils.utils import preprocess_data_csv
import json

# [LANGUAGE VOCABULARY]
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: SOS_TOKEN, 1: EOS_TOKEN}
        self.n_words = 2  # Count SOS and EOS

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
        return len(p[0].split(' ')) < max_length and \
            len(p[1].split(' ')) < max_length
    
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

if __name__ == "__main__":
    input_lang, output_lang, pairs = prepare_data('data/raw/train.csv')
    print(input_lang.get_vocabulary())
    print(output_lang.get_vocabulary())
