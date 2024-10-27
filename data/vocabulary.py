import re
import random
import numpy as np
import pandas as pd

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def getVocabulary(self, max_words=50000):
        # Sort words by frequency (high to low) and take the top max_words
        sorted_words = sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)
        limited_vocab = sorted_words[:max_words]

        # Create a dictionary with the token as key and index as value
        vocab = {word: self.word2index[word] for word, _ in limited_vocab}
        return vocab

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-ZàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+", r" ", s)
    return s.strip()

def preprocessData(dir, lang1='en', lang2='vi'):
    data = pd.read_csv(dir, encoding='utf-8')
    data = data.applymap(lambda x: normalizeString(x) if isinstance(x, str) else x)
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.split().str.join('_')
    pairs = data.values.tolist()
    return pairs

def readLangs(data, lang1='en', lang2='vi', reverse=False):
    print("Reading lines...")
    
    # Split every line into pairs and normalize
    pairs = preprocessData(data, lang1, lang2)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 100

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(dir, lang1='en', lang2='vi', reverse=False):
    input_lang, output_lang, pairs = readLangs(dir, lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('data/raw/train.csv')
print(input_lang.getVocabulary())
print(output_lang.getVocabulary())
