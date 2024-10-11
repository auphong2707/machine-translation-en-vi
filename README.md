# machine-translation-en-vi
This project aims to develop a Machine Translation model that translates from English to Vietnamese using Deep Learning techniques. All content in this project is intended for the IT3320E - Introduction to Deep Learning module at Hanoi University of Science and Technology.

# Dataset
Dataset: https://huggingface.co/datasets/ncduy/mt-en-vi/tree/main

This dataset is already splitted in 3 parts: **Train, Validation, Test**

|       | Train      | Validation | Test   |
|-------|------------|------------|--------|
| Number of examples | 2,884,451  | 11,316  | 11,225 |

<br>

Each example contains 3 features:
- *en*: The English sentence.
- *vi*: The corresponding Vietnamese sentence.
- *source*: The source from which this example is taken.

# Embedding
## One-hot encoding
**One-hot encoding** is a technique used to convert categorical data into a binary vector format, where each category is represented as a unique vector with all elements set to 0 except for the one that corresponds to the category, which is set to 1. This is commonly used in machine learning models to represent categorical variables, making them easier to process.

### Example
Suppose we have three categories: `apple`, `banana`, and `cherry`.

- Assign each category an index:
  - `apple` = 0
  - `banana` = 1
  - `cherry` = 2

- Using **one-hot encoding**, each category is converted into a binary vector:
  - `apple`: `[1, 0, 0]`
  - `banana`: `[0, 1, 0]`
  - `cherry`: `[0, 0, 1]`

### Use Case
For a dataset where the `fruit` column contains `apple`, `banana`, and `cherry`, instead of storing the text, you can use one-hot encoded vectors to represent each value. This makes the data suitable for input into machine learning algorithms that require numerical input.
## Word2Vec

**Word2Vec** is a popular algorithm used to create vector representations of words. It transforms words into dense vectors of fixed size, where semantically similar words are mapped to nearby points in the vector space. This technique captures the context of words based on their surrounding words in a sentence, making it useful for various natural language processing tasks.

### How It Works

Word2Vec operates using two main architectures:

1. **Continuous Bag of Words (CBOW)**: This model predicts the target word based on its context (the surrounding words). For example, given the context words "The" and "sat," it predicts the target word "cat" in the sentence "The cat sat on the mat."

2. **Skip-gram**: This model does the opposite of CBOW. It uses the target word to predict the context words. For instance, given the target word "cat," it aims to predict the surrounding words "The," "sat," "on," and "the."

### Example

Suppose we have the following sentences:

- "The cat sat on the mat."
- "Dogs are great pets."

Using Word2Vec, we can generate vector representations for the words:

- `cat`: `[0.12, 0.45, -0.78, ...]`
- `dog`: `[0.34, -0.56, 0.12, ...]`
- `great`: `[0.67, -0.14, 0.85, ...]`

Each word is represented as a dense vector that captures its meaning and context relative to other words.

### Use Case

Word2Vec can be applied in various natural language processing tasks, such as:

- **Semantic Analysis**: Understanding the meanings of words based on their context.
- **Text Classification**: Representing words as vectors for use in classification algorithms.
- **Recommendation Systems**: Finding similar items based on textual data.

By converting words into numerical vectors, Word2Vec enables machine learning models to analyze and understand textual data more effectively.

## BERT (Bidirectional Encoder Representations from Transformers)

**BERT** is a state-of-the-art transformer-based model developed by Google for natural language processing (NLP) tasks. Unlike traditional models that process text in a unidirectional manner (left-to-right or right-to-left), BERT uses a bidirectional approach, allowing it to consider the full context of a word by looking at the words that come before and after it in a sentence. This results in a deeper understanding of language nuances and relationships.

### How It Works

BERT is pre-trained on vast amounts of text using two primary tasks:

1. **Masked Language Model (MLM)**: During training, some words in the input are masked (replaced with a special token). The model learns to predict these masked words based on their context. For example, in the sentence "The cat sat on the [MASK]," the model would learn to predict "mat."

2. **Next Sentence Prediction (NSP)**: This task helps the model understand the relationship between sentences. Given two sentences, BERT learns to predict whether the second sentence follows the first in the text. 

### Example

Suppose we have the following sentences:

- Sentence 1: "The cat sat on the mat."
- Sentence 2: "It was very comfortable."

Using BERT, we can generate context-aware embeddings for each token in a sentence. For instance, the word "cat" might be represented as:
- `cat`: `[0.23, -0.67, 0.91, ...]`

The vector representation captures not just the meaning of "cat," but also its context within the sentence.

### Use Case

BERT can be applied in various natural language processing tasks, such as:

- **Text Classification**: Categorizing documents based on their content.
- **Question Answering**: Providing answers to questions based on context.
- **Named Entity Recognition (NER)**: Identifying and classifying entities in text.

By leveraging its bidirectional context and powerful embeddings, BERT enables significant advancements in understanding and processing human language.

# Architecture
## RNNs

**Recurrent Neural Networks (RNNs)** are widely used in machine translation tasks due to their ability to handle sequential data. In this context, RNNs process input sentences word by word, capturing the context and dependencies between words to generate translated output in another language.

### Key Features

- **Encoder-Decoder Architecture**: RNNs are commonly used in an encoder-decoder framework, where the encoder processes the source sentence and compresses it into a fixed-size context vector, while the decoder generates the translated sentence from this context.
- **Contextual Understanding**: By maintaining a hidden state, RNNs can remember relevant information from the source sentence, enabling them to produce more accurate translations.

### Advantages

- RNNs can model variable-length input and output sequences, making them well-suited for translation tasks where sentence lengths may vary.
- They capture sequential dependencies, allowing for better handling of word order and context.

However, traditional RNNs may struggle with long-range dependencies, which can affect translation quality. Techniques like attention mechanisms and advanced architectures such as LSTM and GRU have been developed to improve performance in machine translation.

## RNNs with Attention Mechanism

**Recurrent Neural Networks (RNNs) with Attention Mechanism** enhance traditional RNN architectures by allowing the model to focus on specific parts of the input sequence when generating translations. This approach significantly improves the quality of machine translation by addressing the limitations of standard RNNs in handling long-range dependencies.

### Key Features

- **Attention Mechanism**: The attention mechanism computes a set of attention weights that indicate the importance of each word in the input sequence at every decoding step. This allows the model to selectively focus on relevant words when generating each word in the output sequence.
  
- **Encoder-Decoder Architecture**: Similar to standard RNNs, this model uses an encoder-decoder framework. The encoder processes the entire source sentence and creates a context vector, while the decoder uses the attention weights to generate the translated sentence, leveraging the context more effectively.

### Advantages

- **Improved Contextual Understanding**: By dynamically focusing on different parts of the input sequence, the model captures relationships and dependencies more effectively, leading to more accurate translations.
  
- **Handling Long Sentences**: The attention mechanism alleviates the vanishing gradient problem often faced by traditional RNNs, making it easier to handle long sentences and complex sentence structures.

Overall, RNNs with attention mechanisms are a powerful advancement in machine translation, providing enhanced performance and greater flexibility in capturing semantic meaning across different languages.

## Transformer Architecture

The **Transformer** is a revolutionary neural network architecture introduced by Vaswani et al. in the paper "Attention is All You Need." It is specifically designed to handle sequential data, making it particularly effective for machine translation tasks. Unlike traditional RNN-based models, the Transformer relies entirely on self-attention mechanisms and feedforward neural networks, eliminating the need for recurrent connections.

### Key Features

- **Self-Attention Mechanism**: The Transformer uses self-attention to weigh the significance of different words in a sequence, allowing it to capture relationships between words regardless of their position. This enables the model to consider the entire input sequence simultaneously, rather than one step at a time.

- **Encoder-Decoder Structure**: The Transformer consists of an encoder that processes the input sentence and a decoder that generates the translated output. Both the encoder and decoder are composed of multiple layers of self-attention and feedforward networks.

- **Positional Encoding**: Since Transformers do not have a built-in notion of sequence order, they use positional encodings to inject information about the position of each word in the sequence.

### Advantages

- **Parallelization**: The architecture allows for parallel processing of input data, significantly speeding up training and inference times compared to RNNs.
  
- **Handling Long Dependencies**: The self-attention mechanism effectively captures long-range dependencies in the data, improving translation quality for complex sentences.

- **Scalability**: Transformers can be scaled up by adding more layers and attention heads, resulting in models like BERT and GPT that achieve state-of-the-art performance on a wide range of NLP tasks.

Overall, the Transformer architecture represents a significant advancement in the field of machine translation, enabling more accurate and efficient processing of natural language.

# References
- [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)