# machine-translation-en-vi
This project aims to develop a Machine Translation model that translates from English to Vietnamese using Deep Learning techniques. All content in this project is intended for the IT3320E - Introduction to Deep Learning module at Hanoi University of Science and Technology.

# Table of Contents

- [machine-translation-en-vi](#machine-translation-en-vi)
- [Table of Contents](#table-of-contents)
- [Dataset](#dataset)
- [Architecture](#architecture)
  - [RNNs](#rnns)
    - [Key Features](#key-features)
    - [Advantages](#advantages)
  - [RNNs with Attention Mechanism](#rnns-with-attention-mechanism)
    - [Key Features](#key-features-1)
    - [Advantages](#advantages-1)
  - [Transformer Architecture](#transformer-architecture)
    - [Key Features](#key-features-2)
    - [Advantages](#advantages-2)
- [How to use our code](#how-to-use-our-code)
  - [Installation](#installation)
  - [Training](#training)
  - [Download the models and User Interface](#download-the-models-and-user-interface)
  - [Infer file](#infer-file)
- [Trained models](#trained-models)
- [References](#references)

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

# How to use our code
## Installation
All of our experiments we did so far is runned on **Kaggle** environment. Therefore, we recommend you to use **Kaggle** to run our code instead of installing the `requirements.txt` file.

However, if you want to run our code on your local machine, you can install the required packages by running the following command (There is a chance that you may encounter some errors due to the compatibility of the packages):
```bash
pip install -r requirements.txt
```
## Training
To train the a model, first you need to create a branch from the `main` branch and publish it:
```bash
git checkout -b <branch_name>
git push origin <branch_name>
```
Next step, you need to change the `config.py` file to specify the model you want to train, the dataset you want to use, the training parameters, etc. Please remember to commit and publish the changes you made to the `config.py` file.

After that, you need to create a new notebook on **Kaggle**, clone the repository, switch to the branch you created by running the following commands:
```bash
!git clone https://github.com/auphong2707/machine-translation-en-vi.git
%cd machine-translation-en-vi
!git checkout <branch_name>
```
Finally, you can run the training script by running the following command:
- For RNN:
  ```bash
   
  ```
- For RNN with Attention Mechanism:
  ```bash
  !python main_rnn_attn.py --huggingface_token hf_token
  ```
- For Transformer:
  ```bash
  !python main_optimus_prime.py --huggingface_token hf_token --wandb-token wandb_token
  ```
As you can see, you need the `hf_token` to store the model on the Hugging Face model hub, please also remember to change the repository in `main` file to your own repository.

Addtionally, for the Transformer model, you need to specify the `wandb_token` to log the training process on the Weights & Biases platform. Also, you need to change the `wandb_project` in the `config.py` file to your own project.

**Our code will automatically save the model each epochs to continue training later.**
**After training, the model will be saved in the Hugging Face**

## Download the models and User Interface
After training the model, you can download the model from the Hugging Face model hub by running the following command:
Please remember to change the `REPO_ID` in `download_model.py` file to your own repository.
```bash
python download_model.py --hf_dir model_directory_on_hf/* --local_dir ./trained_models/type_of_model
```
For examples, I will use our trained models:
```bash
python download_model.py --hf_dir experiment_0_3/* --local_dir ./trained_models/rnn
python download_model.py --hf_dir experiment_1_1/* --local_dir ./trained_models/rnn_attention
python download_model.py --hf_dir experiment_2_0/best_model/* --local_dir ./trained_models/transformer
```

Then after that, you can run the User Interface by running the following command:
```bash
python flask_app/app.py
```
The User Interface will be available at `localhost:5000`.

## Infer file
If you want to use infer file instead of the User Interface, first you need to download the trained model from the Hugging Face model hub by running the following command:
```bash
python download_model.py --hf_dir model_directory_on_hf/* --local_dir ./trained_models/type_of_model
```
For examples, I will use transformer model:
```bash
python download_model.py --hf_dir experiment_2_0/best_model/* --local_dir ./trained_models/transformer
```

Then you can run the infer file by running the following command (model_type can be `rnn`, `rnn_attention`, `transformer`):
```bash
python infer.py --model_type model_type --text "text_to_translate"
```

For examples, I will use the transformer model:
```bash
python infer.py --model_type transformer --text "I am a cat"
```

# Trained models
The trained models are available on the Hugging Face model hub: [Hugging Face model hub](https://huggingface.co/auphong2707/machine-translation-en-vi)

You can see so many models here. However, you will only need to focus on the models that have the following names: experiment_0_3, experiment_1_1, experiment_2_0,... (Pattern: experiment_{type_of_model}_{experiment_number}). Type of model can be `rnn`, `rnn_attention`, `transformer` correspond to 0, 1, 2 respectively.

# References
- [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)