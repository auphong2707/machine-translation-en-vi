import importlib
import torch
from transformers import MarianMTModel, MarianTokenizer

from data.dataloader import Lang
from models.seq2seq import Seq2SeqRNN, Seq2SeqRNNAttn
from utils.helper import load_checkpoint
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Machine Translation Inference")
    parser.add_argument('--model_type', type=str, choices=['rnn', 'rnn_attention', 'transformer'], required=True, help="Type of model to use for translation")
    parser.add_argument('--text', type=str, required=True, help="Text to translate")
    return parser.parse_args()

args = parse_args()

# Load Transformer model and tokenizer
def load_transformer_model():
    """Load the Transformer model and tokenizer from the local directory."""
    model_path = "./trained_models/transformer"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path)
        print("Transformer model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Transformer model: {e}")
        return None, None

# Load RNN + Attention model
def load_config(config_path):
    """Load a Python configuration file dynamically."""
    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        print("Configuration loaded successfully.")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def load_rnn_attn_model():
    """Load the RNN with Attention model and its associated language objects."""
    model_path = f"./trained_models/rnn_attention/best.pth"
    input_lang_path = f"./trained_models/rnn_attention/input_lang.pkl"
    output_lang_path = f"./trained_models/rnn_attention/output_lang.pkl"

    config = load_config("./trained_models/rnn_attention/config.py")

    try:
        # Load language objects
        input_lang = Lang("en")
        input_lang.load(input_lang_path)
        
        output_lang = Lang("vi")
        output_lang.load(output_lang_path)

        # Initialize the model using config parameters
        model = Seq2SeqRNNAttn(
            batch_size=1,
            max_seq_length =config.MAX_SEQ_LENGTH,
            num_layers=config.RNN_ATTN_NUM_LAYERS,
            input_size=config.VOCAB_SIZE,
            output_size=config.VOCAB_SIZE,
            embedding_size=config.RNN_ATTN_EMBEDDING_SIZE,
            hidden_size=config.RNN_ATTN_HIDDEN_SIZE,
            dropout_rate=config.RNN_ATTN_DROPOUT_RATE,
            encoder_bidirectional=config.RNN_ATTN_ENCODER_BIDIRECTIONAL,
            teacher_forcing_ratio=config.TEACHER_FORCING_RATIO,
            sos_token=config.SOS_TOKEN,
            device='cpu')

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        load_checkpoint(model, optimizer, model_path)

        # Load model weights
        model.eval()  # Set model to evaluation mode

        print("RNN with Attention model loaded successfully.")
        return model, input_lang, output_lang
    except Exception as e:
        print(f"Error loading RNN + Attention model: {e}")
        return None, None, None

def load_rnn_model():
    """Load the simple RNN model."""
    model_path = f"./trained_models/rnn/best.pth"
    input_lang_path = f"./trained_models/rnn/input_lang.pkl"
    output_lang_path = f"./trained_models/rnn/output_lang.pkl"

    config = load_config("./trained_models/rnn/config.py")

    try:
        # Load language objects
        input_lang = Lang("en")
        input_lang.load(input_lang_path)
        
        output_lang = Lang("vi")
        output_lang.load(output_lang_path)

        # Initialize the model using config parameters
        model = Seq2SeqRNN(
            batch_size=1,
            max_seq_length=config.MAX_SEQ_LENGTH,
            num_layers=config.RNN_NUM_LAYERS,
            input_size=config.VOCAB_SIZE,
            output_size=config.VOCAB_SIZE,
            embedding_size=config.RNN_EMBEDDING_SIZE,
            hidden_size=config.RNN_HIDDEN_SIZE,
            dropout_rate=config.RNN_DROPOUT_RATE,
            encoder_bidirectional=config.RNN_ENCODER_BIDIRECTIONAL,
            teacher_forcing_ratio=config.TEACHER_FORCING_RATIO,
            sos_token=config.SOS_TOKEN,
            device='cpu'
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        load_checkpoint(model, optimizer, model_path)

        # Load model weights
        model.eval()  # Set model to evaluation mode

        print("Simple RNN model loaded successfully.")
        return model, input_lang, output_lang
    except Exception as e:
        print(f"Error loading simple RNN model: {e}")
        return None, None, None

MODELS = {
    "rnn": {"name": "Simple RNN", "model": None},
    "rnn_attention": {"name": "RNN + Attention", "model": None, "input_lang": None, "output_lang": None},
    "transformer": {"name": "Transformer", "model": None, "tokenizer": None},
}
MODELS["transformer"]["model"], MODELS["transformer"]["tokenizer"] = load_transformer_model()
MODELS["rnn_attention"]["model"], MODELS["rnn_attention"]["input_lang"], MODELS["rnn_attention"]["output_lang"] = load_rnn_attn_model()
MODELS["rnn"]["model"], MODELS["rnn"]["input_lang"], MODELS["rnn"]["output_lang"] = load_rnn_model()

def translate_with_transformer(text):
    """Translate text using the Transformer."""
    tokenizer = MODELS["transformer"]["tokenizer"]
    model = MODELS["transformer"]["model"]
    if not tokenizer or not model:
        return "Transformer model is not loaded."

    tokens = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_with_rnn_attention(text):
    """Translate text using the RNN with Attention model."""
    model = MODELS["rnn_attention"]["model"]
    input_lang = MODELS["rnn_attention"]["input_lang"]
    output_lang = MODELS["rnn_attention"]["output_lang"]

    if not model or not input_lang or not output_lang:
        return "RNN with Attention model is not loaded."

    tokens = [input_lang.get_index(word) for word in text.split(" ")]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        translated_indices = output.argmax(dim=-1).squeeze().tolist()
        translated_sentence = " ".join(output_lang.get_word(idx) for idx in translated_indices)
        
    # Remove <EOS> and <PAD> tokens from the translated sentence
    translated_sentence = translated_sentence.replace("<EOS>", "").replace("<PAD>", "").strip()
            
    return translated_sentence

def translate_with_rnn(text):
    """Translate text using the simple RNN model."""
    model = MODELS["rnn"]["model"]
    input_lang = MODELS["rnn"]["input_lang"]
    output_lang = MODELS["rnn"]["output_lang"]

    if not model or not input_lang or not output_lang:
        return "Simple RNN model is not loaded."

    tokens = [input_lang.get_index(word) for word in text.split(" ")]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        translated_indices = output.argmax(dim=-1).squeeze().tolist()
        translated_sentence = " ".join(output_lang.get_word(idx) for idx in translated_indices)
        
    # Remove <EOS> and <PAD> tokens from the translated sentence
    translated_sentence = translated_sentence.replace("<EOS>", "").replace("<PAD>", "").strip()
            
    return translated_sentence

if __name__ == "__main__":
    if args.model_type == "transformer":
        model, tokenizer = MODELS["transformer"]["model"], MODELS["transformer"]["tokenizer"]
        
        if not model or not tokenizer:
            print("Transformer model is not loaded.")
        else:
            print(translate_with_transformer(args.text))
            
    elif args.model_type == "rnn_attention":
        model, input_lang, output_lang = MODELS["rnn_attention"]["model"], MODELS["rnn_attention"]["input_lang"], MODELS["rnn_attention"]["output_lang"]
        if not model or not input_lang or not output_lang:
            print("RNN with Attention model is not loaded.")
        else:
            print(translate_with_rnn_attention(args.text))
            
    elif args.model_type == "rnn":
        model, input_lang, output_lang = MODELS["rnn"]["model"], MODELS["rnn"]["input_lang"], MODELS["rnn"]["output_lang"]
        print(translate_with_rnn(args.text))
    