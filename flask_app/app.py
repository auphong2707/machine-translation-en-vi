from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer
import torch.nn as nn
import torch

app = Flask(__name__)

# Define available models
MODELS = {
    "rnn": {"name": "Simple RNN", "model": None},  # Replace with your RNN loading logic
    "rnn_attention": {"name": "RNN + Attention", "model": None},  # Replace with RNN + Attention logic
    "transformer": {
        "name": "Transformer",
        "model": None,
        "tokenizer": None,
    },
}

def translate_with_transformer(text):
    """Translate text using the Transformer."""
    tokenizer = MODELS["transformer"]["tokenizer"]
    model = MODELS["transformer"]["model"]
    tokens = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_with_rnn(text):
    """Dummy translation for RNN (replace with actual implementation)."""
    return "Translation using RNN is not implemented."

def translate_with_rnn_attention(text):
    """Dummy translation for RNN with Attention (replace with actual implementation)."""
    return "Translation using RNN + Attention is not implemented."

@app.route("/", methods=["GET", "POST"])
def index():
    english_text = ""
    vietnamese_text = ""
    selected_model = "transformer"  # Default model

    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        selected_model = request.form.get("model", "transformer")

        if english_text.strip():
            if selected_model == "rnn":
                vietnamese_text = translate_with_rnn(english_text)
            elif selected_model == "rnn_attention":
                vietnamese_text = translate_with_rnn_attention(english_text)
            elif selected_model == "transformer":
                vietnamese_text = translate_with_transformer(english_text)

    return render_template(
        "index.html",
        english_text=english_text,
        vietnamese_text=vietnamese_text,
        selected_model=selected_model,
        models=MODELS,
    )

if __name__ == "__main__":
    app.run(debug=True)
