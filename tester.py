import math
import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import sys
sys.path.append('../machine-translation-en-vi')
from config import *

from helper import load_checkpoint
from models.seq2seq import Seq2SeqGRU
from models.transformer import Transformer

from data.dataloader import Lang

class Beam:
    def __init__(self, model, beam_width=BEAM_WIDTH, max_seq_length=MAX_SEQ_LENGTH, device=DEVICE):
        self.model = model
        self.beam_width = beam_width
        self.max_seq_length = max_seq_length
        self.device = device

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        all_outputs = []
        
        input_lang = Lang('en')
        input_lang.load('results/{}/input_lang.pkl'.format(EXPERIMENT_NAME))
        
        output_lang = Lang('vi')
        output_lang.load('results/{}/output_lang.pkl'.format(EXPERIMENT_NAME))
        
        for i in range(batch_size):
            # Run beam search for each data point in the batch
            outputs = self.beam_search(input_tensor[i].unsqueeze(0))
            all_outputs.append(outputs)
            
            
            
            input_sentence = [input_lang.get_word(token_id) for token_id in input_tensor[i].tolist()]
            print(f"Input: {' '.join(input_sentence)}")
            
            for j, output in enumerate(outputs):
                output_sentence = [output_lang.get_word(token_id) for token_id in output.squeeze().tolist()]
                print(f"Output {j+1}: {' '.join(output_sentence)}")
        
        return all_outputs

    def beam_search(self, input_tensor):
        if type(self.model) == Seq2SeqGRU:
            # Initialize the beam with a list of tuples [(sequence, score, hidden)]
            encoder_outputs, encoder_hidden = self.model.encoder(input_tensor)
            
            # For bidirectional GRU, we need to combine the last hidden states
            if self.model.encoder_bidirectional:
                # Concatenate the final forward and backward hidden states
                hidden_forward = encoder_hidden[0::2]
                hidden_backward = encoder_hidden[1::2]
                
                decoder_hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
            else:
                decoder_hidden = encoder_hidden  # Use the hidden state directly
            
            # Start the beam with the <sos> token and zero score
            beams = [(torch.empty(1, 1, dtype=torch.long, device=self.device).fill_(self.model.decoder.sos_token), 0, decoder_hidden)]
            
            for _ in range(self.max_seq_length):
                new_beams = []
                for sequence, score, hidden in beams:
                    if sequence[0, -1].item() == EOS_TOKEN:
                        new_beams.append((sequence, score, hidden))
                        continue
                    
                    # Pass through the decoder step by step
                    output, hidden = self.model.decoder.forward_step(sequence[:, -1:], hidden)
                    log_probs = F.log_softmax(output, dim=-1).squeeze(1)
                    
                    # Get top beam_width candidates
                    top_log_probs, top_indices = log_probs.topk(self.beam_width)
                    
                    # Create new beams with updated sequences and scores
                    for k in range(self.beam_width):
                        new_seq = torch.cat([sequence, top_indices[:, k].unsqueeze(1)], dim=1)
                        new_score = score + top_log_probs[:, k].item()
                        new_beams.append((new_seq, new_score, hidden))

                # Select the top beam_width beams
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_width]
                beams = new_beams
                
                # Stop if all beams have end with the <eos> token
                if all([seq[0, -1].item() == EOS_TOKEN for seq, _ in beams]):
                    break

            # Return sequences from the final beams
            final_sequences = [seq for seq, _, _ in beams]
            return final_sequences
        elif type(self.model) == Transformer:
            # Initialize beam with [(sequence, score)]
            input_mask, _, input_padding_mask, _ = self.model.create_masks(input_tensor, input_tensor)
            
            # Encoder forward pass
            memory = self.model.transformer.encoder(
                self.model.positional_encoding(self.model.input_embedding(input_tensor) * math.sqrt(self.model.embedding_size)),
                src_key_padding_mask=input_padding_mask
            )
            
            beams = [(torch.full((1, 1), SOS_TOKEN, dtype=torch.long, device=self.device), 0)]  # (sequence, score)

            for _ in range(self.max_seq_length):
                new_beams = []
                for sequence, score in beams:
                    if sequence[0, -1].item() == EOS_TOKEN:
                        new_beams.append((sequence, score))
                        continue
                    
                    # Create target mask
                    _, target_mask, _, target_padding_mask = self.model.create_masks(sequence, sequence)
                    
                    # Decoder forward pass
                    target_embedding = self.model.positional_encoding(
                        self.model.target_embedding(sequence) * math.sqrt(self.model.embedding_size)
                    )
                    
                    output = self.model.transformer.decoder(
                        target_embedding, memory,
                        tgt_mask=target_mask,
                        tgt_key_padding_mask=target_padding_mask,
                        memory_key_padding_mask=input_padding_mask
                    )
                    logits = self.model.linear(output[:, -1])  # Last token logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get top-k predictions
                    top_log_probs, top_indices = log_probs.topk(self.beam_width)
                    
                    # Update beams
                    for k in range(self.beam_width):
                        new_seq = torch.cat([sequence, top_indices[:, k].unsqueeze(1)], dim=1)
                        new_score = score + top_log_probs[:, k].item()
                        new_beams.append((new_seq, new_score))

                # Prune to top-k beams
                new_beams = sorted(new_beams, key=lambda x: x[1]/(len(x[0]) ** 0.75), reverse=True)[:self.beam_width]
                beams = new_beams
                
                # Stop if all beams have end with the <eos> token
                if all([seq[0, -1].item() == EOS_TOKEN for seq, _ in beams]):
                    break
                
            # Return sequences from the final beams
            final_sequences = [seq for seq, _ in beams]
            return final_sequences
                
    
def remove_unnecessary_tokens(sentence):
    return [word for word in sentence if word not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
    
    
def evaluate(experiment_name, output_lang, test_loader, model_type=Seq2SeqGRU, optimizer_type=torch.optim.Adam):
    model = model_type()    
    optimizer = optimizer_type(model.parameters(), lr=LEARNING_RATE)
    load_checkpoint(model, optimizer, f'results/{experiment_name}/best.pth')

    model.eval()
    beam_search = Beam(model)
    
    smoothing = SmoothingFunction().method1
    bleu_score_best = []
    bleu_score_avg = []
    
    for batch in test_loader:
        inputs, targets = batch

        outputs = beam_search.forward(inputs.to(DEVICE))
    
        for i in range(BATCH_SIZE):
            target_sentence = remove_unnecessary_tokens(targets[i].tolist())
            target_sentence = [output_lang.get_word(token_id) for token_id in target_sentence]
            
            
            sentence_bleu_score = []
            
            for output in outputs[i]:
                output_sentence = remove_unnecessary_tokens(output.squeeze().tolist())
                output_sentence = [output_lang.get_word(token_id) for token_id in output_sentence]
                
                sentence_bleu_score.append(sentence_bleu([target_sentence], output_sentence, smoothing_function=smoothing))
                
            bleu_score_best.append(max(sentence_bleu_score))
            bleu_score_avg.append(np.mean(sentence_bleu_score))
    
    num_elements = len(bleu_score_best)
    
    mean_bleu_best = np.mean(bleu_score_best)
    variance_bleu_best = np.var(bleu_score_best)
    
    mean_bleu_avg = np.mean(bleu_score_avg)
    variance_bleu_avg = np.var(bleu_score_avg)

    with open(r"results/{}/bleu_score_stats.txt".format(experiment_name), "w") as f:
        f.write(f"Number of elements in BLEU score array: {num_elements}\n\n")
        
        f.write(f"Mean BLEU score (best): {mean_bleu_best}\n")
        f.write(f"Variance of BLEU score (Best): {variance_bleu_best}\n\n")
        
        f.write(f"Mean BLEU score (avg): {mean_bleu_avg}\n")
        f.write(f"Variance of BLEU score (avg): {variance_bleu_avg}\n")
