import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from utils.helper import load_checkpoint
from models.seq2seq import Seq2SeqGRU, Seq2SeqAttn

class Beam:
    def __init__(self, model, beam_width=BEAM_WIDTH, max_seq_length=MAX_SEQ_LENGTH, device=DEVICE):
        self.model = model
        self.beam_width = beam_width
        self.max_seq_length = max_seq_length
        self.device = device

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        all_outputs = []
        
        for i in range(batch_size):
            # Run beam search for each data point in the batch
            outputs = self.beam_search(input_tensor[i].unsqueeze(0))
            all_outputs.append(outputs)
        
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
            beams = [(0, torch.empty(1, 1, dtype=torch.long, device=self.device).fill_(self.model.decoder.sos_token), decoder_hidden)]
            
            for _ in range(self.max_seq_length):
                all_candidates = []
                for score, sequence, hidden in beams:
                    # Skip sequences ending with <eos>
                    if sequence[0, -1].item() == EOS_TOKEN:
                        all_candidates.append((score, sequence, hidden))
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
                        all_candidates.append((new_score, new_seq, hidden))

                # Select the top beam_width beams
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:self.beam_width]

            # Return sequences from the final beams
            final_sequences = [seq for _, seq, _ in beams]
            return final_sequences
        
        elif type(self.model) == Seq2SeqAttn:
            # Encode the input sequence
            encoder_outputs, encoder_hidden = self.model.encoder(input_tensor)
        
            # For bidirectional GRU, we need to combine the last hidden states
            if self.encoder_bidirectional:
                # Concatenate the final forward and backward hidden states
                hidden_forward = encoder_hidden[0::2]
                hidden_backward = encoder_hidden[1::2]
                
                decoder_hidden = torch.cat((hidden_forward, hidden_backward), dim=-1)
            else:
                decoder_hidden = encoder_hidden  # Use the hidden state directly
                
            # Start the beam with the <sos> token and zero score
            beams = [(0, torch.empty(1, 1, dtype=torch.long, device=self.device).fill_(self.model.decoder.sos_token), decoder_hidden, encoder_outputs)]
            
            for _ in range(self.max_seq_length):
                all_candidates = []
                for score, sequence, hidden, enc_outputs in beams:
                    # Skip sequences ending with <eos>
                    if sequence[0, -1].item() == EOS_TOKEN:
                        all_candidates.append((score, sequence, hidden))
                        continue
                
                    # Pass through the decoder step by step
                    output, hidden, _ = self.model.decoder.forward_step(sequence[:, -1:], hidden, enc_outputs)
                    log_probs = F.log_softmax(output, dim=-1).squeeze(1)
                    
                    # Get top beam_width candidates
                    top_log_probs, top_indices = log_probs.topk(self.beam_width)
                    
                    # Create new beams with updated sequences and scores
                    for k in range(self.beam_width):
                        new_seq = torch.cat([sequence, top_indices[:, k].unsqueeze(1)], dim=1)
                        new_score = score + top_log_probs[:, k].item()
                        all_candidates.append((new_score, new_seq, hidden, enc_outputs))

                # Select the top beam_width beams
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:self.beam_width]

            # Return sequences from the final beams
            final_sequences = [seq for _, seq, _, _ in beams]
            return final_sequences
    
def remove_unnecessary_tokens(sentence):
    return [word for word in sentence if word not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
    
    
def evaluate(experiment_name, output_lang, test_loader, model, optimizer):
    load_checkpoint(model, optimizer, f'results/{experiment_name}/best.pth')

    model.eval()
    beam_search = Beam(model)
    
    smoothing = SmoothingFunction().method1
    bleu_score_best = []
    bleu_score_avg = []
    
    for idx, batch in enumerate(test_loader):
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
            
        print(f"Batch {idx + 1}/{len(test_loader)} completed")
    
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
