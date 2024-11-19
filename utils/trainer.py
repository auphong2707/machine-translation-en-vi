import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

import sys
sys.path.append('../machine-translation-en-vi')
from config import *
from utils.helper import time_since, save_checkpoint, load_checkpoint, save_loss, save_plot
from utils.logger import setup_logger

class Seq2SeqTrainer:
    def __init__(self, model, name,
                 learning_rate=LEARNING_RATE, 
                 device=DEVICE, 
                 criterion=nn.NLLLoss(), 
                 max_norm=1.0):
        # [SAVE TRAINER PARAMETERS]
        self.name = name
        self.model = model.to(device)
        self.checkpoint_directory = f'results/{self.name}'
        
        self.device = device
        self.max_norm = max_norm
        
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.logger = setup_logger(self.checkpoint_directory + '/training.log')
        
        # Prepare checkpoint directory
        self.best_loss = float('inf')
        best_checkpoint_path = os.path.join(self.checkpoint_directory, '/best.pth')
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path)
            self.best_loss = checkpoint['loss']
            

    def train_epoch(self, dataloader):
        total_loss = 0
        self.model.train()  # Set model to training mode

        for data in dataloader:
            input_tensor, target_tensor = data
            input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            decoder_outputs = self.model(input_tensor, target_tensor)

            # Calculate loss
            loss = self.criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            # Update weights
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def validate(self, val_dataloader):
        total_loss = 0
        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for data in val_dataloader:
                input_tensor, target_tensor = data
                input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)

                # Forward pass
                decoder_outputs = self.model(input_tensor, target_tensor)

                # Calculate loss
                loss = self.criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)),
                    target_tensor.view(-1)
                )

                # Accumulate loss
                total_loss += loss.item()

        return total_loss / len(val_dataloader)

    def train(self, train_dataloader, val_dataloader, n_epochs, print_every=1, plot_every=1):
        # Load checkpoint if available
        load_checkpoint_path = self.checkpoint_directory + '/checkpoint.pth'
        start_epoch = load_checkpoint(self.model, self.optimizer, load_checkpoint_path) + 1
        
        print(f"\nStart training from epoch {start_epoch}")
        print('-' * 80)
        
        # Initialize variables
        start = time.time()

        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(start_epoch, n_epochs + 1):
            # Training step
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation step
            val_loss = self.validate(val_dataloader)
                
            # Save checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, train_loss, self.checkpoint_directory + '/checkpoint.pth')

            # Save the best checkpoint
            if epoch == 1 or val_loss < self.best_loss:
                self.best_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, train_loss, self.checkpoint_directory + '/best.pth', best=True)
                
            # Print loss every 'print_every' epochs
            if epoch % print_every == 0:
                self.logger.info('%s (%d %d%%) Train loss: %.4f | Val loss: %.4f' % 
                                 (time_since(start, epoch / n_epochs), 
                                  epoch, epoch / n_epochs * 100, 
                                  train_loss, val_loss))
                print('-' * 80)
            
            # Collect losses for plotting
            if epoch % plot_every == 0:
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
            save_loss(epoch, train_loss, val_loss, self.checkpoint_directory + '/losses.csv')

        # Save the plot
        save_plot(self.checkpoint_directory + '/losses.csv', self.checkpoint_directory + '/losses.png')
