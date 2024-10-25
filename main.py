import torch
import argparse
from data.data_loader import DataLoader  # Custom data loader
from models.seq2seq import Seq2SeqModel  # Encoder-decoder model
from training.train import train  # Training function
from training.evaluation import evaluate  # Evaluation function
from utils.logger import setup_logger
from utils.utils import set_seed, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an RNN encoder-decoder model.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--log_dir', type=str, default="logs", help="Directory for logs")
    return parser.parse_args()

def main():
    # Parse arguments and set random seed
    args = parse_args()
    set_seed(42)
    
    # Set up logger
    logger = setup_logger(log_dir=args.log_dir, log_file="train.log")
    logger.info("Starting training...")
    
    # Initialize data loader
    data_loader = DataLoader(batch_size=args.batch_size)
    train_loader, val_loader = data_loader.get_loaders()
    
    # Initialize model and optimizer
    model = Seq2SeqModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    
    # Load from checkpoint if resuming
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, filepath="checkpoints/latest_checkpoint.pth")
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, filepath="checkpoints/latest_checkpoint.pth")
    
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
