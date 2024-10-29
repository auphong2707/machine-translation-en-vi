from config import DEVICE
from data.dataloader import get_dataloader
from models.seq2seq import Seq2SeqGRU
from trainer import Seq2SeqTrainer
from config import BATCH_SIZE, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR
from helper import set_seed
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def main():
    # Set seed for reproducibility
    set_seed()
    
    # Load data
    input_lang, output_lang, (val_loader, test_loader) = \
        get_dataloader(dirs=[VAL_DATA_DIR, TEST_DATA_DIR], batch_size=BATCH_SIZE)
    
    # Initialize model
    model = Seq2SeqGRU().to(DEVICE)

    # Initialize trainer
    trainer = Seq2SeqTrainer(model, 'experiment_0')

    # Train the model
    trainer.train(val_loader, test_loader, n_epochs=10, print_every=1, plot_every=1)

if __name__ == "__main__":
    main()
