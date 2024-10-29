from helper import set_seed
set_seed()

from config import DEVICE
from data.dataloader import get_dataloader
from models.seq2seq import Seq2SeqGRU
from trainer import Seq2SeqTrainer
from config import BATCH_SIZE, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def main():
    # Set name of experiment
    experiment_name = 'experiment_0'
    
    # Load data
    input_lang, output_lang, (val_loader, test_loader) = \
        get_dataloader(dirs=[VAL_DATA_DIR, TEST_DATA_DIR], batch_size=BATCH_SIZE)
    
    input_lang.save('results/'+experiment_name+'/input_lang.pkl')
    output_lang.save('results/'+experiment_name+'/output_lang.pkl')
    
    # Initialize model
    model = Seq2SeqGRU().to(DEVICE)

    # Initialize trainer
    trainer = Seq2SeqTrainer(model, experiment_name)

    # Train the model
    trainer.train(val_loader, test_loader, n_epochs=5, print_every=1, plot_every=1)

if __name__ == "__main__":
    main()
