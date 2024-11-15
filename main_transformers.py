from helper import set_seed
set_seed()

from data.dataloader import get_dataloader
from models.transformer import Transformer
from trainer import Trainer
from config import *
from tester import evaluate
from huggingface_hub import HfApi, login
import argparse

parser = argparse.ArgumentParser(description="Machine Translation Training Script")
parser.add_argument("--huggingface_token", type=str, required=True, help="Hugging Face token for authentication")
args = parser.parse_args()

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def main():
    # Set name of experiment
    experiment_name = EXPERIMENT_NAME
    
    # Load data
    input_lang, output_lang, (train_loader, val_loader, test_loader) = \
        get_dataloader(dirs=[TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR], batch_size=BATCH_SIZE)
    
    input_lang.save('results/'+experiment_name+'/input_lang.pkl')
    output_lang.save('results/'+experiment_name+'/output_lang.pkl')
    
    # Initialize model
    model = Transformer()

    # Initialize trainer
    trainer = Trainer(model, experiment_name)

    # Train the model
    trainer.train(train_loader, val_loader, n_epochs=EPOCHS, print_every=1, plot_every=1)

    # Test the model
    evaluate(experiment_name, output_lang, test_loader)
    
    
    # Push to Hugging Face)
    login(token=args.huggingface_token)
    
    api = HfApi()
    api.upload_large_folder(
        folder_path='results',
        repo_type='model',
        repo_id='auphong2707/machine-translation-en-vi',
        private=True
    )
    

if __name__ == "__main__":
    main()
 