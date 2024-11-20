from config import *
from utils.helper import set_seed
set_seed(SEED)

from data.dataloader import get_dataloader
from models.seq2seq import Seq2SeqRNN
from utils.trainer import Seq2SeqTrainer
from utils.tester import evaluate
from huggingface_hub import HfApi, login
import argparse

parser = argparse.ArgumentParser(description="Machine Translation Training Script")
parser.add_argument("--huggingface_token", type=str, required=True, help="Hugging Face token for authentication")
args = parser.parse_args()

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def main():
    # Set name of experiment
    experiment_name = RNN_EXPERIMENT_NAME
    
    # Load data
    input_lang, output_lang, (train_loader, val_loader, test_loader) = \
        get_dataloader(dirs=[TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR], batch_size=BATCH_SIZE)
    
    input_lang.save('results/'+experiment_name+'/input_lang.pkl')
    output_lang.save('results/'+experiment_name+'/output_lang.pkl')
    
    # Initialize model
    model = Seq2SeqRNN()

    # Initialize trainer
    trainer = Seq2SeqTrainer(model, experiment_name)

    # Train the model
    trainer.train(train_loader, val_loader, n_epochs=EPOCHS)

    # Test the model
    evaluate(experiment_name, output_lang, test_loader, model, trainer.optimizer)
    
    
    # Push to Hugging Face
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
 