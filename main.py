from helper import set_seed
set_seed()

from config import DEVICE
from data.dataloader import get_dataloader
from models.seq2seq import Seq2SeqGRU
from trainer import Seq2SeqTrainer
from config import BATCH_SIZE, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR
from tester import tester
from huggingface_hub import HfApi, login
from kaggle_secrets import UserSecretsClient


import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def main():
    # Set name of experiment
    experiment_name = 'experiment_0'
    
    # Load data
    input_lang, output_lang, (train_loader, val_loader, test_loader) = \
        get_dataloader(dirs=[TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR], batch_size=BATCH_SIZE)
    
    input_lang.save('results/'+experiment_name+'/input_lang.pkl')
    output_lang.save('results/'+experiment_name+'/output_lang.pkl')
    
    # Initialize model
    model = Seq2SeqGRU().to(DEVICE)

    # Initialize trainer
    trainer = Seq2SeqTrainer(model, experiment_name)

    # Train the model
    trainer.train(train_loader, val_loader, n_epochs=20, print_every=1, plot_every=1)

    # Test the model
    tester(experiment_name, output_lang, test_loader)
    
    
    # Push to Hugging Face
    user_secrets = UserSecretsClient()
    login(token=user_secrets.get_secret("huggingface_token"))
    
    api = HfApi()
    api.upload_large_folder(
        folder_path='results',
        repo_type='model',
        repo_id='auphong2707/machine-translation-en-vi',
        private=True
    )
    

if __name__ == "__main__":
    main()
 