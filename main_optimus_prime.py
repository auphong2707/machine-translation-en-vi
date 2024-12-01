import argparse
import os
import shutil
from config import *
from utils.helper import set_seed
set_seed(SEED)

from data.dataloader_for_marianmt import get_dataset
from transformers import MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from nltk.translate.bleu_score import corpus_bleu

from huggingface_hub import HfApi, login
import wandb


parser = argparse.ArgumentParser(description="Machine Translation Training Script")
parser.add_argument("--huggingface_token", type=str, required=True, help="Hugging Face token for authentication")
parser.add_argument("--wandb-token", type=str, required=True, help="Wandb token for logging")
args = parser.parse_args()

wandb.login(key=args.wandb_token)

wandb.init(project="machine-translation-en-vi", name=TFM_EXPERIMENT_NAME)

def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None


def main():
    # Load the data
    train_dataset, val_dataset, test_dataset, tokenizer = get_dataset([TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR])
    
    # Load the model
    os.makedirs(f"./results/{TFM_EXPERIMENT_NAME}", exist_ok=True)
    checkpoint = get_last_checkpoint(f"./results/{TFM_EXPERIMENT_NAME}")
    if checkpoint:
        model = MarianMTModel.from_pretrained(checkpoint)
    else:
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
    
    # Set up trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results/" + TFM_EXPERIMENT_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_dir="./results/" + TFM_EXPERIMENT_NAME,
        logging_steps=len(train_dataset) // BATCH_SIZE,
        predict_with_generate=True,
        generation_num_beams=BEAM_WIDTH,
        generation_max_length=MAX_SEQ_LENGTH,
        learning_rate=LEARNING_RATE,
        eval_strategy='epoch',
        save_strategy='epoch',
        fp16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb"
    )
    
    def compute_metrics_for_marinmtmodel(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        # Convert labels into a list of lists (each reference translation for BLEU calculation)
        references = [[label] for label in labels]
        bleu_score = corpus_bleu(references, decoded_preds)
        return {"bleu": bleu_score}
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_for_marinmtmodel
    )
    
    # Train the model
    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()
    
    # Save the model and tokenizer at the end of training
    model.save_pretrained(f"./results/{TFM_EXPERIMENT_NAME}/best_model")
    tokenizer.save_pretrained(f"./results/{TFM_EXPERIMENT_NAME}/best_model")
    
    # Evaluate the model
    # Best model is loaded automatically
    test_result = trainer.evaluate(test_dataset)
    with open(f"./results/{TFM_EXPERIMENT_NAME}/bleu_score_stats_best.txt", "w") as f:
        f.write(str(test_result))
        
    # Last model
    checkpoint = get_last_checkpoint(f"./results/{TFM_EXPERIMENT_NAME}")
    model = MarianMTModel.from_pretrained(checkpoint).to(DEVICE)
    trainer.model = model
    test_result = trainer.evaluate(test_dataset)
    with open(f"./results/{TFM_EXPERIMENT_NAME}/bleu_score_stats_last.txt", "w") as f:
        f.write(str(test_result))
    
    # Copy config.py to results folder
    shutil.copyfile('config.py', 'results/'+TFM_EXPERIMENT_NAME+'/config.py')
    
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