from config import *
from utils.helper import set_seed
set_seed(SEED)

from data.dataloader_for_marianmt import get_dataloader
from transformers import MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from nltk.translate.bleu_score import corpus_bleu

def main():
    # Load the training, validation, and tokenizer
    train_dataloader, val_dataloader, tokenizer = get_dataloader(dirs=[TRAIN_DATA_DIR, VAL_DATA_DIR], batch_size=BATCH_SIZE)
    
    # Load the model
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
    
    # Set up trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results/" + TFM_EXPERIMENT_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_dir="./results/" + TFM_EXPERIMENT_NAME,
        logging_steps=1,
        predict_with_generate=True,
        generation_num_beams=5,
        generation_max_length=MAX_SEQ_LENGTH,
        learning_rate=LEARNING_RATE,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        fp16=True,
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
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_for_marinmtmodel
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(f"./results/{TFM_EXPERIMENT_NAME}/fine_tuned_model")
    tokenizer.save_pretrained(f"./results/{TFM_EXPERIMENT_NAME}/fine_tuned_model")

    
if __name__ == "__main__":
    main()