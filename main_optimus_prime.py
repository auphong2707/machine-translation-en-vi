from config import *
from utils.helper import set_seed
set_seed(SEED)

from data.dataloader_for_marianmt import get_dataset
from transformers import MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from nltk.translate.bleu_score import corpus_bleu

def main():
    # Load the data
    train_dataset, val_dataset, tokenizer = get_dataset([TRAIN_DATA_DIR, VAL_DATA_DIR])
    
    # Load the model
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
        generation_num_beams=5,
        generation_max_length=MAX_SEQ_LENGTH,
        learning_rate=LEARNING_RATE,
        eval_strategy='epoch',
        save_strategy='epoch',
        fp16=True,
        report_to="none"
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
    trainer.train()
    
    # Save the model
    model.save_pretrained(f"./results/{TFM_EXPERIMENT_NAME}/fine_tuned_model")
    tokenizer.save_pretrained(f"./results/{TFM_EXPERIMENT_NAME}/fine_tuned_model")

    
if __name__ == "__main__":
    main()