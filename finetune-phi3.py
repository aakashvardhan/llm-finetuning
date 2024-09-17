import logging
from transformers import TrainingArguments
from datasets import set_caching_enabled
from trl import SFTTrainer
from config import load_config
from data_processing import get_train_val_ds, preprocess_datasets
from model_utils import setup_model_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    config = load_config("config.yaml")

    # Disable caching to save memory
    set_caching_enabled(False)

    # Get datasets
    train_ds, val_ds = get_train_val_ds(config["dataset"])

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config["model"])

    # Preprocess datasets
    train_ds, val_ds = preprocess_datasets(
        train_ds, val_ds, tokenizer, config["preprocessing"]
    )

    # Setup training arguments
    training_args = TrainingArguments(**config["training"])

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=config["peft"],
        dataset_text_field="conversation",
        max_seq_length=config["model"]["max_length"],
        tokenizer=tokenizer,
        args=training_args,
    )

    # Start training
    trainer.train()

    # Save the final pre-trained model
    trainer.model.save_pretrained(config["training"]["output_dir"])


if __name__ == "__main__":
    main()
