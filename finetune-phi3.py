import logging
import os
from datasets import disable_caching
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import wandb
from config import load_config
from data_processing import get_train_val_ds, preprocess_datasets
from model_utils import setup_model_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    config = load_config("config.yaml")

    # Initialize wandb
    wandb.init(project="phi-3-fine-tuning", config=config, name="qloratest")

    # Disable caching to save memory
    disable_caching()

    # Get datasets
    train_ds, val_ds = get_train_val_ds(config["base_config"])

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        config["model_config"], config["peft_config"], config["bnb_config"]
    )

    # Preprocess datasets
    train_ds, val_ds = preprocess_datasets(
        train_ds, val_ds, tokenizer, config["preprocessing"]
    )

    # Add this sanity check
    if (
        "conversation" not in train_ds.features
        or "input_ids" not in train_ds.features
        or "attention_mask" not in train_ds.features
    ):
        raise ValueError(
            "Preprocessed datasets are missing expected features. Check the preprocessing function."
        )

    # Sanity check: Verify dataset sizes
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Validation dataset size: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "One or both datasets are empty. Please check your data loading process."
        )

    # Sanity check: Verify model and tokenizer compatibility
    if tokenizer.vocab_size != model.config.vocab_size:
        logger.warning(f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match model vocab size ({model.config.vocab_size}). This may not be an issue for some models.")
        # Optionally, you can still raise the error if you want to be strict:
        # raise ValueError("Tokenizer and model vocabulary sizes do not match.")

    # Sanity check: Verify a sample input
    sample_input = train_ds[0]["conversation"]
    encoded_input = tokenizer(
        sample_input,
        return_tensors="pt",
        truncation=True,
        max_length=config["model_config"]["max_length"],
    )
    try:
        _ = model(**encoded_input)
        logger.info("Sample input successfully passed through the model.")
    except Exception as e:
        raise RuntimeError(f"Error processing sample input through the model: {str(e)}")

    # Setup training arguments
    training_args = TrainingArguments(**config["training_config"])

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(**config["peft_config"]),
        dataset_text_field="conversation",
        max_seq_length=config["model_config"]["max_length"],
        tokenizer=tokenizer,
        args=training_args,
    )

    # Start training
    trainer.train()

    # Sanity check: Verify training completed successfully
    if trainer.state.global_step == 0:
        raise RuntimeError(
            "Training did not progress. Check your training configuration and data."
        )

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Sanity check: Verify evaluation results
    if "eval_loss" not in eval_results:
        raise ValueError(
            "Evaluation did not produce expected metrics. Check your evaluation process."
        )

    # Log the evaluation results
    wandb.log({"eval_loss": eval_results["eval_loss"]})

    # Sanity check: Verify model saving
    if not os.path.exists(config["training_config"]["output_dir"]):
        raise RuntimeError(
            "Model output directory does not exist. Model may not have been saved correctly."
        )

    logger.info("Fine-tuning completed successfully!")

    # Save the final pre-trained model
    trainer.model.save_pretrained(config["training_config"]["output_dir"])


if __name__ == "__main__":
    main()
