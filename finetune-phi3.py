import logging
import wandb
from transformers import TrainingArguments
from datasets import disable_caching
from trl import SFTTrainer
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
    model, tokenizer = setup_model_and_tokenizer(config["model_config"])

    # Preprocess datasets
    train_ds, val_ds = preprocess_datasets(
        train_ds, val_ds, tokenizer, config["preprocessing"]
    )

    # Setup training arguments
    training_args = TrainingArguments(**config["training_config"], report_to=["wandb"])

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=config["peft_config"],
        dataset_text_field="conversation",
        max_seq_length=config["model_config"]["max_length"],
        tokenizer=tokenizer,
        args=training_args,
    )

    # Start training
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Log the evaluation results
    wandb.log({"eval_loss": trainer.state.log_history[-1]["eval_loss"]})

    # close wandb
    wandb.finish()

    # Save the final pre-trained model
    trainer.model.save_pretrained(config["training_config"]["output_dir"])


if __name__ == "__main__":
    main()
