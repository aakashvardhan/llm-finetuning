from datasets import load_dataset, Dataset
from typing import Tuple, Dict, Any


def get_train_val_ds(dataset_config: Dict) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare the training and validation datasets.

    Args:
        dataset_config (Dict): Configuration for dataset loading.

    Returns:
        Tuple[Dataset, Dataset]: Training and validation datasets.
    """
    dataset = load_dataset(dataset_config["dataset_name"])
    dataset = dataset.shuffle().select(range(dataset_config["num_samples"]))

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    return train_dataset, val_dataset


def preprocess_datasets(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: Any,
    preprocessing_config: Dict[str, Any],
) -> Tuple[Dataset, Dataset]:
    """
    Preprocess the training and validation datasets.

    Args:
        train_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        tokenizer (Any): The tokenizer to use for preprocessing.
        preprocessing_config (Dict[str, Any]): Configuration for preprocessing.

    Returns:
        Tuple[Dataset, Dataset]: Preprocessed training and validation datasets.
    """

    def preprocess_function(examples):
        system_message = preprocessing_config.get("system_message", "")

        conversations = []
        for prompt, response in zip(examples["prompt"], examples["response"]):
            conversation = f"{system_message}\nHuman: {prompt}\nAssistant: {response}"
            conversations.append(conversation)

        tokenized = tokenizer(
            conversations,
            truncation=True,
            max_length=preprocessing_config.get("max_length", 2048),
            padding="max_length",
        )

        return {
            "conversation": conversations,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    # Apply preprocessing to both datasets
    train_ds = train_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        preprocess_function,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    return train_ds, val_ds
