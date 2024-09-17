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
    
    # Shuffle and select samples from both train and validation sets
    train_dataset = dataset["train"].shuffle(seed=42).select(range(dataset_config.get("num_train_samples", len(dataset["train"]))))
    val_dataset = dataset["validation"].shuffle(seed=42).select(range(dataset_config.get("num_val_samples", len(dataset["validation"]))))

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
        
        # Group examples by conversation
        conversations = {}
        for role, text, parent_id in zip(examples["role"], examples["text"], examples["parent_id"]):
            if parent_id not in conversations:
                conversations[parent_id] = {"prompter": [], "assistant": []}
            conversations[parent_id][role].append(text)

        processed_conversations = []
        for conv in conversations.values():
            prompts = conv["prompter"]
            responses = [" ".join(conv["assistant"])]

            if not prompts or len(prompts) != len(responses):
                continue

            base_prompt = prompts[0]
            augmented_prompts = [base_prompt] + [f"{base_prompt} {prompt}" for prompt in prompts[1:]]

            for prompt, response in zip(augmented_prompts, responses):
                conversation = f"{system_message}\nHuman: {prompt}\nAssistant: {response}"
                processed_conversations.append(conversation)

        tokenized = tokenizer(
            processed_conversations,
            truncation=True,
            max_length=preprocessing_config.get("max_length", 2048),
            padding="max_length",
        )

        return {
            "conversation": processed_conversations,
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
