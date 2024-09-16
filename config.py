from dataclasses import dataclass
import torch


@dataclass
class BaseConfig:
    """Configuration settings for the model and dataset."""

    # Model and dataset
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    dataset_name: str = "OpenAssistant/oasst1"


@dataclass
class PeftConfig:
    """Configuration settings for LoRA"""

    # LoRA config
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    task_type = "CAUSAL_LM"
    target_modules = ["o_proj", "qkv_proj"]


@dataclass
class TrainingConfig:
    """Configuration settings for training process."""

    # Training config
    output_dir = "./results"
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 500
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"
    max_seq_length = 512


@dataclass
class BnbConfig:
    """Configuration settings for BitsAndBytesConfig"""

    load_in_8bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
