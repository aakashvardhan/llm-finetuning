import datasets
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from config import BaseConfig, PeftConfig, TrainingConfig, BnbConfig


bnb_config = BitsAndBytesConfig(**BnbConfig.__dict__)
lora_config = LoraConfig(**PeftConfig.__dict__)
train_config = TrainingConfig(**TrainingConfig.__dict__)

model = AutoModelForCausalLM.from_pretrained(
    BaseConfig.model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_cache=False,
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained(BaseConfig.model_name)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
