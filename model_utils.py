from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

def setup_model_and_tokenizer(model_config):
    """
    Set up the model and tokenizer based on the provided configuration.

    Args:
        model_config (dict): Configuration for the model and tokenizer.

    Returns:
        tuple: (model, tokenizer)
    """
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Set up LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    # Get the PEFT model
    model = get_peft_model(model, peft_config)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer