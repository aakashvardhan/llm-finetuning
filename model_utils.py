from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

def setup_model_and_tokenizer(model_config, peft_config, bnb_config):
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_config['load_in_4bit'],
        bnb_4bit_quant_type=bnb_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, bnb_config['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=bnb_config['bnb_4bit_use_double_quant'],
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_config['trust_remote_code'],
        torch_dtype=getattr(torch, model_config['torch_dtype']),
    )
    model.config.use_cache = model_config['use_cache']
    model.config.pretraining_tp = 1

    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Set up LoRA configuration
    peft_config = LoraConfig(**peft_config)

    # Get the PEFT model
    model = get_peft_model(model, peft_config)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'], trust_remote_code=model_config['trust_remote_code'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer