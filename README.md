# LLM Finetuning


## Project Overview

This project involves fine-tuning the `microsoft/Phi-3-mini-4k-instruct` model using the `OpenAssistant` dataset. We employ the QLoRA (Quantized Low-Rank Adaptation) strategy for efficient fine-tuning. The resulting model is deployed on Hugging Face Spaces for easy access and querying.

## Optimization Strategies

- Implements QLoRA strategy for efficient training
    - QLoRA allows for fine-tuning models with 4-bit quantization, reducing memory usage and enabling training on GPUs with less VRAM.

- Supports 4-bit quantization using `bitsandbytes`
    - This allows for training on GPUs with less VRAM.


## Project Structure

- `config.yaml`: Configuration file for model, training, and data processing settings
- `finetune-phi3.py`: Main script for fine-tuning the model
- `model_utils.py`: Utility functions for setting up the model and tokenizer
- `data_processing.py`: Functions for loading and preprocessing the dataset


## Installation

Clone this repository:

```
git clone https://github.com/aakashvardhan/microsoft-phi3-finetuning
cd microsoft-phi3-finetuning
```

Install the required packages:

```
pip install transformers datasets peft trl bitsandbytes wandb
```

Set up a Weights & Biases account and log in:

```
wandb login
```


## Usage

Adjust the settings in config.yaml as needed.
Run the fine-tuning script:

```
python finetune-phi3.py
```

Monitor the training progress on your Weights & Biases dashboard.
After training, the model will be saved in the directory specified by `output_dir` in the configuration.


## Hugging Face Spaces

[Click here to view the Hugging Face Spaces](https://huggingface.co/spaces/aakashv100/phi3-oass1-chatbot)

![Hugging Face Spaces](./assets/hf_spaces.png)

## W&B Report

[Click here to view the W&B report](https://api.wandb.ai/links/akv1000/e2qf8bap)