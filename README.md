# LLM Finetuning


## Project Overview

This project involves fine-tuning the `microsoft/Phi-3-mini-4k-instruct` model using the `OpenAssistant` dataset. We employ the QLoRA (Quantized Low-Rank Adaptation) strategy for efficient fine-tuning. The resulting model is deployed on Hugging Face Spaces for easy access and querying.

## Optimization Strategies

### QLoRA Implementation

- Utilizes Quantized Low-Rank Adaptation for efficient training
- Enables fine-tuning with 4-bit quantization, significantly reducing memory usage
- Facilitates training on GPUs with limited VRAM

### 4-bit Quantization

- Implements 4-bit quantization using bitsandbytes
- Further optimizes GPU memory utilization, allowing for training on less powerful hardware


## Project Structure

```
.
├── config.yaml           # Configuration for model, training, and data processing
├── finetune-phi3.py      # Main fine-tuning script
├── model_utils.py        # Utility functions for model and tokenizer setup
└── data_processing.py    # Dataset loading and preprocessing functions
```

## Installation

Clone this repository:

```bash
git clone https://github.com/aakashvardhan/microsoft-phi3-finetuning
cd microsoft-phi3-finetuning
```

Install the required packages:

```bash
pip install transformers datasets peft trl bitsandbytes wandb
```

Set up a Weights & Biases account and log in:

```bash
wandb login
```


## Usage

Adjust the settings in config.yaml as needed.
Run the fine-tuning script:

```bash
python finetune-phi3.py
```

Monitor the training progress on your Weights & Biases dashboard.
After training, the model will be saved in the directory specified by `output_dir` in the configuration.


## Hugging Face Spaces

[Click here to view the Hugging Face Spaces](https://huggingface.co/spaces/aakashv100/phi3-oass1-chatbot)

![Hugging Face Spaces](https://github.com/aakashvardhan/microsoft-phi3-finetuning/blob/main/asset/Screenshot%202024-09-20%20at%2012.46.54%E2%80%AFPM.png)

## W&B Report

[Click here to view the W&B report](https://api.wandb.ai/links/akv1000/e2qf8bap)