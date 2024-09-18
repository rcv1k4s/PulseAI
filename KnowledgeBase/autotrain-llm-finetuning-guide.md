# AutoTrain LLM Finetuning Guide

## Installation

To install AutoTrain, use the following command:

```bash
pip install -U autotrain-advanced
```

For Colab environments, you may need to set up additional components:

```bash
autotrain setup --colab
```

## Use Cases

AutoTrain LLM is designed for fine-tuning large language models. Common use cases include:

1. Customizing language models for specific domains (e.g., legal, medical, technical)
2. Improving model performance on specific tasks (e.g., summarization, question-answering)
3. Adapting models to understand company-specific jargon or writing styles
4. Creating chatbots with specialized knowledge
5. Fine-tuning models for different languages or dialects

## Examples

### Example 1: Basic Fine-tuning

This example demonstrates how to fine-tune a model using a CSV file with a text column.

1. Prepare your data:
   - Create a folder named `data/`
   - Place your `train.csv` file in the `data/` folder
   - Ensure `train.csv` has a `text` column

2. Create a configuration file `conf.yaml`:

```yaml
task: llm-sft
base_model: abhishek/llama-2-7b-hf-small-shards
project_name: my-autotrain-llm
log: tensorboard
backend: local

data:
  path: data/
  train_split: train
  valid_split: null
  column_mapping:
    text_column: text

params:
  block_size: 1024
  lr: 2e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  epochs: 1
  batch_size: 1
  gradient_accumulation: 4
  mixed_precision: fp16
  peft: true
  quantization: int4
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

hub:
  push_to_hub: false
```

3. Run AutoTrain:

```bash
autotrain --config conf.yaml
```

### Example 2: Fine-tuning with Push to Hub

This example shows how to fine-tune a model and push it to the Hugging Face Hub.

1. Prepare your data as in Example 1.

2. Create a configuration file `conf.yaml`:

```yaml
task: llm-sft
base_model: meta-llama/Llama-2-7b-hf
project_name: my-custom-llama
log: tensorboard
backend: local

data:
  path: data/
  train_split: train
  valid_split: null
  column_mapping:
    text_column: text

params:
  block_size: 2048
  lr: 1e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  epochs: 3
  batch_size: 4
  gradient_accumulation: 8
  mixed_precision: bf16
  peft: true
  quantization: int8
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.1

hub:
  username: ${HF_USERNAME}
  token: ${HF_TOKEN}
  push_to_hub: true
```

3. Set environment variables:

```bash
export HF_USERNAME="your-username"
export HF_TOKEN="your-huggingface-token"
```

4. Run AutoTrain:

```bash
autotrain --config conf.yaml
```

## Tips for Effective Fine-tuning

1. Start with a pre-trained model that's close to your target domain.
2. Use a learning rate that's typically an order of magnitude lower than the original training rate.
3. Experiment with different PEFT (Parameter-Efficient Fine-Tuning) methods like LoRA.
4. Monitor training logs to detect overfitting or other issues.
5. Use mixed precision training to save memory and speed up the process.
6. Adjust batch size and gradient accumulation steps based on your hardware capabilities.

Remember to adhere to the licensing terms of the base model you're fine-tuning, especially when using models like Llama-2 which may have specific usage restrictions.
