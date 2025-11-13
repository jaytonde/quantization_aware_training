# Quantization-Aware Training (QAT) vs Post-Training Quantization (PTQ)

This repository contains code for comparing **Quantization-Aware Training (QAT)** and **Post-Training Quantization (PTQ)** techniques for large language models using the Qwen3-4B model.

## Overview

This project demonstrates how to apply different quantization strategies to reduce model size while maintaining performance:

- **PTQ (Post-Training Quantization)**: Quantize a pre-trained model after training
- **QAT (Quantization-Aware Training)**: Train the model with quantization in mind

## Features

- Fine-tune Qwen3-4B model using LoRA adapters
- Apply INT4 quantization using TorchAO
- Compare model performance between QAT and PTQ approaches
- WandB integration for experiment tracking
- Automated model upload to Hugging Face Hub

## Requirements

```bash
pip install unsloth
pip install torchao==0.14.0 fbgemm-gpu-genai==1.3.0
pip install transformers==4.55.4
pip install --no-deps trl==0.22.2
pip install wandb
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/quantization_aware_training.git
   cd quantization_aware_training
   ```

2. **Choose quantization type**
   
   Edit the notebook and set:
   ```python
   QUANTIZATION_TYPE = "PTQ"  # or "QAT"
   ```

3. **Run the notebook**
   
   Open and run [`quantization.ipynb`](quantization.ipynb) in Jupyter or Google Colab.

## Workflow

1. **Load Model**: Load Qwen3-4B-Instruct model
2. **Add LoRA**: Apply LoRA adapters for parameter-efficient fine-tuning
3. **Fine-tune**: Train on FineTome-100k dataset
4. **Quantize**: Apply INT4 quantization (QAT or PTQ)
5. **Save & Upload**: Save model and push to Hugging Face Hub

## Dataset

The notebook uses the [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset for fine-tuning.

## Models

Trained models are available on Hugging Face:

* **Trained Models:**
      * [Qwen3-4B-Baseline (BF16)](https://huggingface.co/jaytonde05/Qwen3_4B_baseline/tree/main)
      * [Qwen3-4B-PTQ (4-bit)](https://huggingface.co/jaytonde05/Qwen3_4B_PTQ/tree/main)
      * [Qwen3-4B-QAT (4-bit)](https://huggingface.co/jaytonde05/Qwen3_4B_QAT-torchao/tree/main)
  * **Wandb runs:**
      * [Qwen3-4B-QAT (4-bit)](https://wandb.ai/jaytonde05/QuantizationTraining/runs/qj03sk8d?nw=nwuserjaytonde05)
      * [Qwen3-4B-PTQ (4-bit)](https://wandb.ai/jaytonde05/QuantizationTraining/runs/1txptynn) 

## Key Components

- **Model**: Qwen3-4B-Instruct
- **Quantization**: INT4 weight-only quantization
- **Training**: LoRA fine-tuning with SFTTrainer
- **Monitoring**: Weights & Biases (WandB)

## Configuration

Key parameters you can adjust:

```python
r = 16                           # LoRA rank
learning_rate = 2e-5             # Learning rate
per_device_train_batch_size = 16 # Batch size
num_train_epochs = 1             # Training epochs
```

## Results

Compare model sizes and performance:

- **Baseline (BF16)**: ~8GB
- **PTQ/QAT (INT4)**: ~2GB (~4x compression)

## References

- [Unsloth QAT Documentation](https://docs.unsloth.ai/basics/quantization-aware-training-qat)
- [TorchAO GitHub](https://github.com/pytorch/ao)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-4B-Instruct)

## License

MIT License
