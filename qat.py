import os
import re
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats #what is this?

from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig
from torchao.quantization import Int4WeightOnlyConfig


def get_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = 2048,   # Choose any for long context!
    load_in_4bit = False,    # 4 bit quantization to reduce memory
    load_in_8bit = False,    # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    )

    return model, tokenizer


def add_lora_adapter():
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        qat_scheme = "int4",
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    return model

def get_dataset()
    dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
    return dataset

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def get_trainer():
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 30,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )

    return trainer

def show_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")



def main():
    model, tokenizer = get_model_and_tokenizer()
    model            = add_lora_adapter()

    for module in model.modules():
        if "FakeQuantized" in module.__class__.__name__:
            print("QAT is applied!")
            break

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen3-instruct",
    )

    dataset = get_dataset()
    dataset = standardize_data_formats(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True)

    trainer = get_trainer()

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    show_stats()

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    quantize_(model, QATConfig(step = "convert"))

    model.save_pretrained("lora_model")  # Local saving
    tokenizer.save_pretrained("lora_model")

    model.save_pretrained_torchao(
        "model",
        tokenizer,
        torchao_config = model._torchao_config.base_config,
    )

    model.save_pretrained_torchao("model", tokenizer, torchao_config = Int4WeightOnlyConfig())




