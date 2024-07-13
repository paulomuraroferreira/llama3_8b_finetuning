import os
import time
import random
from typing import Dict, Any

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

class ModelConfig:
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B"
        self.use_flash_attention2 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        self.bnb_config = self._get_bnb_config()
        self.peft_config = self._get_peft_config()

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if self.use_flash_attention2 else torch.float16
        )

    def _get_peft_config(self) -> LoraConfig:
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        )

class DatasetLoader:
    @staticmethod
    def load_dataset(file_path: str):
        return load_dataset("json", data_files=file_path, split="train")

    @staticmethod
    def format_instruction(sample: Dict[str, str]) -> str:
        return f"""    
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
"""

class ModelLoader:
    def __init__(self, config: ModelConfig):
        self.config = config
        load_dotenv()

    def load_model_and_tokenizer(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, 
            quantization_config=self.config.bnb_config, 
            use_cache=False, 
            device_map="auto",
            token=os.environ["HF_TOKEN"],
            attn_implementation="flash_attention_2" if self.config.use_flash_attention2 else "sdpa"
        )
        model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            token=os.environ["HF_TOKEN"],
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def save_model_and_tokenizer(self, model, tokenizer, save_directory: str):
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

class Trainer:
    def __init__(self, model, dataset, tokenizer, config: ModelConfig):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config

    def prepare_model(self):
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.config.peft_config)

    def get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir="fine-tuned-snptee",
            num_train_epochs=1,
            per_device_train_batch_size=6 if self.config.use_flash_attention2 else 2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=self.config.use_flash_attention2,
            fp16=not self.config.use_flash_attention2,
            tf32=self.config.use_flash_attention2,
            max_grad_norm=0.3,
            warmup_steps=5,
            lr_scheduler_type="linear",
            disable_tqdm=False,
            report_to="none"
        )

    def train(self):
        args = self.get_training_args()
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.config.peft_config,
            max_seq_length=2048,
            tokenizer=self.tokenizer,
            packing=True,
            formatting_func=DatasetLoader.format_instruction, 
            args=args,
        )
        trainer.train()
        trainer.save_model()

def main():
    random.seed(42)
    
    config = ModelConfig()
    model_loader = ModelLoader(config)
    model, tokenizer = model_loader.load_model_and_tokenizer()
    
    dataset = DatasetLoader.load_dataset("/home/paulo/Python_projects/llama3_7b_fine_tunning/arvix_instruction_dataset.json")
    
    trainer = Trainer(model, dataset, tokenizer, config)
    trainer.prepare_model()
    trainer.train()

if __name__ == "__main__":
    main()