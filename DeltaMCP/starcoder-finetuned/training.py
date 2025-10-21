#!/usr/bin/env python3

import json
import logging
from pathlib import Path
import torch
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir):
    samples = []
    for json_file in Path(data_dir).glob("*.json"):
        if json_file.name.endswith('.log'):
            continue
        try:
            with open(json_file, 'r') as f:
                samples.append(json.load(f))
        except:
            pass
    return samples

def create_dataset(samples):
    instructions, inputs, outputs = [], [], []
    
    for sample in samples:
        try:
            oasdiff = sample.get('oasdiff', {})
            tools_a = sample.get('tools_a', {})
            tools_b = sample.get('tools_b', {})
            
            if not tools_a or not tools_b:
                continue
                
            instruction = "Generate updated server stub tool functions based on API specification changes."
            input_text = f"API Diff:\n{json.dumps(oasdiff, indent=2)}\n\nOriginal Tools:\n{json.dumps(tools_a, indent=2)}"
            output_text = json.dumps(tools_b, indent=2)
            
            if len(input_text) + len(output_text) > 12000:
                continue
                
            instructions.append(instruction)
            inputs.append(input_text)
            outputs.append(output_text)
        except:
            continue
    
    return Dataset.from_dict({'instruction': instructions, 'input': inputs, 'output': outputs})

class InstructionDataset:
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prompt = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
        
        encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding=False, return_tensors="pt")
        input_ids = encoding["input_ids"][0]
        labels = input_ids.clone()
        
        response_ids = self.tokenizer.encode("### Response:", add_special_tokens=False)
        for i in range(len(input_ids) - len(response_ids) + 1):
            if torch.equal(input_ids[i:i+len(response_ids)], torch.tensor(response_ids)):
                labels[:i+len(response_ids)] = -100
                break
        
        return {"input_ids": input_ids, "attention_mask": encoding["attention_mask"][0], "labels": labels}

def setup_model():
    model_name = "bigcode/starcoder2-3b"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def train():
    set_seed(42)
    
    samples = load_data("training_data")
    dataset = create_dataset(samples)
    
    if len(dataset) == 0:
        logger.error("No training data found!")
        return
    
    dataset = dataset.shuffle(seed=42)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    model, tokenizer = setup_model()
    
    train_dataset = InstructionDataset(train_dataset, tokenizer)
    val_dataset = InstructionDataset(val_dataset, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        remove_unused_columns=False,
        group_by_length=True,
        report_to=None,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    output_dir = f"./finetuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Training complete! Model saved to: {output_dir}")

if __name__ == "__main__":
    train()