from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

dataset = load_dataset("json", data_files={
    "train": "dataset/train.jsonl",
    "validation": "dataset/val.jsonl"
})

# 2. Load model + tokenizer
model_name = "microsoft/phi-2"  # or "bigcode/starcoderbase-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenize
def tokenize(batch):
    return tokenizer(
        batch["prompt"],
        text_target=batch["completion"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["prompt", "completion"])

# 4. Apply LoRA for efficient fine-tuning
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
model = get_peft_model(model, config)

# 5. Training arguments
args = TrainingArguments(
    output_dir="finetuned-delta-mcp",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    evaluation_strategy="steps",
    save_strategy="epoch",
    logging_steps=20,
    report_to="none",
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("finetuned-delta-mcp")
tokenizer.save_pretrained("finetuned-delta-mcp")
