import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import os

# Load cleaned training data
DATA_PATH = "./data/smartest_compact_training_dataset.csv"

df = pd.read_csv(DATA_PATH)
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess function
MAX_LEN = 256

def preprocess(examples):
    inputs = ["translate to SQL: " + q for q in examples["natural_language"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True, padding="max_length")
    labels = tokenizer(examples["sql_query"], max_length=MAX_LEN, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
dataset = dataset.map(preprocess, batched=True)

# Define training args
training_args = TrainingArguments(
    output_dir="./model/t5_fixed_income_model_compact",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=30,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=10,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    fp16=False,
    push_to_hub=False,
    report_to="none"
)
# training_args = TrainingArguments(
#     output_dir="./t5_fixed_income_model_compact",
#     overwrite_output_dir=True,
#     per_device_train_batch_size=4,  # Increased with better memory management
#     gradient_accumulation_steps=4,  # Better balance between memory and speed
#     num_train_epochs=30,  # Slightly more for better convergence
#     learning_rate=3e-4,  # Better for T5 architecture
#     weight_decay=0.01,
#     warmup_ratio=0.1,  # Better than absolute steps for small datasets
#     save_strategy="steps",
#     save_steps=500,  # More frequent checkpoints
#     save_total_limit=3,
#     logging_steps=100,
#     evaluation_strategy="no",
#     fp16=False,
#     push_to_hub=False,
#     report_to="none",
#     gradient_checkpointing=True,  # Memory optimization
#     dataloader_num_workers=2,  # Utilize multiple cores
#     optim="adafactor"  # Better for resource-constrained training
# )

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

if __name__ == "__main__":
    print("\n🚀 Starting training on CPU with memory-safe settings...")
    trainer.train()
    model.save_pretrained("./model/t5_fixed_income_model_compact")
    tokenizer.save_pretrained("./model/t5_fixed_income_model_compact")
    print("\n✅ Model training complete. Model saved to './model/t5_fixed_income_model_compact'")
