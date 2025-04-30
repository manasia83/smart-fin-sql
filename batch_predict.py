import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model + tokenizer
model_dir = "./t5_fixed_income_model_compact"  # adjust if needed
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.eval().to("cpu")

# Custom hardcoded queries from your screenshots
queries = [
    "Can you show all oci changes?",
    "Show holdings where amortized cost is greater than 500",
    "Can you show top 5 securities with highest coupon rate?",
    "Show portfolios that contain the word 'sovereign'",    
    "Can you show securities with currency in usd or eur?",
    "Show securities where maturity date is missing"
]

print("\n=== 📘 Natural Language to SQL Predictions ===\n")

for nl in queries:
    prompt = f"translate to SQL: {nl.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    sql = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"🟦 NL Query : {nl}")
    print(f"🟨 SQL      : {sql}")
    print("-" * 60)

print("\n✅ Done.\n")
