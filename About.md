
<p align="center">
  <img src="https://img.shields.io/badge/Model-T5--small-blue" />
  <img src="https://img.shields.io/badge/Status-Prototyped-green" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

# 📊 SmartFin SQL: Natural Language to SQL

**Smart-Fin-SQL** is a compact, fine-tuned Natural Language to SQL (NL2SQL) model designed specifically for **Fixed Income Portfolios**. It enables finance teams, analysts, and testers to run complex SQL queries on structured bond portfolio databases using plain English.

---

## 🙋‍♂️ Who Can Use This?

- Finance teams running daily ad-hoc reports  
- Testers validating data pipelines  
- Analysts who don’t know SQL syntax  

---

## 🚀 Features

- Trained on a curated dataset of ~50+ NL query templates with domain knowledge  
- Uses `T5-small` transformer for fast inference (CPU-friendly)  
- Clean web UI to enter natural language → see SQL → view query results  
- FastAPI backend with `/predict` and `/run-query` endpoints
- Secure — only SELECT queries are allowed  

---

## ⚙️ How It Works

1. User enters a natural language query via UI or API.  
2. Backend invokes fine-tuned `T5-small` model to generate SQL.  
3. SQL is executed against MS SQL Server (SELECT only).  
4. Result is rendered as a formatted table.

---

## 🔎 Example Queries and Output

| 💬 Natural Language Query                                        | 🧠 Generated SQL                                                                                          |
|------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Can you show each portfolio and the ISIN of securities it holds? | `SELECT p.PortfolioName, f.ISIN FROM PortfolioHoldings h JOIN Portfolios p ON h.PortfolioId = p.PortfolioId JOIN FixedIncomeSecurities f ON h.SecurityId = f.SecurityId` |
| Show top 5 securities with highest coupon rate                   | `SELECT TOP 5 * FROM FixedIncomeSecurities ORDER BY CouponRate DESC`                                     |
| What are the securities with currency in USD or EUR?            | `SELECT * FROM FixedIncomeSecurities WHERE CurrencyCode IN ('USD', 'EUR')`                                |
| List holdings with market value above average                   | `SELECT * FROM Valuations WHERE MarketValue > (SELECT AVG(MarketValue) FROM Valuations)`                 |
| List portfolio names and ISINs for holdings after 2020          | `SELECT p.PortfolioName, f.ISIN FROM PortfolioHoldings h JOIN Portfolios p ON h.PortfolioId = p.PortfolioId JOIN FixedIncomeSecurities f ON h.SecurityId = f.SecurityId WHERE h.PurchaseDate > '2020-01-01'` |

---

## 🧠 Model Architecture

- **Model**: `T5-small` (60M parameters)  
- **Framework**: Hugging Face Transformers  
- **Tokenizer**: SentencePiece (`T5Tokenizer`)  
- **Prompt Format**:  
  ```text
  fixed income query: {{nl_query}} [Tables: ...] [Columns: ...]
  ```

---

## 🔧 Model Training Highlights

- Trained ~6–8 hours on CPU-only (i5, 8GB RAM)  
- Dataset: 50+ handcrafted query-SQL pairs, expanded to 2000+  
- Logic covered: SELECT, WHERE, GROUP BY, ORDER BY, LIKE, IN, IS NULL, HAVING, JOIN  

---

## ⚙️ Training Configuration

```python
TrainingArguments(
    num_train_epochs=30,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    warmup_steps=100,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=50,
    fp16=False,
    logging_dir="./logs"
)
```

---

## 🧪 Key Training Enhancements

- Removed duplicate numeric-only prompts  
- Used structured prompts with table/column hints  
- Domain knowledge included (OCI, amortized cost, etc.)  

---

## 🧠 Other Pretrained Models You Can Try

| Model          | Description                               | Best For                     |
|----------------|-------------------------------------------|------------------------------|
| `t5-base`      | 220M params, better accuracy, slower CPU  | More RAM/GPU environments    |
| `codeT5-small` | Pretrained on code + NL                   | SQL or code-gen tasks        |
| `flan-t5-small`| Instruction-tuned version of t5-small     | Better generalization        |

---

## 📈 Tips to Improve Performance

- Add table schema to prompts (entity-aware tuning)  
- Use more diverse phrasings with conditions/grouping  
- Upgrade model size (T5-base or Flan-T5) if RAM permits  
- Fine-tune using domain-specific edge cases  

---

## 🚀 Deployment

- Run API server:
  ```bash
  python api_server.py
  ```
- Open `index.html` or deploy via Nginx + FastAPI  
- Endpoints:
  - `/predict`: NL → SQL  
  - `/run-query`: SQL → Result (SELECT only)  

---

## 📂 Folder Structure

```
FIN_SQL/
├── api_server.py
├── ui/index.html
├── model/t5_fixed_income_model_compact/
├── data/
├── batch_predict.py
├── train_compact.py
├── DS_fixed_income_training_generator.py #Training Data script using Gen AI
├── README.md
├── About.html
└── requirements.txt
```

---

## 📥 Installation

```bash
pip install -r requirements.txt
python api_server.py
# Visit http://localhost:8000
```


---

## ⚠️ Limitations

- Small dataset (~2000 samples) due to CPU-only training  
- No GPU used — limited model size and coverage  
- Complex SQL constructs like CTEs, nested queries not supported  
- SELECT-only logic — safe, but not full CRUD coverage  

---

## 📜 Future Enhancements

- Add chatbot interface
- Future support for fallback mechanism on invalid prompts
- Trained on queries  only supporting MS SQL database

