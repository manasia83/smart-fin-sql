from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import uvicorn

import pyodbc
import pandas as pd

# --- DB Connection Setup ---
DB_CONN_STRING = (
         "Driver={ODBC Driver 17 for SQL Server};"
         "Server=(localdb)\\MSSQLLocalDB;"
         "Database=Test_FI_Portfolio;"  # Or your specific DB name
         "Trusted_Connection=yes;"
)



# === ✅ Load the fine-tuned model ===
MODEL_DIR = "./model/t5_fixed_income_model_compact"

tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval().to("cpu")

# === ✅ FastAPI setup ===
app = FastAPI()
templates = Jinja2Templates(directory="ui")

# === ✅ API request schema ===
class QueryRequest(BaseModel):
    question: str

# === ✅ SQL prediction helper ===
def predict_sql(nl_query: str) -> str:
    prompt = f"translate to SQL: {nl_query.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === ✅ Route: HTML Home Page ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Route to Run Generated SQL and Return Result Set ---
@app.post("/run-query")
async def run_sql_query(request: Request):
    data = await request.json()
    sql = data.get("sql", "").strip()

    if not sql.lower().startswith("select"):
        return JSONResponse(content={"error": "Only SELECT queries allowed"}, status_code=400)

    try:
        conn = pyodbc.connect(DB_CONN_STRING)
        df = pd.read_sql(sql, conn)
        conn.close()
        result = df.astype(str).to_dict(orient="records")  # ✅ Fix: Convert Timestamp to string
        return JSONResponse(content={"rows": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# === ✅ Route: Prediction via JSON ===
@app.post("/predict")
async def generate_sql(request: QueryRequest):
    if not request.question.strip():
        return JSONResponse(content={"sql": ""})
    
    sql = predict_sql(request.question)
    return JSONResponse(content={"sql": sql})

# === ✅ Local run ===
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
