from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- 新增这行
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="KawalPay Risk API", description="Edge-based Fraud Shield for the Unbanked")

# 🚨 新增 CORS 配置：允许网页连接我们的 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('kawalpay_model.pkl', 'rb') as f:
    model = pickle.load(f)

class TransactionRequest(BaseModel):
    amount_myr: float
    is_new_payee: int
    is_pasted_from_clipboard: int
    is_rooted_or_jailbroken: int
    is_active_call_ongoing: int
    dwell_time_ms: float

@app.post("/api/v1/evaluate_risk")
async def evaluate_risk(request: TransactionRequest):
    input_data = pd.DataFrame([{
        'amount_myr': request.amount_myr,
        'is_new_payee': request.is_new_payee,
        'is_pasted_from_clipboard': request.is_pasted_from_clipboard,
        'is_rooted_or_jailbroken': request.is_rooted_or_jailbroken,
        'is_active_call_ongoing': request.is_active_call_ongoing,
        'dwell_time_ms': request.dwell_time_ms
    }])

    risk_probability = float(model.predict_proba(input_data)[0][1])
    risk_score = round(risk_probability * 100, 2)

    if risk_score < 40:
        action = "APPROVE"
        message = "✅ 极速无感通过"
    elif risk_score < 80:
        action = "CHALLENGE"
        message = "🟡 触发动态摩擦：需要本地化验证"
    else:
        action = "BLOCK"
        message = "🔴 极高风险：硬件熔断拦截"

    return {
        "risk_score_percentage": risk_score,
        "action": action,
        "system_message": message
    }