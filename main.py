from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

# 1. Start FastAPI application
app = FastAPI(title="KawalPay Risk API", description="Edge-based Fraud Shield for the Unbanked")

# 🚨 CORS Configuration: Allow web pages to connect to our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the model brain we just trained
with open('kawalpay_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 3. Define the received JSON data structure
class TransactionRequest(BaseModel):
    amount_myr: float
    is_new_payee: int
    is_pasted_from_clipboard: int
    is_rooted_or_jailbroken: int
    is_active_call_ongoing: int
    dwell_time_ms: float

# 4. Create risk evaluation endpoint (Core logic)
@app.post("/api/v1/evaluate_risk")
async def evaluate_risk(request: TransactionRequest):
    # Convert the JSON from the frontend into a dataframe the model can understand
    input_data = pd.DataFrame([{
        'amount_myr': request.amount_myr,
        'is_new_payee': request.is_new_payee,
        'is_pasted_from_clipboard': request.is_pasted_from_clipboard,
        'is_rooted_or_jailbroken': request.is_rooted_or_jailbroken,
        'is_active_call_ongoing': request.is_active_call_ongoing,
        'dwell_time_ms': request.dwell_time_ms
    }])

    # Get risk probability (a decimal between 0 and 1 output by the model)
    risk_probability = float(model.predict_proba(input_data)[0][1])
    risk_score = round(risk_probability * 100, 2)

    # 5. Implement your "Risk Decision Matrix"
    if risk_score < 15:  
        # Only extremely low scores (absolutely safe) are allowed to pass directly
        action = "APPROVE"
        message = "✅ Frictionless Approval"
    elif risk_score < 95: 
        # Broaden the medium risk area (15% ~ 95%) to ensure it catches suspicious anomalies
        action = "CHALLENGE"
        message = "🟡 Dynamic Friction Triggered: Localized verification required"
    else:
        action = "BLOCK"
        message = "🔴 Critical Risk: Hardware-level circuit breaker activated"

    # Return results to the frontend
    return {
        "risk_score_percentage": risk_score,
        "action": action,
        "system_message": message
    }