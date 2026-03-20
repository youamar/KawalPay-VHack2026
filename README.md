# 🛡️ KawalPay: Edge-Based Fraud Shield for the Unbanked

**Varsity Hackathon 2026 (V HACK 2026) Submission** **Track:** Case Study 2 - Digital Trust (Real-Time Fraud Shield for the Unbanked)  
**Primary Goal:** SDG 8: Decent Work and Economic Growth (Target 8.10)  

---

## 🚨 The Problem: The "Binary" Trap in Digital Trust
As Southeast Asia experiences a massive surge in digital payment adoption, millions of "unbanked" or low-digital-literacy users are entering the ecosystem. 

Current fraud detection systems are strictly binary:
1. **Too Loose:** They rely on simple rules, failing to catch sophisticated Social Engineering attacks (e.g., victims manipulated into transferring funds while on a phone call).
2. **Too Strict (High False Positives):** They lock accounts over minor anomalies (like logging in from a new village), forcing elderly users to navigate complex customer service hurdles, ultimately destroying trust in digital banking.

## 💡 Our Solution: "Dynamic Friction" & Contextual AI
**KawalPay** is a high-performance, edge-ready Risk API that bridges hard-core Cyber Security with empathetic Human-Computer Interaction (HCI). 

Instead of a simple "Approve or Block" approach, KawalPay introduces **Dynamic Friction**. By analyzing non-transactional contextual data (device states, behavioral biometrics), our AI intelligently categorizes risk and deploys zero-learning-curve visual challenges to break the psychological momentum of scams, without frustrating legitimate users.

### 🌟 Key Features
* **🧠 Contextual CyberSec Profiling:** Goes beyond transaction amounts to analyze `dwell_time_ms`, `clipboard_anomalies`, and `active_call_status` to detect social engineering and device hijacking.
* **⚖️ Imbalanced Class Handling:** Powered by **SMOTE** oversampling and an **XGBoost** classifier, ensuring the model accurately identifies the 0.01% of fraudulent anomalies without drowning in false positives.
* **⚡ Ultra-Low Latency API:** Built with **FastAPI** to evaluate complex behavioral telemetry and return risk scores in milliseconds, ensuring seamless checkout flows.
* **🎯 Zero-Barrier Cognitive UI:** For "Medium Risk" scenarios, we bypass traditional SMS OTPs (which are easily phished). Instead, we trigger localized, zero-literacy visual challenges (e.g., *"Click the Nasi Lemak to verify"*).

---

## 🏗️ Technical Architecture & Under The Hood

Our prototype is divided into three core components:

1. **The Edge Context Engine (Simulated):** Extracts deep OS-level state and behavioral biometrics at the moment of transaction.
2. **The ML Risk Brain (`train_model.py`):** * Trained on heavily imbalanced synthetic datasets mimicking real-world SME/unbanked transaction patterns.
   * Utilizes XGBoost for high-precision probability scoring rather than hard classifications.
3. **The API & HCI Gateway (`main.py` & `index.html`):**
   * A FastAPI backend routing the Risk Matrix.
   * A tailored Front-End demonstrating the graceful degradation of user experience based on the exact AI risk score (0-100%).

### Risk Decision Matrix
| AI Risk Score | Threat Level | KawalPay Action | User Experience (HCI) |
| :--- | :--- | :--- | :--- |
| **0% - 14%** | Safe / Familiar | `APPROVE` | Frictionless. Instant transfer. |
| **15% - 94%** | Suspicious / Anomaly | `CHALLENGE` | **Dynamic Friction.** Triggers local visual cognitive test (e.g., Image selection). |
| **95% - 100%** | Critical / Hijack | `BLOCK` | Physical transaction circuit breaker. Account frozen. |

---

## 💼 Business Model & Market Potential
KawalPay operates on a **B2B2C** model. We provide our Risk Engine as a scalable API/SDK for regional "Super Apps", rural co-op banks, and micro-lending platforms. 
* **Revenue Stream:** Pay-per-API-call or tiered enterprise licensing.
* **Value Proposition:** We reduce the operational costs of handling false-positive customer support tickets while preventing catastrophic PR disasters caused by social engineering scams.

---

## 🚀 How to Run the Prototype Locally

**Prerequisites:** Python 3.10+, `pip`

**1. Clone and Install Dependencies**
```bash
git clone [YOUR_GITHUB_REPO_LINK]
cd KawalPay_Project
pip install pandas scikit-learn xgboost imbalanced-learn fastapi uvicorn pydantic