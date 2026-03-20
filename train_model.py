import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle

print("🚀 KawalPay: Starting to generate mock training data...")

# 1. Generate Mock Data
# We generate 10,000 normal transactions and only 100 fraud transactions (highly imbalanced)
np.random.seed(42)

def generate_data(num_samples, is_fraud):
    return pd.DataFrame({
        'amount_myr': np.random.uniform(10, 5000, num_samples),
        'is_new_payee': np.random.choice([0, 1], num_samples, p=[0.8, 0.2] if not is_fraud else [0.1, 0.9]),
        'is_pasted_from_clipboard': np.random.choice([0, 1], num_samples, p=[0.95, 0.05] if not is_fraud else [0.2, 0.8]),
        'is_rooted_or_jailbroken': np.random.choice([0, 1], num_samples, p=[0.99, 0.01] if not is_fraud else [0.4, 0.6]),
        'is_active_call_ongoing': np.random.choice([0, 1], num_samples, p=[0.9, 0.1] if not is_fraud else [0.3, 0.7]),
        'dwell_time_ms': np.random.uniform(1000, 5000, num_samples) if not is_fraud else np.random.uniform(20000, 60000, num_samples),
        'is_fraud': 1 if is_fraud else 0 # Label: 1 is fraud, 0 is normal
    })

normal_tx = generate_data(10000, is_fraud=False)
fraud_tx = generate_data(100, is_fraud=True)
df = pd.concat([normal_tx, fraud_tx]).sample(frac=1).reset_index(drop=True)

# Split features (X) and target labels (y)
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print(f"📊 Original Data: Normal transactions {sum(y==0)}, Fraud transactions {sum(y==1)}.")

# 2. The Magic Moment: Using SMOTE to handle imbalanced data
# SMOTE observes the 100 fraud samples and creates reasonable synthetic samples through mathematical interpolation to balance the ratio.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"⚖️ After SMOTE sampling: Normal transactions {sum(y_resampled==0)}, Fraud transactions {sum(y_resampled==1)}.")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 3. Train the XGBoost Model
print("🧠 Training XGBoost risk perception model...")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 4. Save the Model (save the trained brain)
with open('kawalpay_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("✅ Model training complete! Saved as 'kawalpay_model.pkl'")