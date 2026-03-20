import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle

print("🚀 KawalPay: 开始生成模拟训练数据...")

# 1. 生成模拟数据 (Mock Data)
# 我们生成 10,000 条正常交易，和仅仅 100 条欺诈交易 (极度不平衡)
np.random.seed(42)

def generate_data(num_samples, is_fraud):
    return pd.DataFrame({
        'amount_myr': np.random.uniform(10, 5000, num_samples),
        'is_new_payee': np.random.choice([0, 1], num_samples, p=[0.8, 0.2] if not is_fraud else [0.1, 0.9]),
        'is_pasted_from_clipboard': np.random.choice([0, 1], num_samples, p=[0.95, 0.05] if not is_fraud else [0.2, 0.8]),
        'is_rooted_or_jailbroken': np.random.choice([0, 1], num_samples, p=[0.99, 0.01] if not is_fraud else [0.4, 0.6]),
        'is_active_call_ongoing': np.random.choice([0, 1], num_samples, p=[0.9, 0.1] if not is_fraud else [0.3, 0.7]),
        'dwell_time_ms': np.random.uniform(1000, 5000, num_samples) if not is_fraud else np.random.uniform(20000, 60000, num_samples),
        'is_fraud': 1 if is_fraud else 0 # 标签：1是坏人，0是好人
    })

normal_tx = generate_data(10000, is_fraud=False)
fraud_tx = generate_data(100, is_fraud=True)
df = pd.concat([normal_tx, fraud_tx]).sample(frac=1).reset_index(drop=True)

# 拆分特征 (X) 和 目标标签 (y)
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print(f"📊 原始数据: 正常交易 {sum(y==0)} 笔, 欺诈交易 {sum(y==1)} 笔。")

# 2. 魔法时刻：使用 SMOTE 处理不平衡数据
# SMOTE 会观察那 100 个欺诈样本，然后通过数学插值，"凭空"捏造出合理的假样本，让好坏比例平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"⚖️ SMOTE 采样后: 正常交易 {sum(y_resampled==0)} 笔, 欺诈交易 {sum(y_resampled==1)} 笔。")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 3. 训练 XGBoost 模型
print("🧠 正在训练 XGBoost 风险感知模型...")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 4. 保存模型 (把训练好的大脑存下来)
with open('kawalpay_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("✅ 模型训练完成！已保存为 'kawalpay_model.pkl'")