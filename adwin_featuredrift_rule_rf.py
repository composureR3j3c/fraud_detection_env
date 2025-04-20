import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from river.drift import ADWIN

# Load and sort data
df = pd.read_csv("creditcard.csv")
df = df.sort_values(by="Time").reset_index(drop=True)

X = df.drop("Class", axis=1)
y = df["Class"]

chunk_size = 1000
train_chunks = 30
predict_chunks = 30

X_train = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])
y_train = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])

fraud_count = sum(y_train == 1)
if fraud_count >= 2:
    k = min(5, fraud_count - 1)
    sm = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"✅ SMOTE applied (k={k}) | Fraud count = {fraud_count}")
else:
    X_res, y_res = X_train, y_train
    print("⚠️ Not enough frauds for SMOTE")

scaler = StandardScaler()
X_res_scaled = pd.DataFrame(scaler.fit_transform(X_res), columns=X.columns)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_res_scaled, y_res)

# Feature drift tracking
feature_means = X_res.mean()
feature_threshold = 2.5

# Simulate concept drift
for i in range(20, 30):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    df.loc[start:end-1, "V14"] *= -2
    df.loc[start:end-1, "V3"] += 10
    df.loc[start:end-1, "V10"] -= 5
    df.loc[start:end-1, "V4"] *= 3
    df.loc[start:end-1, "V12"] = df.loc[start:end-1, "V12"].apply(lambda x: x**2 if x > 0 else x)
    df.loc[start:end-1, "V17"] += 8

print("🔀 Simulated concept drift on V14, V3, V10, V4, V12, V17")

adwin = ADWIN(delta=0.0005)

latencies = []
drift_points = []
y_true_all = []
y_pred_all = []
rule_mode_count = 0
model_mode_count = 0

print("\n🚀 Starting Stream Prediction...\n")

for i in range(train_chunks, train_chunks + predict_chunks):
    X_chunk = X.iloc[i*chunk_size:(i+1)*chunk_size]
    y_chunk = y.iloc[i*chunk_size:(i+1)*chunk_size]

    for idx, row in X_chunk.iterrows():
        row_df = pd.DataFrame([row], columns=X.columns)
        row_scaled = pd.DataFrame(scaler.transform(row_df), columns=X.columns)
        true_label = y_chunk.loc[idx]

        start_time = time.time()
        y_pred_prob = model.predict_proba(row_scaled)[0][1]
        y_pred = int(y_pred_prob > 0.5)
        error = int(y_pred != true_label)
        drift = adwin.update(error)

        feature_drift = any(abs(row[feat] - feature_means[feat]) > feature_threshold for feat in ["V14", "V3", "V10", "V4", "V12", "V17"])

        if adwin.drift_detected or feature_drift:
            print(f"\n⚠️ Drift detected at row {idx} | Switching to rule-based mode.")
            drift_points.append(idx)
            rule_mode_count += 1
            amount = row["Amount"]
            v14 = row["V14"]
            v17 = row["V17"]
            rule_pred = 1 if (amount > 10000 or v14 < -50 or v17 > 20) else 0
            y_pred_all.append(rule_pred)
        else:
            model_mode_count += 1
            y_pred_all.append(y_pred)

        latency = time.time() - start_time
        latencies.append(latency)
        y_true_all.append(true_label)

print("\n📊 Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4))
print(f"📍 Drift points detected at rows: {drift_points}")
print(f"🔁 Rule-based mode used: {rule_mode_count} times")
print(f"🧠 Model-based mode used: {model_mode_count} times")
print(f"✅ Overall accuracy: {np.mean(np.array(y_true_all) == np.array(y_pred_all)):.4f}")
print(f"\n⏱ Avg Inference Latency: {np.mean(latencies):.6f} seconds")
print(f"⏱ Max Inference Latency: {np.max(latencies):.6f} seconds")
print(f"⏱ Min Inference Latency: {np.min(latencies):.6f} seconds")
