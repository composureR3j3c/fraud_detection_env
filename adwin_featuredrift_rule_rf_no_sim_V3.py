# sparated measures
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from river.drift import ADWIN

# Load and sort data
print(f"start time={datetime.datetime.now()}")
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

adwin = ADWIN(delta=0.0005)
rolling_window = [] 
prev_row_vals = None

latencies = []
drift_points = []
y_true_all = []
y_pred_all = []
y_true_rule = []
y_pred_rule = []
rule_mode_count = 0
model_mode_count = 0

print("\n🚀 Starting Stream Prediction...\n")
cooldown_period = 500 # Number of rows between accepted drifts 
        
last_drift_row = -cooldown_period # Initialize before your main loop
for i in range(train_chunks, train_chunks + predict_chunks):
    X_chunk = X.iloc[i*chunk_size:(i+1)*chunk_size]
    y_chunk = y.iloc[i*chunk_size:(i+1)*chunk_size]

    for idx, row in X_chunk.iterrows():
        row_df = pd.DataFrame([row], columns=X.columns)
        row_scaled = pd.DataFrame(scaler.transform(row_df), columns=X.columns)
        true_label = y_chunk.loc[idx]

        y_pred_prob = model.predict_proba(row_scaled)[0][1]
        y_pred = int(y_pred_prob > 0.5)
        error = int(y_pred != true_label)
        drift = adwin.update(error)

         # Feature drift
        top_features = ['V14', 'V3', 'V10', 'V4', 'V12', 'V17']
        current_vals = row_scaled[top_features].values[0]
        feature_drift = False
        if prev_row_vals is not None:
            diff = np.abs(current_vals - prev_row_vals)
            if len(rolling_window) >= 30:
                std_dev = np.std(rolling_window, axis=0)
                feature_drift = np.any((diff > 2.5) & (diff > 3 * std_dev))
                rolling_window.pop(0)
            rolling_window.append(diff)
        else:
            rolling_window.append(np.zeros_like(current_vals))
        prev_row_vals = current_vals

        # Start timer
        start_time = time.time()
        
        if (adwin.drift_detected or feature_drift) and (idx - last_drift_row >= cooldown_period): 
            print(f"\n⚠️ Drift detected at row {idx} | Switching to rule-based mode.") 
            drift_points.append(idx) 
            last_drift_row = idx # Update the last drift trigger point 
            rule_mode_count += 1 
            amount = row["Amount"] 
            v14 = row["V14"] 
            v17 = row["V17"] 
            rule_pred = 1 if (amount > 10000 or v14 < -50 or v17 > 20) else 0 
            y_pred_rule.append(rule_pred) 
            y_true_rule.append(true_label) 
        else: 
            model_mode_count += 1 
            y_pred_all.append(y_pred)
            y_true_all.append(true_label)

        latency = time.time() - start_time
        latencies.append(latency)

# Final Evaluation for Model-Based Predictions
print("\n🧠 Model-based Predictions Report:\n")
print(classification_report(y_true_all, y_pred_all, digits=4))

# Final Evaluation for Rule-Based Predictions
print("\n🔁 Rule-based Predictions Report:\n")
print(classification_report(y_true_rule, y_pred_rule, digits=4))

# Overall Performance Evaluation
print("\n📊 Final Overall Evaluation:\n")
print(f"📍 Drift points detected at rows: {drift_points}")
print(f"🔁 Rule-based mode used: {rule_mode_count} times")
print(f"🧠 Model-based mode used: {model_mode_count} times")
print(f"✅ Overall accuracy: {np.mean(np.array(y_true_all) == np.array(y_pred_all)):.4f}")

# Latency Statistics
print(f"\n⏱ Avg Inference Latency: {np.mean(latencies):.6f} seconds")
print(f"⏱ Max Inference Latency: {np.max(latencies):.6f} seconds")
print(f"⏱ Min Inference Latency: {np.min(latencies):.6f} seconds")
print(f"Finish time={datetime.datetime.now()}")

