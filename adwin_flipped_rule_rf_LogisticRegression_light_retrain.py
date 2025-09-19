import pandas as pd
from river.drift import ADWIN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import deque
import time
import numpy as np

# Load and sort data
df = pd.read_csv("creditcard.csv")
df = df.sort_values(by="Time").reset_index(drop=True)

X = df.drop("Class", axis=1)
y = df["Class"]

chunk_size = 1000
train_chunks = 20
predict_chunks = 10
history_buffer = deque(maxlen=2000)

# Prepare training data from first 20 chunks
X_train = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])
y_train = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])

# Apply SMOTE
fraud_count = sum(y_train == 1)
if fraud_count >= 2:
    k = min(5, fraud_count - 1)
    sm = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"✅ SMOTE applied (k={k}) | Fraud count = {fraud_count}")
else:
    X_res, y_res = X_train, y_train
    print("⚠️ Skipped SMOTE: too few fraud cases")

# Train initial model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_res, y_res)

# Fit scaler to training data to avoid "not fitted" error
scaler = StandardScaler()
scaler.fit(X_res)

# Simulate drift by flipping labels in chunks 20–29
for i in range(20, 30): 
    start = i * chunk_size 
    end = (i + 1) * chunk_size 
    if i % 2 == 0:
        df.loc[start:end-1, "Amount"] *= 100

print("🔀 Simulated concept drift: Scaled Amount by 100x in chunks 20–29")

# Re-split data after label flipping
X = df.drop("Class", axis=1)
y = df["Class"]

# Drift detector
adwin = ADWIN(delta=0.001)
use_rule = False
retraining = False
drift_points = []
latencies = []
retrain_latencies = []
y_true_all = []
y_pred_all = []
mode_log = []

# Rule-based function
def rule_predictor(row):
    return int(row["Amount"] > 1500 or row["V14"] < -5)

# Streaming prediction loop
print("\n🚀 Starting Stream Prediction...\n")

for i in range(train_chunks, train_chunks + predict_chunks):
    X_chunk = X.iloc[i*chunk_size:(i+1)*chunk_size]
    y_chunk = y.iloc[i*chunk_size:(i+1)*chunk_size]

    for idx, row in X_chunk.iterrows():
        row_df = pd.DataFrame([row], columns=X.columns)
        true_label = y_chunk.loc[idx]
        history_buffer.append((row.values, true_label))

        # Measure inference latency
        start_time = time.perf_counter()

        if use_rule:
            pred = rule_predictor(row)
        else:
            row_scaled = pd.DataFrame(scaler.transform(row_df), columns=X.columns)
            y_pred_prob = model.predict_proba(row_scaled)[0][1]
            pred = int(y_pred_prob > 0.5)

        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

        y_true_all.append(true_label)
        y_pred_all.append(pred)
        mode_log.append("Rule" if use_rule else "Model")

        # Detect drift
        error = int(pred != true_label)
        adwin.update(error)
        if adwin.drift_detected and not retraining:
            drift_row = i * chunk_size + idx
            print(f"\n⚠️ Drift detected at row {drift_row} | Switching to rule-based mode.")
            use_rule = True
            retraining = True
            drift_points.append(drift_row)

            # Retraining process
            X_hist = pd.DataFrame([x for x, _ in history_buffer], columns=X.columns)
            y_hist = pd.Series([y for _, y in history_buffer])

            retrain_start = time.perf_counter()

            try:
                frauds = sum(y_hist == 1)
                k = min(5, frauds - 1) if frauds > 1 else 1
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_hist_res, y_hist_res = sm.fit_resample(X_hist, y_hist)
            except:
                X_hist_res, y_hist_res = X_hist, y_hist

            X_scaled = pd.DataFrame(scaler.fit_transform(X_hist_res), columns=X.columns)
            new_model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
            new_model.fit(X_scaled, y_hist_res)

            retrain_end = time.perf_counter()
            retrain_latencies.append(retrain_end - retrain_start)

            model = new_model
            use_rule = False
            retraining = False
            print("✅ Retraining complete. Switching back to model-based prediction.\n")

# Evaluation
print("\n📊 Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4))
print(f"📍 Drift points detected at rows: {drift_points}")
print(f"🔁 Rule-based mode used: {mode_log.count('Rule')} times")
print(f"🧠 Model-based mode used: {mode_log.count('Model')} times")
print(f"✅ Overall accuracy: {accuracy_score(y_true_all, y_pred_all):.4f}")

# Latency summary
print(f"\n⏱ Avg Inference Latency: {np.mean(latencies):.6f} seconds")
print(f"⏱ Max Inference Latency: {np.max(latencies):.6f} seconds")
print(f"⏱ Min Inference Latency: {np.min(latencies):.6f} seconds")

if retrain_latencies:
    print(f"🔁 Avg Retraining Time: {np.mean(retrain_latencies):.2f} seconds")
    print(f"🔁 Total Retraining Events: {len(retrain_latencies)}")
