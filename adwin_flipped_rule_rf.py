import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

# Prepare training data
X_train = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])
y_train = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])

# Apply SMOTE if enough fraud
fraud_count = sum(y_train == 1)
if fraud_count >= 2:
    k = min(5, fraud_count - 1)
    sm = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"✅ SMOTE applied (k={k}) | Fraud count = {fraud_count}")
else:
    X_res, y_res = X_train, y_train
    print("⚠️ Not enough frauds for SMOTE")

# Scale features
scaler = StandardScaler()
# X_res_scaled = scaler.fit_transform(X_res)
X_res_scaled = pd.DataFrame(scaler.fit_transform(X_res), columns=X.columns)


# Train initial model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_res_scaled, y_res)
print("Class weight:", model.get_params()["class_weight"])
print("Number of estimators:", model.get_params()["n_estimators"])
# Get feature importances as a Series
importances = pd.Series(model.feature_importances_, index=X.columns)

# Sort and print
importances = importances.sort_values(ascending=False)
print("\n🌟 Feature Importances:\n")
# print(importances)
# Simulate drift in chunks 20–29
# Simulate drift by altering the most influential features in chunks 20–29
for i in range(20, 30):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    
    df.loc[start:end-1, "V14"] *= -2       # Invert and exaggerate
    df.loc[start:end-1, "V3"] += 10        # Positive shift
    df.loc[start:end-1, "V10"] -= 5        # Negative shift
    df.loc[start:end-1, "V4"] *= 3         # Amplify
    df.loc[start:end-1, "V12"] = df.loc[start:end-1, "V12"].apply(lambda x: x**2 if x > 0 else x)
    df.loc[start:end-1, "V17"] += 8        # Shift upward
    
    # Optionally, flip labels to simulate mislabeling or fraud confusion
    # df.loc[start:end-1, "Class"] = 1

print("🔀 Simulated concept drift on top features V14, V3, V10, V4, V12, V17 ")

# Initialize ADWIN
adwin = ADWIN(delta=0.0005)

# Evaluation setup
y_true_all = []
y_pred_all = []
drift_points = []
latencies = []
rule_mode_count = 0
model_mode_count = 0

buffer_X = []
buffer_y = []

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
        if error: 
            print(idx, y_pred_prob,true_label)
        drift = adwin.update(error)

        latency = time.time() - start_time
        latencies.append(latency)

        if adwin.drift_detected:
            print(f"\n⚠️ Drift detected at row {idx} | Switching to rule-based mode.")
            drift_points.append(idx)
            rule_mode_count += 1

            # Rule-based fallback
            amount = row["Amount"]
            v14 = row["V14"]
            v17 = row["V17"]
            rule_pred = 1 if (amount > 10000 or v14 < -50 or v17 > 20) else 0
            y_pred_all.append(rule_pred)
            y_true_all.append(true_label)

            # Collect data to retrain
            buffer_X.append(row)
            buffer_y.append(true_label)

            # if len(buffer_X) >= 100:
            #     X_buf = pd.DataFrame(buffer_X)
            #     y_buf = pd.Series(buffer_y)
            #     X_buf_scaled = scaler.fit_transform(X_buf)

            #     model = LogisticRegression(max_iter=200, class_weight="balanced")
            #     model.fit(X_buf_scaled, y_buf)
            #     print("✅ Retraining complete. Switching back to model-based prediction.")
            #     buffer_X.clear()
                # buffer_y.clear()
        else:
            model_mode_count += 1
            y_pred_all.append(y_pred)
            y_true_all.append(true_label)

# Final Evaluation
print("\n📊 Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4))
print(f"📍 Drift points detected at rows: {drift_points}")
print(f"🔁 Rule-based mode used: {rule_mode_count} times")
print(f"🧠 Model-based mode used: {model_mode_count} times")
print(f"✅ Overall accuracy: {np.mean(np.array(y_true_all) == np.array(y_pred_all)):.4f}")
print(f"\n⏱ Avg Inference Latency: {np.mean(latencies):.6f} seconds")
print(f"⏱ Max Inference Latency: {np.max(latencies):.6f} seconds")
print(f"⏱ Min Inference Latency: {np.min(latencies):.6f} seconds")
