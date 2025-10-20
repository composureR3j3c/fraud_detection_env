import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text
from imblearn.over_sampling import SMOTE
from river.drift import ADWIN
import threading
import warnings

warnings.filterwarnings("ignore", message=".*fitted without feature names.*")

# Function: Mine rules from misclassified instances
def mine_rules_from_misclassified(X_recent, y_true, y_pred, feature_names, max_depth=2):
    misclassified_mask = (y_true != y_pred)
    X_mis = X_recent[misclassified_mask]
    y_mis = y_true[misclassified_mask]

    if len(X_mis) < 10 or y_mis.sum() == 0:
        return None, None

    clf = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=42)
    clf.fit(X_mis, y_mis)
    rules_text = export_text(clf, feature_names=feature_names.tolist())
    return rules_text, clf

# Load data
df = pd.read_csv("creditcard.csv")
df = df.sort_values(by="Time").reset_index(drop=True)

X = df.drop("Class", axis=1)
y = df["Class"]

chunk_size = 1000
train_chunks = 30
predict_chunks = 30
X_train = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])
y_train = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])

# SMOTE
fraud_count = sum(y_train == 1)
if fraud_count >= 2:
    k = min(5, fraud_count - 1)
    X_res, y_res = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_train, y_train)
    print(f" SMOTE applied (k={k}) | Fraud count = {fraud_count}")
else:
    X_res, y_res = X_train, y_train
    print(" Not enough frauds for SMOTE")

scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_res_scaled, y_res)

adwin = ADWIN(delta=0.0005)
rolling_window = []
prev_row_vals = None
cooldown_period = 500
last_drift_row = -cooldown_period

feature_names = X.columns
top_features = ['V14', 'V3', 'V10', 'V4', 'V12', 'V17']
feature_means = X_res[top_features].mean()

# Tracking
drift_points = []
latencies = []
y_true_all = []
y_pred_all = []
y_true_rule = []
y_pred_rule = []
y_true_model = []
y_pred_model = []

buffer_X = []
buffer_y = []
rule_model = None
rule_model_text = ""
retrain_lock = threading.Lock()
in_rule_mode = False
retraining_complete = True


def retrain_model_async(X_recent, y_recent):
    global model, scaler, retraining_complete, in_rule_mode, rule_model, rule_model_text
    try:
        k_local = min(5, sum(y_recent == 1) - 1)
        X_resampled, y_resampled = SMOTE(k_neighbors=k_local, random_state=42).fit_resample(X_recent, y_recent)
        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X_resampled)

        new_model = LogisticRegression(max_iter=200, class_weight="balanced")
        new_model.fit(X_scaled, y_resampled)

        y_pred_temp = new_model.predict(X_scaled)
        rules, rule_clf = mine_rules_from_misclassified(pd.DataFrame(X_resampled, columns=feature_names), y_resampled, y_pred_temp, feature_names)
        
        with retrain_lock:
            model = new_model
            scaler = new_scaler
            rule_model = rule_clf
            rule_model_text = rules or ""
            retraining_complete = True
            in_rule_mode = False
        print(" Retraining complete. Switched back to model-based mode.")
        if rule_model_text:
            print(" Mined Rules:\n", rule_model_text)
    except Exception as e:
        print(" Retraining failed:", e)

for i in range(train_chunks, train_chunks + predict_chunks):
    X_chunk = X.iloc[i*chunk_size:(i+1)*chunk_size]
    y_chunk = y.iloc[i*chunk_size:(i+1)*chunk_size]

    for idx, row in X_chunk.iterrows():
        row_df = pd.DataFrame([row], columns=feature_names)
        row_scaled = pd.DataFrame(scaler.transform(row_df), columns=feature_names)
        true_label = y_chunk.loc[idx]
        start_time = time.time()

        y_pred_prob = model.predict_proba(row_scaled)[0][1]
        y_pred = int(y_pred_prob > 0.5)
        error = int(y_pred != true_label)
        drift = adwin.update(error)

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

        if (drift or feature_drift) and (idx - last_drift_row >= cooldown_period):
            print(f"Drift detected at row {idx} | Switching to rule-based mode.")
            drift_points.append(idx)
            last_drift_row = idx
            in_rule_mode = True
            retraining_complete = False

            # Start retraining asynchronously
            buffer_df = pd.DataFrame(buffer_X, columns=feature_names)
            buffer_y_series = pd.Series(buffer_y)
            thread = threading.Thread(target=retrain_model_async, args=(buffer_df, buffer_y_series))
            thread.start()

        if in_rule_mode:
            if rule_model:
                rule_pred = rule_model.predict(row_df)[0]
            else:
                amount = row["Amount"]
                v14 = row["V14"]
                v17 = row["V17"]
                rule_pred = 1 if (amount > 10000 or v14 < -50 or v17 > 20) else 0
            y_pred_all.append(rule_pred)
            y_true_all.append(true_label)
            y_true_rule.append(true_label)
            y_pred_rule.append(rule_pred)
        else:
            y_pred_all.append(y_pred)
            y_true_all.append(true_label)
            y_true_model.append(true_label)
            y_pred_model.append(y_pred)

        buffer_X.append(row)
        buffer_y.append(true_label)

        latency = time.time() - start_time
        latencies.append(latency)

# Results
print("\n Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4, target_names=["Non-Fraud", "Fraud"]))

print(f" Drift points detected at rows: {drift_points}")
print(f" Overall accuracy: {np.mean(np.array(y_true_all) == np.array(y_pred_all)):.4f}")
print(f"\n Avg Inference Latency: {np.mean(latencies):.6f} seconds")
print(f" Max Inference Latency: {np.max(latencies):.6f} seconds")
print(f" Min Inference Latency: {np.min(latencies):.6f} seconds")

print("\n Rule-based Report:")
if y_pred_rule:
    print(classification_report(y_true_rule, y_pred_rule, digits=4))
else:
    print("No rule-based predictions.")

print("\n Model-based Report:")
if y_pred_model:
    print(classification_report(y_true_model, y_pred_model, digits=4))
else:
    print("No model-based predictions.")