import pandas as pd
import numpy as np
import time
import datetime
import threading
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from river.drift import ADWIN
from itertools import combinations
import warnings

warnings.filterwarnings("ignore", message=".*fitted without feature names.*")

print(f"Start time={datetime.datetime.now()}")

# 🚀 Load and Prepare Data
df = pd.read_csv("creditcard.csv")
df = df.sort_values(by="Time").reset_index(drop=True)
X = df.drop("Class", axis=1)
y = df["Class"]

chunk_size = 1000
train_chunks = 30
predict_chunks = 30
X_train = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])
y_train = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])

# 🚀 SMOTE Resampling
fraud_count = sum(y_train == 1)
if fraud_count >= 2:
    k = min(5, fraud_count - 1)
    X_res, y_res = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_train, y_train)
    print(f"✅ SMOTE applied (k={k}) | Fraud count = {fraud_count}")
else:
    X_res, y_res = X_train, y_train
    print("⚠️ Not enough frauds for SMOTE")

# 🚀 Model and Scaler Training
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
model = XGBClassifier(scale_pos_weight=fraud_count / (len(y_train) - fraud_count), random_state=42)
model.fit(X_res_scaled, y_res)

# Drift Detector
adwin = ADWIN(delta=0.0005)

# Drift and Feature Monitoring Setup
top_features = ['V14', 'V3', 'V10', 'V4', 'V12', 'V17']
feature_means = X_res[top_features].mean()
rolling_window = []
prev_row_vals = None
cooldown_period = 500
last_drift_row = -cooldown_period

# Tracking Variables
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
mined_simple_rules = []
mined_combo_rules = []
retrain_lock = threading.Lock()
in_rule_mode = False
retraining_complete = True

# 🚀 Rule Mining Functions
def mine_simple_rules(X_recent, y_recent, top_features, quantile_threshold=0.99, min_fraud=5):
    rules = []
    fraud_samples = X_recent[y_recent == 1]
    if len(fraud_samples) < min_fraud:
        print("⚠️ Not enough fraud samples for simple rule mining.")
        return rules

    for feature in top_features:
        fraud_mean = fraud_samples[feature].mean()
        overall_mean = X_recent[feature].mean()

        if fraud_mean > overall_mean:
            threshold = X_recent[feature].quantile(quantile_threshold)
            rules.append((feature, threshold, 'greater'))
        else:
            threshold = X_recent[feature].quantile(1 - quantile_threshold)
            rules.append((feature, threshold, 'less'))

    print(f"🧠 {len(rules)} simple rules mined.")
    return rules

def mine_combo_rules(X_recent, y_recent, top_features, quantiles=[0.85], min_fraud=5):
    rules = []
    fraud_samples = X_recent[y_recent == 1]
    if len(fraud_samples) < min_fraud:
        print("⚠️ Not enough fraud samples for combo rule mining.")
        return rules

    feature_pairs = list(combinations(top_features, 2))
    for feat1, feat2 in feature_pairs:
        best_rule = None
        best_f1 = 0

        for dir1 in ['greater', 'less']:
            for dir2 in ['greater', 'less']:
                for q1 in quantiles:
                    for q2 in quantiles:
                        if dir1 == 'greater':
                            thresh1 = fraud_samples[feat1].quantile(q1)
                            cond1 = X_recent[feat1] > thresh1
                        else:
                            thresh1 = fraud_samples[feat1].quantile(1 - q1)
                            cond1 = X_recent[feat1] < thresh1

                        if dir2 == 'greater':
                            thresh2 = fraud_samples[feat2].quantile(q2)
                            cond2 = X_recent[feat2] > thresh2
                        else:
                            thresh2 = fraud_samples[feat2].quantile(1 - q2)
                            cond2 = X_recent[feat2] < thresh2

                        preds = (cond1 & cond2).astype(int)
                        tp = ((preds == 1) & (y_recent == 1)).sum()
                        fp = ((preds == 1) & (y_recent == 0)).sum()
                        fn = ((preds == 0) & (y_recent == 1)).sum()

                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)

                        if f1 > best_f1 and precision >= 0.3 and recall >= 0.3:
                            best_f1 = f1
                            best_rule = (feat1, thresh1, dir1, feat2, thresh2, dir2)

        if best_rule:
            rules.append(best_rule)

    print(f"🧠 {len(rules)} combo rules mined.")
    return rules

# 🚀 Rule Application Functions
def apply_simple_rules(row, rules, min_votes=1):
    votes = 0
    for feature, threshold, direction in rules:
        if direction == 'greater' and row[feature] > threshold:
            votes += 1
        if direction == 'less' and row[feature] < threshold:
            votes += 1
    return votes

def apply_combo_rules(row, rules, min_votes=1):
    votes = 0
    for feat1, thresh1, dir1, feat2, thresh2, dir2 in rules:
        cond1 = row[feat1] > thresh1 if dir1 == 'greater' else row[feat1] < thresh1
        cond2 = row[feat2] > thresh2 if dir2 == 'greater' else row[feat2] < thresh2
        if cond1 and cond2:
            votes += 1
    return votes

def apply_all_rules(row, simple_rules, combo_rules, min_votes=1):
    simple_votes = apply_simple_rules(row, simple_rules)
    combo_votes = apply_combo_rules(row, combo_rules)
    total_votes = simple_votes + combo_votes
    return int(total_votes >= min_votes)

# 🚀 Model Retraining Function
def retrain_model(X_recent, y_recent):
    global model, scaler, retraining_complete, in_rule_mode, mined_simple_rules, mined_combo_rules
    try:
        fraud_count = sum(y_recent == 1)
        if fraud_count < 2:
            print(f"⚠️ Not enough fraud cases to retrain (fraud_count={fraud_count}). Skipping.")
            return
        k_local = min(5, fraud_count - 1)
        print(f"✅ SMOTE during retraining (k={k_local}) | Fraud count = {fraud_count}")
        X_resampled, y_resampled = SMOTE(k_neighbors=k_local, random_state=42).fit_resample(X_recent, y_recent)

        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X_resampled)
        scale_weight = fraud_count / (len(y_resampled) - fraud_count)
        new_model = XGBClassifier(scale_pos_weight=scale_weight, random_state=42)
        new_model.fit(X_scaled, y_resampled)

        with retrain_lock:
            model = new_model
            scaler = new_scaler
            mined_simple_rules = mine_simple_rules(pd.DataFrame(X_resampled, columns=X.columns), y_resampled, top_features)
            mined_combo_rules = mine_combo_rules(pd.DataFrame(X_resampled, columns=X.columns), y_resampled, top_features)
            retraining_complete = True
            in_rule_mode = False

        print("✅ Retraining complete. Switched back to model-based mode.")
        if mined_simple_rules or mined_combo_rules:
            print("🧠 New rules mined.")

    except Exception as e:
        print("❌ Retraining failed:", e)

# 🚀 Stream Prediction
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
            print(f"\n⚠️ Drift detected at row {idx} | Switching to rule-based mode.")
            drift_points.append(idx)
            last_drift_row = idx
            in_rule_mode = True
            retraining_complete = False
            buffer_df = pd.DataFrame(buffer_X, columns=X.columns)
            buffer_y_series = pd.Series(buffer_y)
            threading.Thread(target=retrain_model, args=(buffer_df, buffer_y_series)).start()

        if in_rule_mode:
            rule_pred = apply_all_rules(row, mined_simple_rules, mined_combo_rules)
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
        latencies.append(time.time() - start_time)

# 🚀 Final Reports
print("\n📊 Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4, target_names=["Non-Fraud", "Fraud"]))
print(f"📍 Drift points detected at rows: {drift_points}")
print(f"✅ Overall accuracy: {np.mean(np.array(y_true_all) == np.array(y_pred_all)):.4f}")
print(f"\n⏱ Avg Inference Latency: {np.mean(latencies):.6f} seconds")
print(f"⏱ Max Inference Latency: {np.max(latencies):.6f} seconds")
print(f"⏱ Min Inference Latency: {np.min(latencies):.6f} seconds")

print("\n🧾 Rule-based Report:")
if y_pred_rule:
    print(classification_report(y_true_rule, y_pred_rule, digits=4, target_names=["Non-Fraud", "Fraud"]))
else:
    print("No rule-based predictions.")

print("\n🤖 Model-based Report:")
if y_pred_model:
    print(classification_report(y_true_model, y_pred_model, digits=4, target_names=["Non-Fraud", "Fraud"]))
else:
    print("No model-based predictions.")

print(f"End time={datetime.datetime.now()}")
