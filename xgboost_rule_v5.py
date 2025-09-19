import pandas as pd
import numpy as np
import time
import datetime
import threading
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from river.drift import ADWIN
from itertools import combinations
import warnings

warnings.filterwarnings("ignore", message=".*fitted without feature names.*")

print(f"Start time={datetime.datetime.now()}")

# ----------------------
# Load and Prepare Data
# ----------------------
df = pd.read_csv("creditcard.csv")
df = df.sort_values(by="Time").reset_index(drop=True)
X = df.drop("Class", axis=1)
y = df["Class"]

chunk_size = 1000
train_chunks = 50
val_chunks = 2   
predict_chunks = 15

X_train = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])
y_train = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])

X_val = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks, train_chunks + val_chunks)])
y_val = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks, train_chunks + val_chunks)])

# ----------------------
# SMOTEENN Resampling
# ----------------------
fraud_count = sum(y_train == 1)

if fraud_count >= 2:
    k = min(3, max(1, fraud_count-1))
    smote_enn = SMOTEENN(smote=SMOTE(k_neighbors=k, sampling_strategy=1, random_state=42), random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    print(f"✅ SMOTEENN applied | k={k} | Fraud count = {fraud_count} -> {sum(y_res==1)}")
else:
    X_res, y_res = X_train, y_train
    print("⚠️ Not enough frauds for SMOTE")

# ----------------------
# Model Training
# ----------------------
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
scale_weight = (len(y_train) - fraud_count) / fraud_count
model = XGBClassifier(scale_pos_weight=scale_weight, random_state=42)
model.fit(X_res_scaled, y_res)

# ----------------------
# Validation
# ----------------------
if len(X_val) > 0:
    X_val_scaled = scaler.transform(X_val)
    y_val_pred = model.predict(X_val_scaled)
    print("\n📊 Validation Set Evaluation:\n")
    print(classification_report(y_val, y_val_pred, digits=4, target_names=["Non-Fraud", "Fraud"]))

# ----------------------
# Hyperparameter Tuning
# ----------------------
best_f1 = 0
best_params = {}

for max_depth in [3, 5, 7]:
    for learning_rate in [0.05, 0.1, 0.2]:
        model_tune = XGBClassifier(
            scale_pos_weight=scale_weight,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        model_tune.fit(X_res_scaled, y_res)
        y_val_pred = model_tune.predict(X_val_scaled)
        f1 = f1_score(y_val, y_val_pred, pos_label=1)
        print(f"max_depth={max_depth}, lr={learning_rate}, F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_params = {"max_depth": max_depth, "learning_rate": learning_rate}

print(f"\n✅ Best validation F1 for fraud: {best_f1:.4f}")
print(f"Best hyperparameters: {best_params}")

# ----------------------
# Drift Detection & Rules
# ----------------------
adwin = ADWIN(delta=0.0005)
top_features = ['V14', 'V3', 'V10', 'V4', 'V12', 'V17']
rolling_window = []
prev_row_vals = None
cooldown_period = 500
last_drift_row = -cooldown_period

drift_points = []
latencies = []
y_true_all, y_pred_all = [], []
y_true_rule, y_pred_rule = [], []
y_true_model, y_pred_model = [], []

buffer_X, buffer_y = [], []
mined_simple_rules, mined_combo_rules = [], []
retrain_lock = threading.Lock()
in_rule_mode = False
retraining_complete = True

# ----------------------
# Rule Mining
# ----------------------
def mine_simple_rules(X_recent, y_recent, top_features, quantile_threshold=0.99, min_fraud=5):
    rules = []
    fraud_samples = X_recent[y_recent == 1]
    if len(fraud_samples) < min_fraud:
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
    return rules

def mine_combo_rules(X_recent, y_recent, top_features, quantiles=[0.85], min_fraud=5):
    rules = []
    fraud_samples = X_recent[y_recent == 1]
    if len(fraud_samples) < min_fraud:
        return rules
    feature_pairs = list(combinations(top_features, 2))
    for feat1, feat2 in feature_pairs:
        best_rule = None
        best_f1 = 0
        for dir1 in ['greater','less']:
            for dir2 in ['greater','less']:
                for q1 in quantiles:
                    for q2 in quantiles:
                        thresh1 = fraud_samples[feat1].quantile(q1 if dir1=='greater' else 1-q1)
                        thresh2 = fraud_samples[feat2].quantile(q2 if dir2=='greater' else 1-q2)
                        cond1 = X_recent[feat1] > thresh1 if dir1=='greater' else X_recent[feat1] < thresh1
                        cond2 = X_recent[feat2] > thresh2 if dir2=='greater' else X_recent[feat2] < thresh2
                        preds = (cond1 & cond2).astype(int)
                        tp = ((preds==1)&(y_recent==1)).sum()
                        fp = ((preds==1)&(y_recent==0)).sum()
                        fn = ((preds==0)&(y_recent==1)).sum()
                        precision = tp/(tp+fp+1e-8)
                        recall = tp/(tp+fn+1e-8)
                        f1 = 2*precision*recall/(precision+recall+1e-8)
                        if f1>best_f1 and precision>=0.3 and recall>=0.3:
                            best_f1 = f1
                            best_rule = (feat1, thresh1, dir1, feat2, thresh2, dir2)
        if best_rule:
            rules.append(best_rule)
    return rules

# ----------------------
# Vectorized Rule Application
# ----------------------
def apply_all_rules_vectorized(df, simple_rules, combo_rules):
    votes = np.zeros(len(df))
    for feature, thresh, direction in simple_rules:
        if direction=='greater':
            votes += (df[feature].values > thresh)
        else:
            votes += (df[feature].values < thresh)
    for feat1, thresh1, dir1, feat2, thresh2, dir2 in combo_rules:
        cond1 = df[feat1].values > thresh1 if dir1=='greater' else df[feat1].values < thresh1
        cond2 = df[feat2].values > thresh2 if dir2=='greater' else df[feat2].values < thresh2
        votes += (cond1 & cond2)
    return (votes >= 1).astype(int)

# ----------------------
# Retraining
# ----------------------
def retrain_model(X_recent, y_recent):
    global model, scaler, retraining_complete, in_rule_mode, mined_simple_rules, mined_combo_rules
    fraud_count = sum(y_recent==1)
    if fraud_count<2: return
    k_local = min(3,max(1,fraud_count-1))
    smote_enn = SMOTEENN(smote=SMOTE(k_neighbors=k_local, sampling_strategy=1, random_state=42), random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_recent, y_recent)
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X_resampled)
    scale_weight = (len(y_resampled) - fraud_count) / fraud_count
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

# ----------------------
# Stream Prediction
# ----------------------
print("\n🚀 Starting Stream Prediction...\n")
fraud_count_in_buffer = 0

for i in range(train_chunks, train_chunks + predict_chunks):
    X_chunk = X.iloc[i*chunk_size:(i+1)*chunk_size].values
    y_chunk = y.iloc[i*chunk_size:(i+1)*chunk_size].values

    for idx in range(len(X_chunk)):
        row = X_chunk[idx]
        true_label = y_chunk[idx]

        start_time = time.time()
        # Model prediction
        row_scaled = scaler.transform(row.reshape(1,-1))
        y_pred_prob = model.predict_proba(row_scaled)[0][1]
        model_pred = int(y_pred_prob > 0.7)
        error = int(model_pred != true_label)
        drift = adwin.update(error)

        # Feature drift detection
        current_vals = row_scaled[0][[X.columns.get_loc(f) for f in top_features]]
        feature_drift = False
        if prev_row_vals is not None:
            diff = np.abs(current_vals - prev_row_vals)
            if len(rolling_window) >= 30:
                std_dev = np.std(rolling_window, axis=0)
                feature_drift = np.any((diff > 2.5) & (diff > 3*std_dev))
                rolling_window.pop(0)
            rolling_window.append(diff)
        else:
            rolling_window.append(np.zeros_like(current_vals))
        prev_row_vals = current_vals

        if true_label==1: fraud_count_in_buffer+=1

        if (drift or feature_drift) and (idx - last_drift_row >= cooldown_period):
            if fraud_count_in_buffer >= 10:
                print(f"\n⚠️ Drift detected at row {idx}. Retraining triggered.")
                drift_points.append(idx)
                last_drift_row = idx
                in_rule_mode = True
                retraining_complete = False
                buffer_df = pd.DataFrame(buffer_X, columns=X.columns)
                buffer_y_series = pd.Series(buffer_y)
                threading.Thread(target=retrain_model, args=(buffer_df, buffer_y_series)).start()
                fraud_count_in_buffer = 0

        # Apply rules vectorized
        row_df = pd.DataFrame([row], columns=X.columns)
        rule_pred = apply_all_rules_vectorized(row_df, mined_simple_rules, mined_combo_rules)

        # Final prediction
        final_pred = int(model_pred or rule_pred) if in_rule_mode else model_pred

        # Record
        y_pred_all.append(final_pred)
        y_true_all.append(true_label)
        if in_rule_mode:
            y_true_rule.append(true_label)
            y_pred_rule.append(final_pred)
        else:
            y_true_model.append(true_label)
            y_pred_model.append(final_pred)

        buffer_X.append(row)
        buffer_y.append(true_label)
        latencies.append(time.time() - start_time)

# ----------------------
# Reports
# ----------------------
print("\n📊 Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4, target_names=["Non-Fraud","Fraud"]))
print(f"📍 Drift points detected at rows: {drift_points}")
print(f"✅ Overall accuracy: {np.mean(np.array(y_true_all)==np.array(y_pred_all)):.4f}")
print(f"\n⏱ Avg Inference Latency: {np.mean(latencies):.6f} sec")
print(f"⏱ Max Inference Latency: {np.max(latencies):.6f} sec")
print(f"⏱ Min Inference Latency: {np.min(latencies):.6f} sec")

print("\n🧾 Rule-based Report:")
if y_pred_rule: print(classification_report(y_true_rule, y_pred_rule, digits=4, target_names=["Non-Fraud","Fraud"]))
else: print("No rule-based predictions.")

print("\n🤖 Model-based Report:")
if y_pred_model: print(classification_report(y_true_model, y_pred_model, digits=4, target_names=["Non-Fraud","Fraud"]))
else: print("No model-based predictions.")

print(f"End time={datetime.datetime.now()}")
