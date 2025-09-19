import pandas as pd
import numpy as np
import time
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from river.drift import ADWIN

# Load and prepare data
df = pd.read_csv("creditcard.csv").sort_values(by="Time").reset_index(drop=True)
X = df.drop("Class", axis=1)
y = df["Class"]

chunk_size = 1000
train_chunks = 30
predict_chunks = 30

X_train = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])
y_train = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])

fraud_count = sum(y_train == 1)
k = min(5, fraud_count - 1) if fraud_count >= 2 else 1
X_res, y_res = SMOTE(k_neighbors=k, random_state=42).fit_resample(X_train, y_train)

scaler = StandardScaler()
X_res_scaled = pd.DataFrame(scaler.fit_transform(X_res), columns=X.columns)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_res_scaled, y_res)

# Drift and Rule Tracking
adwin = ADWIN(delta=0.0005)
rolling_window = []
prev_row_vals = None
top_features = ['V14', 'V3', 'V10', 'V4', 'V12', 'V17']
feature_means = X_res.mean()

# Prediction stats
latencies = []
drift_points = []
y_true_all, y_pred_all = [], []
y_true_rule, y_pred_rule = [], []
y_true_model, y_pred_model = [], []

cooldown_period = 500
last_drift_row = -cooldown_period
retraining_thread = None
in_rule_mode = False
retraining_complete = True
retrain_lock = threading.Lock()

# 🧠 Automatically extract rule thresholds based on recent drift chunk

def mine_rules_from_drift(chunk_X, chunk_y, top_features, percentile=95):
    """
    Extract feature thresholds from a drift-affected chunk.
    Looks at values of top features in fraudulent transactions.
    Returns a dictionary of rules.
    """
    rules = {}
    fraud_rows = chunk_X[chunk_y == 1]

    if len(fraud_rows) == 0:
        return rules

    for feature in top_features:
        values = fraud_rows[feature].values
        if np.all(values > 0):
            threshold = np.percentile(values, percentile)
            rules[feature] = (">", threshold)
        elif np.all(values < 0):
            threshold = np.percentile(values, 100 - percentile)
            rules[feature] = ("<", threshold)
        else:
            # Mixed sign: try upper and lower thresholds
            upper = np.percentile(values, percentile)
            lower = np.percentile(values, 100 - percentile)
            rules[feature] = ("range", (lower, upper))

    return rules


# 🧪 Apply the rules to a row

def apply_mined_rules(row, rules):
    for feature, rule in rules.items():
        if feature not in row:
            continue
        value = row[feature]
        if rule[0] == ">" and value > rule[1]:
            return 1
        elif rule[0] == "<" and value < rule[1]:
            return 1
        elif rule[0] == "range" and (value < rule[1][0] or value > rule[1][1]):
            return 1
    return 0


def retrain_model(recent_X, recent_y):
    global model, scaler, in_rule_mode, retraining_complete
    try:
        fraud_count = sum(recent_y == 1)
        if fraud_count >= 6:
            k_local = min(5, fraud_count - 1)
            X_res, y_res = SMOTE(k_neighbors=k_local, random_state=42).fit_resample(recent_X, recent_y)
        else:
            print(f"⚠️ Not enough fraud cases for SMOTE during retraining (found {fraud_count}). Skipping SMOTE.")
            X_res, y_res = recent_X, recent_y

        new_scaler = StandardScaler()
        # X_scaled = new_scaler.fit_transform(X_res)
        X_scaled = pd.DataFrame(new_scaler.fit_transform(X_res), columns=X_res.columns)


        new_model = LogisticRegression(max_iter=200, solver="lbfgs", class_weight="balanced", random_state=42)
        new_model.fit(X_scaled, y_res)

        with retrain_lock:
            model = new_model
            scaler = new_scaler
            retraining_complete = True
            in_rule_mode = False
        print("✅ Retraining complete. Switched back to model-based.")
    except Exception as e:
        print("❌ Retraining failed:", e)


# Automatic rule generation using Decision Tree
def generate_rules(X_recent, y_recent):
    dt = DecisionTreeClassifier(max_depth=2)
    dt.fit(X_recent[top_features], y_recent)
    rules = []
    for i, t in enumerate(dt.tree_.feature):
        if t != -2:
            threshold = dt.tree_.threshold[i]
            fname = top_features[t]
            rules.append((fname, threshold))
    return rules

# Begin Streaming
print("\n🚀 Starting Stream Prediction...\n")

buffer_X, buffer_y = [], []
rules = [(f, 0) for f in top_features]  # Default fallbacks
mined_rules = {}
rule_mining_window = 100 # number of recent rows to use for rule mining



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
        adwin.update(error)

        # Feature drift
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

        # Handle drift detection
        if (adwin.drift_detected or feature_drift) and (idx - last_drift_row > cooldown_period):
            print(f"\n⚠️ Drift detected at row {idx} | Switching to rule-based mode.")
            drift_points.append(idx)
            last_drift_row = idx
            in_rule_mode = True

            # 🧠 Mine new rules based on recent data  
            recent_start = max(0, idx - rule_mining_window)  
            recent_X = X.iloc[recent_start:idx]  
            recent_y = y.iloc[recent_start:idx]  
            mined_rules = mine_rules_from_drift(recent_X, recent_y, top_features)  
            print(f"🔍 Extracted rules: {mined_rules}")

            retraining_complete = False

            # Collect recent window
            recent_df = pd.DataFrame(buffer_X[-300:])  # last 300
            recent_y = pd.Series(buffer_y[-300:])
            # rules = generate_rules(recent_df, recent_y)
            retraining_thread = threading.Thread(target=retrain_model, args=(recent_df, recent_y))
            retraining_thread.start()

        if in_rule_mode and not retraining_complete:
            # Rule-based prediction
            pred = 0
            # for feat, thresh in rules:
            #     if row[feat] > thresh:
            #         pred = 1
            #         break
            pred = apply_mined_rules(row, mined_rules)
            y_pred_all.append(pred)
            y_true_all.append(true_label)
            y_pred_rule.append(pred)
            y_true_rule.append(true_label)
        else:
            y_pred_all.append(y_pred)
            y_true_all.append(true_label)
            y_pred_model.append(y_pred)
            y_true_model.append(true_label)

        buffer_X.append(row)
        buffer_y.append(true_label)
        latencies.append(time.time() - start_time)

# Reports
print("\n📊 Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4))
print(f"📍 Drift points detected: {drift_points}")
print(f"🔁 Rule-based used: {len(y_pred_rule)}")
print(f"🤖 Model-based used: {len(y_pred_model)}")
print(f"✅ Accuracy: {np.mean(np.array(y_true_all)==np.array(y_pred_all)):.4f}")
print(f"⏱ Avg latency: {np.mean(latencies):.6f} sec")

print("\n🧾 Rule-based Report:")
if y_pred_rule:
    print(classification_report(y_true_rule, y_pred_rule, digits=4))
else:
    print("No rule-based predictions were made.")

print("\n🤖 Model-based Report:")
print(classification_report(y_true_model, y_pred_model, digits=4))
