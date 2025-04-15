import pandas as pd
from river.drift import ADWIN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import deque

# Load and sort data
df = pd.read_csv("creditcard.csv")
df = df.sort_values(by="Time").reset_index(drop=True)

X = df.drop("Class", axis=1)
y = df["Class"]

chunk_size = 1000
train_chunks = 20
predict_chunks = 10
history_buffer = deque(maxlen=2000)

# Train model on first 20 chunks
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

# Initial model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_res, y_res)

# Simulate concept drift by flipping labels in chunks 20–29
for i in range(20, 30):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    if i % 2 == 0:
        df.loc[start:end-1, "Class"] = df.loc[start:end-1, "Class"].apply(lambda x: 0 if x == 1 else 1)
print("🔀 Simulated concept drift: Flipped labels in chunks 20–29")

# Recreate X and y after flipping
X = df.drop("Class", axis=1)
y = df["Class"]

# ADWIN setup
adwin = ADWIN(delta=0.001)
use_rule = False
retraining = False
scaler = StandardScaler()
drift_points = []
y_true_all = []
y_pred_all = []
mode_log = []

# Rule-based fallback
def rule_predictor(row):
    return int(row['Amount'] > 1500 or row['V14'] < -5)

# Streaming evaluation
print("\n🚀 Starting Stream Prediction...\n")

for i in range(train_chunks, train_chunks + predict_chunks):
    X_chunk = X.iloc[i*chunk_size:(i+1)*chunk_size]
    y_chunk = y.iloc[i*chunk_size:(i+1)*chunk_size]
    
    for idx, row in X_chunk.iterrows():
        row_df = pd.DataFrame([row], columns=X.columns)
        true_label = y_chunk.loc[idx]
        history_buffer.append((row.values, true_label))

        # Predict
        if use_rule:
            pred = rule_predictor(row)
        else:
            y_pred_prob = model.predict_proba(row_df)[0][1]
            pred = int(y_pred_prob > 0.5)

        # Track predictions
        y_true_all.append(true_label)
        y_pred_all.append(pred)
        mode_log.append("Rule" if use_rule else "Model")

        # Update ADWIN
        error = int(pred != true_label)
        if adwin.update(error) and not retraining:
            print(f"\n⚠️ Drift detected at row {i * chunk_size + idx} | Switching to rule-based mode.")
            use_rule = True
            retraining = True
            drift_points.append(i * chunk_size + idx)

            # Prepare history buffer
            X_hist = pd.DataFrame([x for x, _ in history_buffer], columns=X.columns)
            y_hist = pd.Series([y for _, y in history_buffer])

            # Scale and retrain
            try:
                sm = SMOTE(k_neighbors=min(5, sum(y_hist==1)-1), random_state=42)
                X_res, y_res = sm.fit_resample(X_hist, y_hist)
            except:
                X_res, y_res = X_hist, y_hist

            X_scaled = scaler.fit_transform(X_res)
            new_model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
            new_model.fit(X_scaled, y_res)
            model = new_model
            print("✅ Retraining complete. Switching back to model-based prediction.\n")
            use_rule = False
            retraining = False

# Final Evaluation
print("\n📊 Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4))
print(f"📍 Drift points detected at rows: {drift_points}")
print(f"\n🔁 Rule-based mode used: {mode_log.count('Rule')} times")
print(f"🧠 Model-based mode used: {mode_log.count('Model')} times")
print(f"✅ Overall accuracy: {accuracy_score(y_true_all, y_pred_all):.4f}")
