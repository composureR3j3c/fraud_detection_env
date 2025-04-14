import pandas as pd
from river.drift import ADWIN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load and sort data
df = pd.read_csv("creditcard.csv")
df = df.sort_values(by="Time").reset_index(drop=True)

X = df.drop("Class", axis=1)
y = df["Class"]

chunk_size = 1000
n_chunks = len(X) // chunk_size

train_chunks = 20
predict_chunks = 10

# Prepare training data
X_train = pd.concat([X.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])
y_train = pd.concat([y.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(train_chunks)])

# Resample
fraud_count = sum(y_train == 1)
if fraud_count >= 2:
    k = min(5, fraud_count - 1)
    sm = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"📍 Applied SMOTE with k={k} | Fraud Samples: {fraud_count}")
else:
    X_res, y_res = X_train, y_train
    print(f"⚠️ Not enough fraud cases for SMOTE. Proceeding without resampling.")

# Train model
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_res, y_res)

# Flip all labels in chunks 20–49
flipped_indices = []

for i in range(20, 30):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    if i % 2 == 0:
        df.loc[start:end-1, "Class"] = df.loc[start:end-1, "Class"].apply(lambda x: 0 if x == 1 else 1)
    flipped_indices.extend(range(start, end))

print("✅ Flipped ALL labels in chunks 20–49")

# Initialize ADWIN
adwin = ADWIN(delta=0.001)

# Evaluation
y_true_all = []
y_pred_all = []
drift_points = []

print("\n🔍 Predictions on flipped data with true vs predicted:\n")
for i in range(train_chunks, train_chunks + predict_chunks):
    X_chunk = X.iloc[i*chunk_size:(i+1)*chunk_size]
    y_chunk = y.iloc[i*chunk_size:(i+1)*chunk_size]
    
    for idx, row in X_chunk.iterrows():
        row_df = pd.DataFrame([row], columns=X.columns)
        y_pred_prob = rf.predict_proba(row_df)[0][1]
        pred = int(y_pred_prob > 0.5)

        true_label = y_chunk.loc[idx]
        error = int(pred != true_label)
        
        drift = adwin.update(error)

        # Collect true and pred
        y_true_all.append(true_label)
        y_pred_all.append(pred)

        # Print some sample outputs
        if idx % 500 == 0:  # Adjust this to print fewer/more
            print(f"Row {idx}: True = {true_label}, Pred = {pred}, Prob = {y_pred_prob:.4f}")
            print(f"Row {i * chunk_size + idx} | Error: {error} | Window size: {adwin.width} | Estimation: {adwin.estimation:.4f} | Drift: {adwin.drift_detected:.4f}")
        if adwin.drift_detected:
            drift_points.append(idx)
            print(f"⚠️ Drift detected at row {idx} | Estimation Error: {adwin.estimation:.4f}")

# Final Evaluation
print("\n📊 Final Evaluation:\n")
print(classification_report(y_true_all, y_pred_all, digits=4))
print(f"📍 Drift points detected at rows: {drift_points}")
