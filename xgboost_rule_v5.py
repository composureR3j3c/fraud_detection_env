import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --------------------
# Data prep
# --------------------
# Example: Replace with your own dataset load
df = pd.read_csv("transactions.csv")  # adjust file
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------
# Model Training
# --------------------
fraud_count = np.sum(y_train == 1)

# ⚡ lighter scale_pos_weight (no threshold tuning needed)
scale_pos_weight = np.sqrt((len(y_train) - fraud_count) / fraud_count)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",     # ⚡ faster
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------
# Evaluation
# --------------------
y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)  # ⚡ fixed 0.5 threshold
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --------------------
# Retraining function (no SMOTE)
# --------------------
def retrain_model(X_recent, y_recent, old_model=None):
    fraud_count = np.sum(y_recent == 1)
    scale_pos_weight = np.sqrt((len(y_recent) - fraud_count) / fraud_count)

    # ⚡ no SMOTE (direct use of recent data)
    X_resampled, y_resampled = X_recent, y_recent

    new_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1
    )

    new_model.fit(X_resampled, y_resampled)
    return new_model
