import time
import pandas as pd
from river.drift import ADWIN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize ADWIN with a more sensitive delta
adwin = ADWIN(delta=0.00001)

# Base sample dataset
data = [
    {"amount": 50, "merchant_id": 1, "time_of_day": 10, "fraud": 0},
    {"amount": 60, "merchant_id": 2, "time_of_day": 12, "fraud": 0},
    {"amount": 15000000, "merchant_id": 3, "time_of_day": 18, "fraud": 1},
    {"amount": 40, "merchant_id": 1, "time_of_day": 9, "fraud": 0},
    {"amount": 30000000, "merchant_id": 4, "time_of_day": 22, "fraud": 1},
]

# Repeat data to simulate 1000+ transactions
streaming_transactions = data * 200  # This will give us 1000 transactions (5*200 = 1000)

# DataFrame for training
df = pd.DataFrame(data)
X = df.drop(columns=["fraud"])
y = df["fraud"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Process transactions
for idx, transaction in enumerate(streaming_transactions):
    # Make prediction
    X_transaction = pd.DataFrame([transaction]).drop(columns=["fraud"])
    true_label = transaction["fraud"]
    prediction = model.predict(X_transaction)[0]

    # Log prediction error
    error = 1 if prediction != true_label else 0
    # print(f"Prediction error for transaction {transaction}: {error}")

    # Update ADWIN with error (1 if incorrect, 0 if correct)
    adwin.update(error)

    # Track ADWIN's internal state and print change detection status
    # print(f"ADWIN: Width={adwin.width} | Estimation={adwin.estimation} | Change Detected={adwin.drift_detected}")

    # Simulate concept drift at certain points
    if idx == 200 or idx == 400 or idx == 600:  # Simulate concept drift at these points
        print(f"Simulating larger concept drift at transaction {transaction}")
        # Flip label and modify features to simulate drift
        transaction["fraud"] = 1 - true_label  # Flip label
        transaction["amount"] *= 100  # Drastically alter the amount
        transaction["time_of_day"] = (transaction["time_of_day"] + 12) % 24  # Time shift

    # If concept drift is detected, retrain the model
    if adwin.drift_detected:
        print(f"🚨 Concept Drift Detected! Retraining model...")
        
        # Simulate retraining by adding the new transaction to training data
        X_train = pd.concat([X_train, X_transaction], ignore_index=True)
        y_train = pd.concat([y_train, pd.Series([transaction["fraud"]])], ignore_index=True)
        model.fit(X_train, y_train)
        print("🔄 Model retrained successfully!")

    # else:
        # print(f"✅ Transaction {transaction} processed normally. Prediction: {prediction}")

    # Simulate a delay to mimic real-time processing
    time.sleep(0.01)  # Adjust as needed for realistic stream processing
