from river import drift
import random


# Initialize ADWIN
adwin = drift.ADWIN(delta=0.000000000000001)

# Simulated fraud detection model (simple rule-based for demonstration)
class FraudModel:
    def predict(self, transaction):
        """Basic rule: Transactions above $1000 are flagged as fraud (1), otherwise normal (0)."""
        return 1 if transaction["amount"] > 1000 else 0

# Instantiate fraud model
fraud_model = FraudModel()

# Normal transaction patterns (no fraud)
transactions = [
    {"amount": random.randint(10, 100), "merchant_id": random.randint(1, 5), "time_of_day": random.randint(8, 20)}
    for _ in range(20)
]

# Introduce concept drift with fraudulent transactions
fraudulent_transactions = [
    {"amount": random.randint(2000, 6000), "merchant_id": random.randint(6, 10), "time_of_day": random.randint(0, 5)}
    for _ in range(5)
]

# Combine normal and fraudulent transactions
transactions.extend(fraudulent_transactions)

# Process transactions and check for drift
for t in transactions:
    prediction = fraud_model.predict(t)  # Get fraud prediction
    adwin.update(prediction)  # Update ADWIN with new data

    print(f"Transaction: {t} | Estimation: {adwin.estimation:.2f} | Change Detected: {adwin.drift_detected}")

    if adwin.drift_detected:
        print("🚨 Concept drift detected! Retraining model...")

