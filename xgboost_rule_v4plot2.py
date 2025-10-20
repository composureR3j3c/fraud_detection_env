import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# Load predictions from file
y_true_all, y_pred_all = [], []
with open("predictions.txt", "r") as f:
    for line in f:
        yt, yp = line.strip().split(',')
        y_true_all.append(int(yt))
        y_pred_all.append(int(yp))

# Parameters
window = 1000  # number of transactions per evaluation window
dot_interval = 1000  

# Metrics storage
precision_avg, recall_avg, x_axis = [], [], []

for i in range(window, len(y_true_all) + 1, window):
    y_true_chunk = y_true_all[i-window:i]
    y_pred_chunk = y_pred_all[i-window:i]

    # Per-class metrics
    prec_nf = precision_score(y_true_chunk, y_pred_chunk, pos_label=0, zero_division=0)
    prec_fr = precision_score(y_true_chunk, y_pred_chunk, pos_label=1, zero_division=0)

    rec_nf = recall_score(y_true_chunk, y_pred_chunk, pos_label=0, zero_division=0)
    rec_fr = recall_score(y_true_chunk, y_pred_chunk, pos_label=1, zero_division=0)

    # Macro averages
    precision_avg.append((prec_nf + prec_fr) / 2)
    recall_avg.append((rec_nf + rec_fr) / 2)

    x_axis.append(i)

# Plot
plt.figure(figsize=(8, 6))

plt.plot(x_axis, precision_avg, marker="o", linestyle="-", label="Precision (Macro Avg)", color="blue")
plt.plot(x_axis, recall_avg, marker="s", linestyle="--", label="Recall (Macro Avg)", color="green")

# y-axis from 0 to 1
plt.ylim(0, 1)  
plt.xlabel("Transactions Processed")
plt.ylabel("Score")
plt.title("Macro-Averaged Precision and Recall Over Time")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
