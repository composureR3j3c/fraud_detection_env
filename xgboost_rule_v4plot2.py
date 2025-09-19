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
window = 10000 # number of transactions per evaluation window
dot_interval = 1000  
# drift_points = [39375, 42549, 43060, 44270, 54019, 63636, 76609, 82243,
#                 90950, 98431, 104125, 115204, 123316, 140786, 143338,
#                 149155, 150646, 151146, 153835, 154676, 158501, 184020,
#                 192528, 203329, 214813, 222357, 230076, 236429, 242460,
#                 248501, 252125, 262560, 276025]

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
plt.figure(figsize=(4, 7))

plt.plot(x_axis, precision_avg, marker="o", label="Precision (Macro Avg)", color="blue")
# plt.plot(x_axis, recall_avg, marker="s", label="Recall (Macro Avg)", color="green")

# Drift markers
# for dp in drift_points:
#     plt.axvline(x=dp, color="gray", linestyle=":", alpha=0.3)
plt.ylim(0, 1)  
plt.xlabel("Transactions Processed")
plt.ylabel("Score")
plt.title("Macro-Averaged Precision & Recall Over Time with Drift Points")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
