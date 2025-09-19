import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# |                      | Predicted Fraud | Predicted Non-Fraud |
# | -------------------- | --------------- | ------------------- |
# | **Actual Fraud**     | 317 (TP)        | 73 (FN)             |
# | **Actual Non-Fraud** | 214 (FP)        | 244,396 (TN)        |


# Confusion matrix values
cm = np.array([[317, 73],
               [214, 244396]])

# Labels
labels = ["Fraud", "Non-Fraud"]

# Row-normalize (so each row sums to 1)
cm_row = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# Build annotation strings: counts + row percentage
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        perc = cm_row[i, j] * 100
        annot[i, j] = f"{count:,}\n({perc:.2f}%)"

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_row, annot=annot, fmt="", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Proportion per Class'})

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix ( Counts and Percentages)")
plt.show()
