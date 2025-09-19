import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix counts
cm = np.array([[303, 95],
               [41, 254368]])
labels = ["Fraud", "Non-Fraud"]

# Row-normalized (recall per class)
cm_row = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# Column-normalized (precision per predicted class)
cm_col = cm.astype(float) / cm.sum(axis=0)[np.newaxis, :]

# Build annotation strings with counts, row%, column%
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        row_perc = cm_row[i, j] * 100
        col_perc = cm_col[i, j] * 100
        annot[i, j] =        f"{count:,}"
        # \n(R:{row_perc:.1f}%, P:{col_perc:.1f}%)"


# Plot heatmap (row-normalized for color intensity so fraud cells are visible)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_row, annot=annot, fmt="", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Recall per Class'})

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix with Counts, Recall & Precision")
plt.show()
