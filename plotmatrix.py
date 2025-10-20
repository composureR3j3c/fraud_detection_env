import matplotlib.pyplot as plt

# Example: Replace these with your actual logged values
epochs = list(range(1, 21))  # 20 epochs
train_loss = [0.68, 0.55, 0.44, 0.37, 0.30, 0.26, 0.23, 0.21, 0.20, 0.19,
              0.18, 0.17, 0.16, 0.15, 0.15, 0.14, 0.14, 0.13, 0.13, 0.12]
val_loss   = [0.70, 0.58, 0.48, 0.39, 0.33, 0.29, 0.26, 0.24, 0.22, 0.22,
              0.21, 0.21, 0.20, 0.19, 0.19, 0.18, 0.18, 0.18, 0.17, 0.17]

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='s', label='Validation Loss')

<<<<<<< HEAD
# Confusion matrix values
# cm = np.array([[47683, 12],
            #    [5, 57]])
cm = np.array([[57, 5],
               [12, 47683]])
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
        annot[i, j] = f"{count:,}"

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_row, annot=annot, fmt="", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Proportion per Class'})

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
=======
plt.title("Loss Function over Epochs", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.ylim(0, max(max(train_loss), max(val_loss)) + 0.1)
plt.legend()
plt.grid(True)
>>>>>>> 60c99818763c8e33c1e177e3b009933790768a79
plt.show()
