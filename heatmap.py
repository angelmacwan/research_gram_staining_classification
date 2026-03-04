import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define the confusion matrix data
data = [
    [27, 0, 2, 1],
    [2, 23, 0, 5],
    [0, 0, 29, 1],
    [0, 0, 2, 28]
]

# Define labels
labels_true = ['True GNB', 'True GNC', 'True GPB', 'True GPC']
labels_pred = ['Pred GNB', 'Pred GNC', 'Pred GPB', 'Pred GPC']

# Create DataFrame
df_cm = pd.DataFrame(data, index=labels_true, columns=labels_pred)

# Plot the heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False,
            linewidths=0.5, linecolor='black', square=True)

plt.title("Confusion Matrix Heatmap", fontsize=14)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

