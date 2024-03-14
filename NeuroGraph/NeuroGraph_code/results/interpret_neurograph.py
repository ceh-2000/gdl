import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

model = "ResidualGCN"
drop_edge = 0.0
dataset = "HCP-Age"
seed = 0

experiment_name = f"{model}_drop_edge={drop_edge}_dataset={dataset}_seed={seed}"


df = pd.read_csv("neurograph_data.csv", header=None)
df["train_acc"] = pd.to_numeric(df[2].str.split(':').str[1])
df["val_acc"] = pd.to_numeric(df[3].str.split(':').str[1])
df["test_acc"] = pd.to_numeric(df[4].str.split(':').str[1])

best_val_idx = np.argmax(df["val_acc"])
best_val = round(np.max(df["val_acc"])*100, 2)

best_test_idx = np.argmax(df["test_acc"])
best_test = round(np.max(df["test_acc"])*100, 2)
final_test = round(df["test_acc"].iloc[-1]*100, 2)

print(f"Best validation accuracy epoch: {best_val_idx}")
print(f"Best validation accuracy: {best_val}")

print(f"Best test accuracy epoch: {best_test_idx}")
print(f"Best test accuracy: {best_test}")

print(f"Final test accuracy: {final_test}")

df = df.drop(columns=[0, 1, 2, 3])
df = (df * 100).round(2)

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(data=df["train_acc"], dashes=False, label='Train')
sns.lineplot(data=df["val_acc"], dashes=False, label='Validation')
sns.lineplot(data=df["test_acc"], dashes=False, label='Test')

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=18)
# plt.title(f'Accuracy vs. epoch number for {model}')

# Save the plot to a new image with tight layout
plt.tight_layout()
plt.savefig(f'{experiment_name}.png', dpi=300)