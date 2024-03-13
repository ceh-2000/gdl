import pandas as pd
import numpy as np

experiment_name = "NeuroGraph_drop_edge=0.0_dataset=HCPActivity_seed=0"

df = pd.read_csv("basic_results.csv", header=None)
df["val_acc"] = pd.to_numeric(df[2].str.split(':').str[1])
df["test_acc"] = pd.to_numeric(df[3].str.split(':').str[1])

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
df.head()