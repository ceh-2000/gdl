import re
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys

model = "Exphormer"
drop_edge = 0.0
dataset = "HCP-Age"
seed = 0

experiment_name = f"{model}_drop_edge={drop_edge}_dataset={dataset}_seed={seed}"



epochs = []
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Define regular expressions to extract relevant information
epoch_pattern = re.compile(r'Epoch (\d+):')
train_pattern = re.compile(r"train: .* 'loss': (\d+\.\d+),.*'accuracy': (\d+\.\d+)")
val_pattern = re.compile(r'val: .* \'loss\': (\d+\.\d+),.*\'accuracy\': (\d+\.\d+)')
test_pattern = re.compile(r'test: .* \'loss\': (\d+\.\d+),.*\'accuracy\': (\d+\.\d+)')

# Read data from file
with open('exphormer_log_data.csv', 'r') as file:
    for line in file:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)

        train_match = train_pattern.search(line)
        if train_match:
            train_losses.append(float(train_match.group(1)))
            train_accuracies.append(float(train_match.group(2)))

        val_match = val_pattern.search(line)
        if val_match:
            val_losses.append(float(val_match.group(1)))
            val_accuracies.append(float(val_match.group(2)))

        test_match = test_pattern.search(line)
        if test_match:
            test_losses.append(float(test_match.group(1)))
            test_accuracies.append(float(test_match.group(2)))

# Create DataFrame
df = pd.DataFrame({
    'epoch': epochs,
    'train_loss': train_losses,
    'train_acc': train_accuracies,
    'val_loss': val_losses,
    'val_acc': val_accuracies,
    'test_loss': test_losses,
    'test_acc': test_accuracies,
})

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(data=df["train_acc"], dashes=False, label='Train')
sns.lineplot(data=df["val_acc"], dashes=False, label='Validation')
sns.lineplot(data=df["test_acc"], dashes=False, label='Test')

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=18)
plt.legend(bbox_to_anchor=(0.8, 0.5))
# plt.title(f'Accuracy vs. epoch number for {model}')

# Save the plot to a new image with tight layout
plt.tight_layout()
plt.savefig(f'{experiment_name}.png', dpi=300)
