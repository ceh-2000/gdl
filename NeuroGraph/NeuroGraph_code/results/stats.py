import pandas as pd

csv_name = "basic_results"

df = pd.read_csv(f"{csv_name}.csv")

print(df)

df = df.drop(columns=["best_val_epoch", "best_val_acc", "test_acc_best_val"])

print(df.head())

if "drop_edge_prob" in df.columns:
    result_df = df.groupby(["dataset_name", "drop_edge_prob", "model_name"])['test_acc'].agg(['mean', 'std']).reset_index()
else:
    result_df = df.groupby(["dataset_name", "model_name"])['test_acc'].agg(['mean', 'std']).reset_index()

result_df = result_df.round(2)

result_df.to_csv(f"{csv_name}_agg.csv", index=False)