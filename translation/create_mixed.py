import os

import pandas as pd

dataset_paths = [
    "datasets/gsm8k-en/",
    "datasets/gsm8k-eo/",
    "datasets/gsm8k-es/",
    "datasets/gsm8k-hmn/",
    "datasets/gsm8k-ig/",
    "datasets/gsm8k-la/",
    "datasets/gsm8k-nl/",
    "datasets/gsm8k-zh/",
]

dfs_train = []
dfs_test = []

for path in dataset_paths:
    train_df = pd.read_json(path + "train.jsonl", lines=True)
    test_df = pd.read_json(path + "test.jsonl", lines=True)
    dfs_train.append(train_df)
    dfs_test.append(test_df)

rows_train = []
for i in range(len(dfs_train[0])):
    for df in dfs_train:
        if i < len(df):
            rows_train.append(df.iloc[i])

final_train = pd.DataFrame(rows_train)

rows_test = []
for i in range(len(dfs_test[0])):
    for df in dfs_test:
        if i < len(df):
            rows_test.append(df.iloc[i])

final_test = pd.DataFrame(rows_test)

print("Final Train:")
print(final_train)
print("\nFinal Test:")
print(final_test)

os.makedirs("datasets/gsm8k-mix_full", exist_ok=True)
final_train.to_json("datasets/gsm8k-mix_full/train.jsonl", orient="records", lines=True)
final_test.to_json("datasets/gsm8k-mix_full/test.jsonl", orient="records", lines=True)