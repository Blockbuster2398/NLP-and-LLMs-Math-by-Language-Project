import os

import pandas as pd

dataset_paths = [
    "answer_only_datasets/gsm8k-en/",
    "answer_only_datasets/gsm8k-eo/",
    "answer_only_datasets/gsm8k-es/",
    "answer_only_datasets/gsm8k-hmn/",
    "answer_only_datasets/gsm8k-ig/",
    "answer_only_datasets/gsm8k-la/",
    "answer_only_datasets/gsm8k-nl/",
    "answer_only_datasets/gsm8k-zh/",
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
    # for df in dfs_train:
    #    if i < len(df):
    #        rows_train.append(df.iloc[i])
    rows_train.append(dfs_train[i%8].iloc[i])

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

os.makedirs("answer_only_datasets/gsm8k-mix", exist_ok=True)
final_train.to_json("answer_only_datasets/gsm8k-mix/train.jsonl", orient="records", lines=True)
final_test.to_json("answer_only_datasets/gsm8k-mix/test.jsonl", orient="records", lines=True)