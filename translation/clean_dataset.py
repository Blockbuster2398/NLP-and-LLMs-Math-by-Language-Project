import re
import pandas as pd
from pathlib import Path
import os

# Quick script for removing the calculator annotations in the dataset

root_path = Path(__file__).resolve().parent.parent
save_directory = root_path / "datasets" / ("gsm8k-" + "en")
os.makedirs(save_directory, exist_ok=True)

# print(re.sub("<<.*?>>","", "<<fewifewi>>fuewijfieowfw<<fjewiofe>>fieowjfe"))

train_df = pd.read_json(root_path / "datasets/gsm8k-en-raw/train.jsonl", lines=True)
test_df = pd.read_json(root_path / "datasets/gsm8k-en-raw/test.jsonl", lines=True)

# Gets rid of the "<< >>'s that confuse the translation model"
train_df = train_df.replace(to_replace='<<.*?>>', value='', regex=True)
test_df = test_df.replace(to_replace='<<.*?>>', value='', regex=True)

train_df.to_json(save_directory / "train.jsonl", orient="records", lines=True)
test_df.to_json(save_directory / "test.jsonl", orient="records", lines=True)
