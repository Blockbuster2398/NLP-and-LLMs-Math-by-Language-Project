import json
import os
from pathlib import Path


root_path = Path(__file__).resolve().parent.parent
datasets_path = root_path / "datasets"

new_datasets_path = root_path / "answer_only_datasets"

train_eng = datasets_path / "gsm8k-en" / "train.jsonl"
test_eng = datasets_path / "gsm8k-en" / "test.jsonl"
train_truth = [json.loads(line) for line in open(train_eng, "r", encoding='utf-8') if line.strip()]
test_truth = [json.loads(line) for line in open(test_eng, "r", encoding='utf-8') if line.strip()]
for qa in train_truth:
    qa["answer"] = qa["answer"].split("#")[-1].strip()
for qa in test_truth:
    qa["answer"] = qa["answer"].split("#")[-1].strip()    

for dataset in os.listdir(datasets_path):
    train_path = datasets_path / dataset / "train.jsonl"
    test_path = datasets_path / dataset / "test.jsonl"
    train_data = [json.loads(line) for line in open(train_path, "r", encoding='utf-8') if line.strip()]
    test_data = [json.loads(line) for line in open(test_path, "r", encoding='utf-8') if line.strip()]

    for i in range(len(train_data)):
        train_data[i]["answer"] = train_truth[i]["answer"]
    for i in range(len(test_data)):
            train_data[i]["answer"] = train_truth[i]["answer"]
            
    os.makedirs((new_datasets_path / dataset), exist_ok=True)
    with open((new_datasets_path / dataset / "train.jsonl"), "w+") as f:
        for record in train_data:
            f.write(json.dumps(record) + "\n")
    with open((new_datasets_path / dataset / "test.jsonl"), "w+") as f:
            for record in test_data:
                f.write(json.dumps(record) + "\n")
    
    