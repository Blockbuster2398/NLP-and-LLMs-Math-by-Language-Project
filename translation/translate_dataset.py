from translation_api import translate_text
import os
import pandas as pd
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent


def translate_dataset(target_language, output_samples=1000):
    train_samples = int(output_samples * .85)
    test_samples = int(output_samples * .15)
    save_directory = root_path / "datasets" / ("gsm8k-" + target_language)
    os.makedirs(save_directory, exist_ok=True)

    source_train = pd.read_json(root_path / "datasets/gsm8k-en/train.jsonl", lines=True)
    source_test = pd.read_json(root_path / "datasets/gsm8k-en/test.jsonl", lines=True)

    columns = ["question", "answer"]
    
    output_test = pd.DataFrame(columns=columns)

    # Create translated training set
    output_questions = []
    output_answers = []
    for i in range(train_samples):
        source_row = source_train.iloc[i]
        question = source_row["question"]
        answer = source_row["answer"]
        new_question = translate_text(question, target_language).translated_text
        new_answer = translate_text(answer, target_language).translated_text
        output_questions.append(new_question)
        output_answers.append(new_answer)
        print(f"{i+1}/{train_samples} training set problems translated!", end = "\r", flush=True)
    output_train = pd.DataFrame({'question':output_questions, 'answer':output_answers})
    # Save
    output_train.to_json(save_directory / "train.jsonl", orient="records", lines=True)
    
    # Create translated test set
    output_questions = []
    output_answers = []
    for i in range(test_samples):
        source_row = source_test.iloc[i]
        question = source_row["question"]
        answer = source_row["answer"]
        new_question = translate_text(question, target_language).translated_text
        new_answer = translate_text(answer, target_language).translated_text
        output_questions.append(new_question)
        output_answers.append(new_answer)
        print(f"{i+1}/{test_samples} test set problems translated!", end = "\r", flush=True)
    output_test = pd.DataFrame({'question':output_questions, 'answer':output_answers})
    # Save
    output_test.to_json(save_directory / "test.jsonl", orient="records", lines=True)
    

translate_dataset(target_language="es", output_samples=10)


