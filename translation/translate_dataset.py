from translation_api import translate_text, print_supported_languages
import os
import pandas as pd
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent


def translate_dataset(target_language, output_samples=100):
    train_samples = int(output_samples * .85)
    test_samples = int(output_samples * .15)
    # save_directory = root_path / "datasets" / ("gsm8k-" + target_language + "-" + str(output_samples)) # Old naming scheme
    save_directory = root_path / "datasets" / ("gsm8k-" + target_language)

    os.makedirs(save_directory, exist_ok=True)

    source_train = pd.read_json(root_path / "datasets/gsm8k-en/train.jsonl", lines=True)
    source_test = pd.read_json(root_path / "datasets/gsm8k-en/test.jsonl", lines=True)

    existing_train = pd.read_json(save_directory / "train.jsonl", lines=True)
    existing_test = pd.read_json(save_directory / "test.jsonl", lines=True)
    
    completed_training_examples = existing_train.shape[0]
    completed_test_examples = existing_test.shape[0]
    total_completed_examples = completed_training_examples + completed_test_examples
    print(f"{total_completed_examples} Existing examples -- Train: {completed_training_examples}, Test: {completed_test_examples}")
    
    columns = ["question", "answer"]
    
    output_test = pd.DataFrame(columns=columns)

    # Create translated training set
    output_questions = []
    output_answers = []
    for i in range(completed_training_examples, train_samples):
        source_row = source_train.iloc[i]
        question = source_row["question"]
        answer = source_row["answer"]
        new_question = translate_text(question, target_language).translated_text
        new_answer = translate_text(answer, target_language).translated_text
        output_questions.append(new_question)
        output_answers.append(new_answer)
        print(f"{i+1}/{train_samples} training set problems translated!", end = "\r", flush=True)
    output_train = pd.DataFrame({'question':output_questions, 'answer':output_answers})
    # Add to existing samples and save
    all_train_examples = pd.concat([existing_train, output_train], ignore_index=False, sort=False)
    all_train_examples.to_json(save_directory / "train.jsonl", orient="records", lines=True)
    
    # Create translated test set
    output_questions = []
    output_answers = []
    for i in range(completed_test_examples, test_samples):
        source_row = source_test.iloc[i]
        question = source_row["question"]
        answer = source_row["answer"]
        new_question = translate_text(question, target_language).translated_text
        new_answer = translate_text(answer, target_language).translated_text
        output_questions.append(new_question)
        output_answers.append(new_answer)
        print(f"{i+1}/{test_samples} test set problems translated!", end = "\r", flush=True)
    output_test = pd.DataFrame({'question':output_questions, 'answer':output_answers})
    # Add to existing samples and save
    all_test_examples = pd.concat([existing_test, output_test], ignore_index=False, sort=False)
    all_test_examples.to_json(save_directory / "test.jsonl", orient="records", lines=True)
    

if __name__ == "__main__":
    # print_supported_languages('en')
    # 100 samples is about 60k (input) characters
    # 500,000 characters/month are provided for free translation
    
    """
    High Resource:
    Chinese - zh
    Spanish - es 
    Latin - la
    Dutch - nl

    Low Resource:
    Esperanto - eo
    Udmert - not supported :(
    Igbo -ig
    Hmong - hmn
    """
    translate_dataset(target_language="zh", output_samples=120)
    translate_dataset(target_language="es", output_samples=100)
    translate_dataset(target_language="la", output_samples=100)
    translate_dataset(target_language="nl", output_samples=100)

    translate_dataset(target_language="eo", output_samples=100)
    translate_dataset(target_language="ig", output_samples=100)
    translate_dataset(target_language="hmn", output_samples=100)

    print(f"Translations Complete!")