from translation_api import translate_text, print_supported_languages
import os
import re
import html
import shutil
import pandas as pd
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent

COLUMNS = ["question", "answer"]

# Matches any mangled version of GSM8K's final-answer marker at the end of an answer:
# "####", "#####", "# # # #", with an optional number after it (digits may be grouped with spaces, e.g. "70 000").
MARKER_RE = re.compile(r"\s*#\s*#\s*#\s*#[#\s]*\$?\s*-?[\d.,\s]*$")


def clean_text(text):
    # Google Translate returns HTML entities (&#39; etc.) unless mime_type="text/plain" is set
    return html.unescape(str(text)).strip()


def fix_answer(translated_answer, english_answer):
    """Strip whatever translation did to the '#### N' marker and re-append it,
    taking N from the English source answer at the same row."""
    final = english_answer.split("####")[-1].strip().replace(",", "")
    body = MARKER_RE.sub("", clean_text(translated_answer)).rstrip()
    return f"{body}\n#### {final}"


def load_existing(path):
    # First run for a language: no file (or an empty one) yet
    try:
        df = pd.read_json(path, lines=True)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame(columns=COLUMNS)
    return df[COLUMNS] if len(df) else pd.DataFrame(columns=COLUMNS)


def repair_existing(existing, source, path):
    """Repair rows translated by earlier runs (row i lines up with source row i)."""
    if existing.empty:
        return existing
    fixed = existing.copy()
    fixed["question"] = fixed["question"].map(clean_text)
    fixed["answer"] = [fix_answer(a, source.iloc[i]["answer"]) for i, a in enumerate(fixed["answer"])]
    changed = int(((existing["question"] != fixed["question"]) | (existing["answer"] != fixed["answer"])).sum())
    if changed:
        backup = path.with_suffix(".jsonl.bak")
        if not backup.exists():
            shutil.copy(path, backup)
        print(f"Repaired {changed} previously translated rows in {path.parent.name}/{path.name}")
    return fixed


def translate_dataset(target_language, output_samples=100):
    train_samples = int(output_samples * .85)
    test_samples = int(output_samples * .15)
    # save_directory = root_path / "datasets" / ("gsm8k-" + target_language + "-" + str(output_samples)) # Old naming scheme
    save_directory = root_path / "datasets" / ("gsm8k-" + target_language)

    os.makedirs(save_directory, exist_ok=True)

    source_train = pd.read_json(root_path / "datasets/gsm8k-en/train.jsonl", lines=True)
    source_test = pd.read_json(root_path / "datasets/gsm8k-en/test.jsonl", lines=True)

    existing_train = repair_existing(load_existing(save_directory / "train.jsonl"), source_train, save_directory / "train.jsonl")
    existing_test = repair_existing(load_existing(save_directory / "test.jsonl"), source_test, save_directory / "test.jsonl")

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
        new_question = clean_text(translate_text(question, target_language).translated_text)
        new_answer = fix_answer(translate_text(answer, target_language).translated_text, answer)
        output_questions.append(new_question)
        output_answers.append(new_answer)
        print(f"{i+1}/{train_samples} training set problems translated!", end = "\r", flush=True)
    output_train = pd.DataFrame({'question':output_questions, 'answer':output_answers})
    # Add to existing samples and save
    all_train_examples = pd.concat([existing_train, output_train], ignore_index=True, sort=False)
    all_train_examples.to_json(save_directory / "train.jsonl", orient="records", lines=True)

    # Create translated test set
    output_questions = []
    output_answers = []
    for i in range(completed_test_examples, test_samples):
        source_row = source_test.iloc[i]
        question = source_row["question"]
        answer = source_row["answer"]
        new_question = clean_text(translate_text(question, target_language).translated_text)
        new_answer = fix_answer(translate_text(answer, target_language).translated_text, answer)
        output_questions.append(new_question)
        output_answers.append(new_answer)
        print(f"{i+1}/{test_samples} test set problems translated!", end = "\r", flush=True)
    output_test = pd.DataFrame({'question':output_questions, 'answer':output_answers})
    # Add to existing samples and save
    all_test_examples = pd.concat([existing_test, output_test], ignore_index=True, sort=False)
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
    translate_dataset(target_language="zh", output_samples=300)
    translate_dataset(target_language="es", output_samples=300)
    translate_dataset(target_language="la", output_samples=300)
    translate_dataset(target_language="nl", output_samples=300)

    translate_dataset(target_language="eo", output_samples=300)
    translate_dataset(target_language="ig", output_samples=300)
    translate_dataset(target_language="hmn", output_samples=300)

    print(f"Translations Complete!")
