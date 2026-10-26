import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
import pandas as pd

api_key = os.environ.get("TRINITY_API_KEY")
if not api_key:
    raise RuntimeError(
        "Set the TRINITY_API_KEY environment variable before running the benchmark."
    )

client = OpenAI(
    base_url="https://spark.cs.trinity.edu/v1",
    api_key=api_key,
)
model = "gemma-3-12b"
max_workers = 2
max_lines = 20  # Set to a positive integer to limit examples per dataset.

# Possible Models:
# qwen2.5-3b
# qwen2.5-7b
# gemma-3-12b
# deepseek-r1-14b
# Arena Model

project_root = Path(__file__).resolve().parents[1]
datasets_dir = project_root / "datasets"
dataset_paths = sorted(datasets_dir.glob("gsm8k-*/test.jsonl"))
results_path = Path(__file__).resolve().parent / "benchmark_results.csv"


def extract_final_answer(text):
    if '####' in text:
        text = text.rsplit('####', 1)[1]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = lines[-1] if lines else ''

    return text.strip().replace(',', '').replace('$', '').rstrip('.').strip()


def prompt_is_correct(prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{prompt.question}\n\n"
                    "Keep your response brief.\n"
                    "Put the final answer after #### on its own line."
                ),
            }
        ],
    )
    generated_text = response.choices[0].message.content
    if generated_text is None:
        raise RuntimeError("The model returned no response content.")

    return extract_final_answer(generated_text) == extract_final_answer(prompt.answer)


if not dataset_paths:
    raise FileNotFoundError(f"No test.jsonl datasets found under {datasets_dir}.")

write_header = not results_path.exists() or results_path.stat().st_size == 0
with results_path.open("a", newline="", encoding="utf-8") as results_file:
    writer = csv.DictWriter(
        results_file,
        fieldnames=("model", "dataset", "correct", "total", "accuracy"),
    )
    if write_header:
        writer.writeheader()

    for dataset_path in dataset_paths:
        df = pd.read_json(dataset_path, lines=True, nrows=max_lines)
        total = len(df)
        correct = 0

        print(f"Running {dataset_path.parent.name}: {total} prompts")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            correct = sum(executor.map(prompt_is_correct, df.itertuples()))

        accuracy = correct / total if total else 0
        writer.writerow(
            {
                "model": model,
                "dataset": dataset_path.parent.name,
                "correct": correct,
                "total": total,
                "accuracy": f"{accuracy:.4f}",
            }
        )
        results_file.flush()
        print(f"{dataset_path.parent.name}: {correct}/{total} = {accuracy:.2%}")

print(f"Results saved to {results_path}")