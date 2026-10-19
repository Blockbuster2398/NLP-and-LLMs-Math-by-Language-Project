from pathlib import Path

import ollama
import pandas as pd

model = 'llama3.2:1B'
dataset_folder = 'gsm8k-es-100'
dataset = 'test.jsonl'

project_root = Path(__file__).resolve().parents[1]
dataset_path = project_root / 'datasets' / dataset_folder / dataset
df = pd.read_json(dataset_path, lines=True)


def extract_final_answer(text):
    if '####' in text:
        text = text.rsplit('####', 1)[1]
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = lines[-1] if lines else ''

    return text.strip().replace(',', '').replace('$', '').rstrip('.').strip()


total = len(df)
correct = 0

for prompt in df.itertuples():
    response = ollama.generate(
        model=model,
        prompt=f'{prompt.question}\n\nKeep your response brief.\nPut the final answer after #### on its own line.',
    )
    predicted_answer = extract_final_answer(response['response'])
    answer = extract_final_answer(prompt.answer)
    if predicted_answer == answer:
        correct += 1
    else:
        print(f"Incorrect for prompt: {prompt.question}")
        print(f"Expected: {answer}")
        print(f"Got: {predicted_answer}")

print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")