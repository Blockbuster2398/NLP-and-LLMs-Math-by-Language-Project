#!/usr/bin/env python3
"""Benchmark the base model and every fine-tuned adapter on every language's test set.

Results go to a CSV with the columns: model,dataset,correct,total,accuracy
Raw model answers go to benchmark_outputs/<model>__<dataset>.jsonl (handy for checking the grader).

Expected project layout (the script finds the project root automatically):

    <root>/datasets/gsm8k-<lang>/test.jsonl
    <root>/fine_tuning/adapters/gemma3-12b-gsm8k-<name>/final/adapter_config.json
    <root>/Benchmarking/benchmark.py            <- this file

Examples:
    python benchmark.py --limit 10                  # quick trial: 10 questions per dataset
    python benchmark.py                             # everything (slow)
    python benchmark.py --adapters zh mix --datasets zh es
    python benchmark.py --no-base --batch-size 16

Re-running resumes: (model, dataset) pairs already in the CSV are skipped unless --overwrite is given.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

from huggingface_hub import login
# login("hf_key")

try:
    from tqdm import tqdm # type: ignore
except ImportError:                      # tqdm is optional
    def tqdm(it, **kwargs):
        return it

BASE_MODEL = "google/gemma-3-12b-it"
BASE_LABEL = "gemma-3-12b"
ADAPTER_PREFIX = "gemma3-12b-gsm8k-"
CSV_FIELDS = ["model", "dataset", "correct", "total", "accuracy"]


# ============================================================================
# GRADING  (edit this section to change how answers are scored)
# ============================================================================
_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def extract_final_number(text):
    """Final number in `text`: after the last '####' if present, else the last number anywhere."""
    if text is None:
        return None
    if "####" in text:
        text = text.split("####")[-1]
    nums = _NUM.findall(text)
    if not nums:
        return None
    return float(nums[-1].replace(",", ""))


def grade(prediction, reference, example):
    """Return True if `prediction` is correct.

    prediction: the model's answer text
    reference:  the dataset's `answer` field
    example:    the whole dataset row (dict with `question` and `answer`)
    """
    pred = extract_final_number(prediction)
    ref = extract_final_number(reference)
    if pred is None or ref is None:
        return False
    return abs(pred - ref) < 1e-4


def build_messages(example):
    """Prompt sent to the model. Matches the training format: just the question."""
    return [{"role": "user", "content": example["question"].strip()}]


# ============================================================================
# Discovery and bookkeeping
# ============================================================================
def find_project_root():
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "datasets").is_dir() and (p / "fine_tuning").is_dir():
            return p
    return here.parent


def find_adapter_dir(folder):
    """Return the directory inside `folder` that holds adapter_config.json, or None."""
    if (folder / "final" / "adapter_config.json").exists():
        return folder / "final"
    if (folder / "adapter_config.json").exists():
        return folder
    checkpoints = []
    for c in folder.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", c.name)
        if m and (c / "adapter_config.json").exists():
            checkpoints.append((int(m.group(1)), c))
    if checkpoints:
        return max(checkpoints)[1]          # latest checkpoint
    return None


def adapter_suffix(folder_name):
    return folder_name[len(ADAPTER_PREFIX):] if folder_name.startswith(ADAPTER_PREFIX) else folder_name


def discover_adapters(adapters_dir, only=None):
    found = []
    for folder in sorted(p for p in Path(adapters_dir).iterdir() if p.is_dir()):
        suffix = adapter_suffix(folder.name)
        if only and suffix not in only:
            continue
        path = find_adapter_dir(folder)
        if path is None:
            print(f"  !! {folder.name}: no adapter_config.json found (checked final/, folder itself, checkpoint-*), skipping")
            continue
        found.append((f"{BASE_LABEL}-ft-{suffix}", path))
    if only:
        missing = set(only) - {adapter_suffix(p.name) for p in Path(adapters_dir).iterdir() if p.is_dir()}
        if missing:
            print(f"  !! no adapter folder for: {sorted(missing)}")
    return found


def discover_datasets(data_root, only=None):
    names = []
    for folder in sorted(p for p in Path(data_root).iterdir() if p.is_dir() and p.name.startswith("gsm8k-")):
        if (folder / "test.jsonl").exists():
            names.append(folder.name)
        else:
            print(f"  !! {folder.name}: no test.jsonl, skipping")
    if only:
        wanted = {o if o.startswith("gsm8k-") else f"gsm8k-{o}" for o in only}
        unknown = wanted - set(names)
        if unknown:
            print(f"  !! requested datasets not found: {sorted(unknown)}")
        names = [n for n in names if n in wanted]
    return names


def load_test_set(data_root, dataset, limit=None):
    with open(Path(data_root) / dataset / "test.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def read_csv(csv_path):
    rows = {}
    if Path(csv_path).exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[(r["model"], r["dataset"])] = {k: r[k] for k in CSV_FIELDS}
    return rows


def update_csv(csv_path, row):
    """Insert or replace the row for (model, dataset)."""
    rows = read_csv(csv_path)
    rows[(row["model"], row["dataset"])] = {
        "model": row["model"], "dataset": row["dataset"],
        "correct": row["correct"], "total": row["total"],
        "accuracy": f"{row['accuracy']:.4f}",
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows.values())


def print_summary(csv_path):
    rows = read_csv(csv_path)
    if not rows:
        return
    models = list(dict.fromkeys(m for m, _ in rows))
    datasets = sorted({d for _, d in rows})
    short = [d.replace("gsm8k-", "") for d in datasets]
    w = max(len(m) for m in models) + 2
    print("\nAccuracy summary (" + str(csv_path) + ")")
    print("model".ljust(w) + "".join(s.rjust(8) for s in short))
    for m in models:
        cells = []
        for d in datasets:
            r = rows.get((m, d))
            cells.append((f"{float(r['accuracy']):.3f}" if r else "-").rjust(8))
        print(m.ljust(w) + "".join(cells))


# ============================================================================
# Model loading and generation (heavy imports are done lazily)
# ============================================================================
def load_base_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


class AdapterSwitcher:
    """Wraps the base model once, then swaps adapters in and out without reloading the 12B weights."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.peft_model = None
        self.current = None
        self.counter = 0

    def activate(self, adapter_path):
        from peft import PeftModel

        name = f"ad{self.counter}"
        self.counter += 1
        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(self.base_model, str(adapter_path), adapter_name=name)
        else:
            self.peft_model.load_adapter(str(adapter_path), adapter_name=name)
            self.peft_model.set_adapter(name)
            if self.current is not None:
                self.peft_model.delete_adapter(self.current)   # free the previous adapter
        self.current = name
        self.peft_model.eval()
        return self.peft_model


def generate_batch(model, tokenizer, batch_messages, max_new_tokens):
    import torch

    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"                  # required for batched generation
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    tokenizer.padding_side = old_side
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None)     # greedy, reproducible
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_robust(model, tokenizer, batch_messages, max_new_tokens):
    """Like generate_batch, but splits the batch in half if the GPU runs out of memory."""
    import torch

    try:
        return generate_batch(model, tokenizer, batch_messages, max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch_messages) == 1:
            raise
        mid = len(batch_messages) // 2
        print(f"  (out of memory: retrying a batch of {len(batch_messages)} as two smaller batches)")
        return (generate_robust(model, tokenizer, batch_messages[:mid], max_new_tokens)
                + generate_robust(model, tokenizer, batch_messages[mid:], max_new_tokens))


# ============================================================================
# Benchmark loop
# ============================================================================
def benchmark_dataset(model, tokenizer, label, dataset, rows, args, output_dir):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["question"]))   # similar lengths per batch
    preds = [None] * len(rows)
    for start in tqdm(range(0, len(order), args.batch_size), desc=f"{label} / {dataset}"):
        idx = order[start:start + args.batch_size]
        outs = generate_robust(model, tokenizer, [build_messages(rows[i]) for i in idx], args.max_new_tokens)
        for i, o in zip(idx, outs):
            preds[i] = o.strip()

    correct, errors, records = 0, 0, []
    for ex, pred in zip(rows, preds):
        try:
            ok = bool(grade(pred, ex["answer"], ex))
        except Exception as e:                         # a grader bug should not kill a long run
            ok, errors = False, errors + 1
            if errors <= 3:
                print("  Grader error:", repr(e))
        correct += ok
        records.append({"question": ex["question"], "reference": ex["answer"],
                        "prediction": pred, "correct": ok})

    with open(output_dir / f"{label}__{dataset}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"model": label, "dataset": dataset, "correct": correct,
            "total": len(rows), "accuracy": correct / max(len(rows), 1)}


def main(argv=None):
    root = find_project_root()
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters-dir", type=Path, default=root / "fine_tuning" / "adapters")
    ap.add_argument("--data-root", type=Path, default=root / "datasets")
    ap.add_argument("--csv", type=Path, default=here / "benchmark_results.csv")
    ap.add_argument("--output-dir", type=Path, default=here / "benchmark_outputs")
    ap.add_argument("--adapters", nargs="*", help="only these adapters, by suffix (e.g. zh mix mix_full)")
    ap.add_argument("--datasets", nargs="*", help="only these datasets (e.g. zh es, or gsm8k-zh)")
    ap.add_argument("--no-base", action="store_true", help="skip benchmarking the base model")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test questions per dataset")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true", help="redo (model, dataset) pairs already in the CSV")
    args = ap.parse_args(argv)

    print("Project root:", root)
    datasets = discover_datasets(args.data_root, args.datasets)
    adapters = discover_adapters(args.adapters_dir, args.adapters)
    if not datasets:
        sys.exit("No datasets found.")

    plan = []                                           # (label, adapter_path or None)
    if not args.no_base:
        plan.append((BASE_LABEL, None))
    plan.extend((label, path) for label, path in adapters)

    done = set() if args.overwrite else set(read_csv(args.csv))
    todo = {label: [d for d in datasets if (label, d) not in done] for label, _ in plan}
    plan = [(label, path) for label, path in plan if todo[label]]

    print(f"Datasets ({len(datasets)}): {', '.join(d.replace('gsm8k-', '') for d in datasets)}")
    print("Models to run:")
    for label, path in plan:
        print(f"  {label:32s} {len(todo[label])} dataset(s)   {path if path else '(base model)'}")
    if not plan:
        print("Nothing to do: every (model, dataset) pair is already in", args.csv)
        print_summary(args.csv)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\nLoading base model...")
    base_model, tokenizer = load_base_model()
    switcher = AdapterSwitcher(base_model)

    # The base model must run first: attaching an adapter modifies the base model in place.
    plan.sort(key=lambda item: item[1] is not None)

    for label, path in plan:
        print(f"\n=== {label} ===")
        model = base_model if path is None else switcher.activate(path)
        for dataset in todo[label]:
            rows = load_test_set(args.data_root, dataset, args.limit)
            row = benchmark_dataset(model, tokenizer, label, dataset, rows, args, args.output_dir)
            print(f"  {dataset}: {row['correct']}/{row['total']} = {row['accuracy']:.4f}")
            update_csv(args.csv, row)                  # saved after every dataset, so interruptions lose little

    print_summary(args.csv)


if __name__ == "__main__":
    main()
