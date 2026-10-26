# NLP-and-LLMs-Math-by-Language-Project
Trinity University Class Project for NLP and LLMs

## Running the benchmark

Install the Python packages used by the benchmark:

```powershell
python -m pip install openai pandas
```

Set your Trinity API key in PowerShell (replace the placeholder with your key):

```powershell
$env:TRINITY_API_KEY = "your-key"
```

Then run the benchmark:

```powershell
python .\Benchmarking\Benchmark.py
```

The benchmark sends up to two requests concurrently to
`https://spark.cs.trinity.edu/v1` using the `qwen2.5-7b` model, evaluating
every `gsm8k-*/test.jsonl` dataset in the `datasets` folder. It writes each
completed dataset's accuracy and counts to
`Benchmarking/benchmark_results.csv`. If the service rate-limits requests,
lower `max_workers` in `Benchmarking/Benchmark.py`. Keep your API key in the
environment variable rather than adding it to the source code.
