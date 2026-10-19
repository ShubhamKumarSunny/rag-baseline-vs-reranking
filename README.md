# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked PDF, compared on queries where plain dense retrieval fails.

- **Document:** The Digital Personal Data Protection Act, 2023 (India), `data/DPDP_Act_2023.pdf` (21 pages, 120 chunks)
- **Pipeline 1:** `all-MiniLM-L6-v2` Bi-Encoder + FAISS `IndexFlatIP` (cosine), Top 10, first 3 chunks go to the LLM
- **Pipeline 2:** same Top 10, re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, new Top 3 go to the LLM
- **LLM:** Gemini API (`gemini-3.7-flash`, falling back to `gemini-3.6-flash` / `gemini-3.5-flash` when the free daily quota of a model is used up or a model stops responding for 60 s); same prompt for both pipelines. Without a key the notebook falls back to a local `Qwen/Qwen2.5-0.5B-Instruct`.

Everything is in one notebook: [`RAG_baseline_vs_reranking.ipynb`](RAG_baseline_vs_reranking.ipynb).
The report is [`report/report.pdf`](report/report.pdf).

## Run

```bash
pip install -r requirements.txt
jupyter notebook RAG_baseline_vs_reranking.ipynb     # Run All
```

Put a Gemini key in a `.env` file next to the notebook (`GEMINI_API_KEY=...`, the file is git-ignored) or export it as an environment variable.
The LLM answers shown in the notebook are stored in `results/llm_cache.json`, so *Run All* reproduces them without any API call; delete that file to regenerate them.
The two retrieval models (about 180 MB) are downloaded from Hugging Face on the first run.

## Files

| Path | Content |
|---|---|
| `RAG_baseline_vs_reranking.ipynb` | all code, with outputs |
| `report/report.pdf` | the report |
| `data/DPDP_Act_2023.pdf` | the source document |
| `results/rank_shift_summary.csv` | rank of the answer chunk before / after re-ranking |
| `results/queries_and_answers.json` | Top-10 with both scores, Answer 1 and Answer 2 per query |
| `results/llm_cache.json` | cached LLM answers with the model that produced each one |
| `results/latency_retrieval.csv` | retrieval vs. re-ranking latency (300 measurements per stage) |
| `results/latency_end_to_end.csv` | per-query timings including LLM generation |
| `results/scaling_cross_encoder.csv` | Cross-Encoder run over the whole corpus |
| `results/rank_shift_and_latency.png` | figure |
