# RAGBench Tech Review

**Jahnavi Ravipudi** (jr84@illinois.edu)
CS 410 — Text Information Systems, UIUC, Spring 2026

Paper being reviewed: [RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2407.11005) (Friel et al., 2024)

---

## What this project does

Most RAG evaluations just check whether the final answer is correct. That tells you very little about *why* something went wrong — was it bad retrieval? Did the model ignore good context? Did it hallucinate?

[RAGBench](https://huggingface.co/datasets/rungalileo/ragbench) tries to fix this with **TRACe**, a set of four metrics that break down what's happening at each stage of the pipeline:

- **Relevance** — of everything we retrieved, how much is actually useful?
- **Utilization** — of the useful stuff we retrieved, how much did the model actually use?
- **Completeness** — how much of the expected answer did the model cover?
- **Adherence** — is the generated answer actually grounded in the context, or is it making things up?

This project applies TRACe to a real retrieval setup (BM25 and dense retrieval with FAISS), compares how retrieval quality affects downstream metrics, and classifies common failure patterns across six datasets from the benchmark.

---

## How it works

Everything runs through `analyze_ragbench.py`. Here's what it does, roughly in order:

1. Loads six RAGBench subsets from HuggingFace (hotpotqa, msmarco, covidqa, expertqa, hagrid, pubmedqa)
2. Loads the MiniLM sentence-transformer for dense retrieval
3. Runs both BM25 and dense/FAISS retrieval at k=2 and k=4, computing relevance and utilization for each config
4. Generates answers with GPT-3.5-turbo under each retrieval config and scores them with adherence/completeness proxies
5. Reproduces the baseline evaluator metrics from the paper (GPT-3.5, RAGAS, TruLens)
6. Analyzes TRACe metric distributions across datasets
7. Classifies each example into a failure mode (irrelevant retrieval, underutilization, hallucination, incomplete answer, or adequate)
8. Computes correlations between the four TRACe metrics
9. Pulls out concrete examples of each failure type
10. Generates all the figures

All figures get saved to `figures/`.

---

## Project structure

```
ragbench_tech_review/
├── analyze_ragbench.py        # Main script
├── ragbench/                  # Vendored from github.com/rungalileo/ragbench
│   ├── constants.py           #   Field names, HuggingFace config
│   ├── evaluation.py          #   RMSE, AUROC, metric calculation
│   ├── calculate_metrics.py   #   CLI to reproduce paper baselines
│   ├── inference.py           #   TruLens / RAGAS annotation wrappers
│   ├── run_inference.py       #   CLI for baseline inference
│   └── trulens_async.py       #   Async TruLens OpenAI provider
├── .env.example               #   API key template
├── requirements.txt
└── README.md
```

---

## Setup and running

You'll need Python 3.9+ and an OpenAI API key for the answer generation step.

```bash
git clone https://github.com/<your-username>/ragbench_tech_review.git
cd ragbench_tech_review

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# add your OpenAI API key to .env
```

To run:

```bash
python analyze_ragbench.py
```

The first run will take a bit longer since it downloads the datasets from HuggingFace and the MiniLM model. After that, everything is cached locally.

---

## Reproducing the paper's baselines

The vendored `ragbench/` package includes the original evaluation scripts from the paper if you want to run them directly:

```bash
python -m ragbench.calculate_metrics --dataset hotpotqa msmarco hagrid expertqa
python -m ragbench.run_inference --dataset msmarco --model trulens --output results
```

---

## Reference

Friel, R., Belyi, M., and Sanyal, A. (2024). *RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems.* arXiv:2407.11005

Evaluation logic adapted from [rungalileo/ragbench](https://github.com/rungalileo/ragbench).