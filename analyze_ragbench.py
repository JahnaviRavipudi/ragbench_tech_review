import os
import json
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

from ragbench import HUGGINGFACE_REPO_NAME, RAGBenchFields

warnings.filterwarnings('ignore')

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: openai package not installed. LLM generation will be skipped.")
    print("  Install with: pip install openai")

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Six datasets we're evaluating
DATASETS = ['hotpotqa', 'msmarco', 'covidqa', 'expertqa', 'hagrid', 'pubmedqa']


# Data loading

def load_all_datasets(dataset_names, split='test'):
    """Download test splits from HuggingFace."""
    data = {}
    for name in dataset_names:
        print(f"  Loading {name}...")
        try:
            ds = load_dataset(HUGGINGFACE_REPO_NAME, name, split=split)
            data[name] = ds
            print(f"    -> {ds.num_rows} examples")
        except Exception as e:
            print(f"    x Failed: {e}")
    return data


# Retrieval using BM25 and MiniLM + FAISS

def flatten_sentences(documents_sentences):
    """Flatten the nested sentence structure into a list of (key, text) tuples."""
    sentences = []
    for doc_group in documents_sentences:
        for sent in doc_group:
            sentences.append((sent[0], sent[1]))
    return sentences


def get_sentence_to_doc_map(documents_sentences):
    """Map each sentence key back to the index of the document it came from."""
    sent_to_doc = {}
    for doc_idx, doc_group in enumerate(documents_sentences):
        for sent in doc_group:
            sent_to_doc[sent[0]] = doc_idx
    return sent_to_doc


def bm25_retrieve(query, sentences, top_k=None):
    """BM25 sparse retrieval over document sentences."""
    tokenized_corpus = [s[1].lower().split() for s in sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    ranked_indices = np.argsort(scores)[::-1]
    if top_k:
        ranked_indices = ranked_indices[:top_k]

    return [(sentences[i][0], scores[i]) for i in ranked_indices]


def dense_retrieve(query, sentences, model, top_k=None):
    """Dense retrieval using sentence-transformers (all-MiniLM-L6-v2) + FAISS."""
    texts = [s[1] for s in sentences]

    # Encode query and sentences
    query_emb = model.encode([query], normalize_embeddings=True)
    sent_embs = model.encode(texts, normalize_embeddings=True)

    # Build FAISS index
    dim = sent_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(sent_embs.astype(np.float32))

    k = top_k if top_k else len(sentences)
    scores, indices = index.search(query_emb.astype(np.float32), k)

    return [(sentences[int(idx)][0], float(score))
            for idx, score in zip(indices[0], scores[0])]


def compute_retrieval_relevance(ranked_keys, relevant_keys):
    """Compute context relevance: fraction of retrieved context that is relevant."""
    retrieved_set = set(ranked_keys)
    relevant_set = set(relevant_keys)
    if len(retrieved_set) == 0:
        return 0.0
    return len(retrieved_set & relevant_set) / len(retrieved_set)


def compute_retrieval_utilization(ranked_keys, utilized_keys):
    """Compute context utilization: fraction of retrieved context that is utilized.
    
    Per TRACe definition: Utilization = len(utilized ∩ retrieved) / len(retrieved)
    """
    retrieved_set = set(ranked_keys)
    utilized_set = set(utilized_keys)

    if len(retrieved_set) == 0:
        return 0.0
    return len(retrieved_set & utilized_set) / len(retrieved_set)


def retrieve_parent_documents(ranked_sentence_keys, documents, sent_to_doc, top_k_docs=None):
    seen_docs = set()
    ordered_docs = []
    for sent_key in ranked_sentence_keys:
        doc_idx = sent_to_doc.get(sent_key)
        if doc_idx is not None and doc_idx not in seen_docs:
            seen_docs.add(doc_idx)
            ordered_docs.append(documents[doc_idx])
            if top_k_docs and len(ordered_docs) >= top_k_docs:
                break
    return ordered_docs


def compute_retrieval_metrics_for_example(example, model, k_values=[2, 4]):
    query = example['question']
    sentences = flatten_sentences(example['documents_sentences'])
    relevant_keys = set(example['all_relevant_sentence_keys'])
    utilized_keys = set(example['all_utilized_sentence_keys'])

    results = {}

    # Original retrieval (all documents as provided by the dataset)
    results['original'] = {
        'relevance': example['relevance_score'],
        'utilization': example['utilization_score'],
        'adherence': float(example['adherence_score']) if example['adherence_score'] is not None else np.nan,
        'completeness': example['completeness_score'],
        'num_sentences': len(sentences),
    }

    # BM25 retrieval at different k
    bm25_ranked = bm25_retrieve(query, sentences)
    for k in k_values:
        label = f'bm25_k{k}'
        top_keys = [r[0] for r in bm25_ranked[:k]]
        rel = compute_retrieval_relevance(top_keys, relevant_keys)
        util = compute_retrieval_utilization(top_keys, utilized_keys)
        results[label] = {
            'relevance': rel,
            'utilization': util,
            'num_retrieved': len(top_keys),
            'ranked_keys': top_keys[:5],
        }

    # Dense retrieval at different k
    dense_ranked = dense_retrieve(query, sentences, model)
    for k in k_values:
        label = f'dense_k{k}'
        top_keys = [r[0] for r in dense_ranked[:k]]
        rel = compute_retrieval_relevance(top_keys, relevant_keys)
        util = compute_retrieval_utilization(top_keys, utilized_keys)
        results[label] = {
            'relevance': rel,
            'utilization': util,
            'num_retrieved': len(top_keys),
            'ranked_keys': top_keys[:5],
        }

    return results


def run_retrieval_experiment(dataset, dataset_name, model, sample_size=50):
    print(f"\n  Running retrieval on {dataset_name} (n={sample_size})...")

    # Sample deterministically
    np.random.seed(42)
    indices = np.random.choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)
    indices.sort()

    all_results = []
    for idx_num, idx in enumerate(indices):
        example = dataset[int(idx)]
        try:
            result = compute_retrieval_metrics_for_example(example, model)
            result['idx'] = int(idx)
            result['question'] = example['question']
            all_results.append(result)
        except Exception as e:
            print(f"    Warning: skipped example {idx}: {e}")

        if (idx_num + 1) % 10 == 0:
            print(f"    Processed {idx_num + 1}/{len(indices)}...")

    return all_results


# Answer generation using GPT-3.5-turbo with retrieved context

RAG_PROMPT_TEMPLATE = """Answer the following question based only on the provided context documents. 
Be concise and factual. If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer_with_llm(client, question, context_docs, model="gpt-3.5-turbo"):
    context_text = "\n\n".join(context_docs)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    API error: {e}")
        return None


# Lightweight proxies for adherence and completeness on generated answers

def _normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _get_ngrams(tokens, n):
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_adherence_proxy(generated_answer, context_sentences):
    """Proxy for adherence (faithfulness), uses weighted n-gram overlap"""
    if not generated_answer or not context_sentences:
        return np.nan

    answer_norm = _normalize_text(generated_answer)
    context_norm = _normalize_text(" ".join(context_sentences))

    answer_tokens = answer_norm.split()
    context_tokens = context_norm.split()

    if len(answer_tokens) < 2:
        return 1.0 if all(t in context_tokens for t in answer_tokens) else 0.0

    # Weighted n-gram overlap: unigrams (0.4), bigrams (0.35), trigrams (0.25)
    weights = {1: 0.4, 2: 0.35, 3: 0.25}
    total_score = 0.0

    for n, weight in weights.items():
        answer_ngrams = _get_ngrams(answer_tokens, n)
        context_ngrams = _get_ngrams(context_tokens, n)
        if len(answer_ngrams) == 0:
            total_score += weight  # trivially covered
            continue
        overlap = len(answer_ngrams & context_ngrams)
        total_score += weight * (overlap / len(answer_ngrams))

    return total_score


def compute_completeness_proxy(generated_answer, reference_answer):
    """Proxy for completeness uses weighted n-gram recall (unigram + bigram)"""
    if not generated_answer or not reference_answer:
        return np.nan

    gen_norm = _normalize_text(generated_answer)
    ref_norm = _normalize_text(reference_answer)

    gen_tokens = gen_norm.split()
    ref_tokens = ref_norm.split()

    if len(ref_tokens) < 2:
        return 1.0 if all(t in gen_tokens for t in ref_tokens) else 0.0

    # Weighted: unigram recall (0.5), bigram recall (0.5)
    weights = {1: 0.5, 2: 0.5}
    total_score = 0.0

    for n, weight in weights.items():
        ref_ngrams = _get_ngrams(ref_tokens, n)
        gen_ngrams = _get_ngrams(gen_tokens, n)
        if len(ref_ngrams) == 0:
            total_score += weight
            continue
        overlap = len(ref_ngrams & gen_ngrams)
        total_score += weight * (overlap / len(ref_ngrams))

    return total_score


def evaluate_generated_answer(generated_answer, context_sentences, reference_answer):
    """Compute proxy TRACe metrics (adherence + completeness) for generated answer"""
    return {
        'adherence_proxy': compute_adherence_proxy(generated_answer, context_sentences),
        'completeness_proxy': compute_completeness_proxy(generated_answer, reference_answer),
    }


def run_llm_generation_experiment(dataset, dataset_name, dense_model, openai_client,
                                   sample_size=20):
    print(f"\n  Generating answers for {dataset_name} (n={sample_size})...")

    np.random.seed(42)
    indices = np.random.choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)
    indices.sort()

    generation_results = []

    for idx_num, idx in enumerate(indices):
        example = dataset[int(idx)]
        question = example['question']
        sentences = flatten_sentences(example['documents_sentences'])
        sent_to_doc = get_sentence_to_doc_map(example['documents_sentences'])
        all_docs = example['documents']
        all_sentence_texts = [s[1] for s in sentences]

        result = {
            'idx': int(idx),
            'question': question,
            'original_response': example['response'],
            'original_adherence': float(example['adherence_score']) if example['adherence_score'] is not None else None,
            'original_completeness': example['completeness_score'],
        }

        # Generate with all documents
        result['original_generated'] = generate_answer_with_llm(
            openai_client, question, all_docs
        )
        if result['original_generated']:
            orig_eval = evaluate_generated_answer(
                result['original_generated'], all_sentence_texts, example['response']
            )
            result['original_adherence_proxy'] = orig_eval['adherence_proxy']
            result['original_completeness_proxy'] = orig_eval['completeness_proxy']

        # Generate with BM25 top-2 sentences
        bm25_ranked = bm25_retrieve(question, sentences)
        bm25_top_keys = [r[0] for r in bm25_ranked[:2]]
        bm25_docs = retrieve_parent_documents(bm25_top_keys, all_docs, sent_to_doc)
        bm25_context_texts = [s[1] for s in sentences if s[0] in set(bm25_top_keys)]

        result['bm25_k2_generated'] = generate_answer_with_llm(
            openai_client, question, bm25_docs
        )
        result['bm25_k2_context_keys'] = bm25_top_keys
        if result['bm25_k2_generated']:
            bm25_eval = evaluate_generated_answer(
                result['bm25_k2_generated'], bm25_context_texts, example['response']
            )
            result['bm25_k2_adherence_proxy'] = bm25_eval['adherence_proxy']
            result['bm25_k2_completeness_proxy'] = bm25_eval['completeness_proxy']

        # Same thing but with dense retrieval top-2
        dense_ranked = dense_retrieve(question, sentences, dense_model)
        dense_top_keys = [r[0] for r in dense_ranked[:2]]
        dense_docs = retrieve_parent_documents(dense_top_keys, all_docs, sent_to_doc)
        dense_context_texts = [s[1] for s in sentences if s[0] in set(dense_top_keys)]

        result['dense_k2_generated'] = generate_answer_with_llm(
            openai_client, question, dense_docs
        )
        result['dense_k2_context_keys'] = dense_top_keys
        if result['dense_k2_generated']:
            dense_eval = evaluate_generated_answer(
                result['dense_k2_generated'], dense_context_texts, example['response']
            )
            result['dense_k2_adherence_proxy'] = dense_eval['adherence_proxy']
            result['dense_k2_completeness_proxy'] = dense_eval['completeness_proxy']

        # Store ground truth for comparison
        result['relevant_keys'] = example['all_relevant_sentence_keys']
        result['bm25_k2_has_relevant'] = bool(set(bm25_top_keys) & set(example['all_relevant_sentence_keys']))
        result['dense_k2_has_relevant'] = bool(set(dense_top_keys) & set(example['all_relevant_sentence_keys']))

        generation_results.append(result)

        if (idx_num + 1) % 5 == 0:
            print(f"    Generated {idx_num + 1}/{len(indices)} examples...")

    return generation_results


def plot_generation_comparison(generation_results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ds_names = list(generation_results.keys())

    # Left: how often did each retriever grab relevant context?
    ax1 = axes[0]
    bm25_hits, dense_hits = [], []
    for ds_name in ds_names:
        examples = generation_results[ds_name]
        bm25_hits.append(np.mean([ex['bm25_k2_has_relevant'] for ex in examples]) * 100)
        dense_hits.append(np.mean([ex['dense_k2_has_relevant'] for ex in examples]) * 100)

    x = np.arange(len(ds_names))
    width = 0.35
    ax1.bar(x - width/2, bm25_hits, width, label='BM25 k=2', color='#3498db')
    ax1.bar(x + width/2, dense_hits, width, label='Dense k=2', color='#e74c3c')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_names, rotation=30)
    ax1.set_ylabel('Retrieval Hit Rate (%)')
    ax1.set_title('Relevant Context Retrieved (top-2)', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 105)

    # Middle: adherence (faithfulness) of generated answers
    ax2 = axes[1]
    orig_adh, bm25_adh, dense_adh = [], [], []
    for ds_name in ds_names:
        examples = generation_results[ds_name]
        orig_adh.append(np.nanmean([ex.get('original_adherence_proxy', np.nan) for ex in examples]))
        bm25_adh.append(np.nanmean([ex.get('bm25_k2_adherence_proxy', np.nan) for ex in examples]))
        dense_adh.append(np.nanmean([ex.get('dense_k2_adherence_proxy', np.nan) for ex in examples]))

    width = 0.25
    ax2.bar(x - width, orig_adh, width, label='All docs', color='gray')
    ax2.bar(x, bm25_adh, width, label='BM25 k=2', color='#3498db')
    ax2.bar(x + width, dense_adh, width, label='Dense k=2', color='#e74c3c')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ds_names, rotation=30)
    ax2.set_ylabel('Adherence Proxy')
    ax2.set_title('Faithfulness by Retrieval Config', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05)

    # Right: completeness (how much of the reference answer is covered)
    ax3 = axes[2]
    orig_comp, bm25_comp, dense_comp = [], [], []
    for ds_name in ds_names:
        examples = generation_results[ds_name]
        orig_comp.append(np.nanmean([ex.get('original_completeness_proxy', np.nan) for ex in examples]))
        bm25_comp.append(np.nanmean([ex.get('bm25_k2_completeness_proxy', np.nan) for ex in examples]))
        dense_comp.append(np.nanmean([ex.get('dense_k2_completeness_proxy', np.nan) for ex in examples]))

    ax3.bar(x - width, orig_comp, width, label='All docs', color='gray')
    ax3.bar(x, bm25_comp, width, label='BM25 k=2', color='#3498db')
    ax3.bar(x + width, dense_comp, width, label='Dense k=2', color='#e74c3c')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ds_names, rotation=30)
    ax3.set_ylabel('Completeness Proxy')
    ax3.set_title('Answer Completeness by Retrieval Config', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'generation_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# Baseline evaluator metrics to reproduce GPT-3.5 / RAGAS / TruLens results

def rmse(trues, preds):
    trues = np.array(trues, dtype=float)
    preds = np.array(preds, dtype=float)
    valid = ~np.isnan(preds) & ~np.isnan(trues)
    if valid.sum() == 0:
        return float('nan')
    return np.sqrt(np.mean((preds[valid] - trues[valid])**2))


def auroc_safe(trues, preds):
    trues = np.array(trues, dtype=float)
    preds = np.array(preds, dtype=float)
    valid = ~np.isnan(trues) & ~np.isnan(preds)
    trues, preds = trues[valid], preds[valid]
    if len(np.unique(trues)) < 2:
        return float('nan')
    return roc_auc_score(trues, preds)


def compute_baseline_metrics(ds):
    """Reproduce GPT-3.5, RAGAS, and TruLens baseline metrics from the paper."""
    results = {}
    adherence_gt = np.array(ds[RAGBenchFields.ADHERENCE], dtype=float)
    halluc_gt = 1.0 - adherence_gt
    relevance_gt = ds[RAGBenchFields.RELEVANCE]
    utilization_gt = ds[RAGBenchFields.UTILIZATION]

    # GPT-3.5
    if RAGBenchFields.GPT35_ADHERENCE_PRED in ds.column_names:
        gpt_adh = 1.0 - np.array(ds[RAGBenchFields.GPT35_ADHERENCE_PRED], dtype=float)
        results['gpt35_hallucination_auroc'] = auroc_safe(halluc_gt, gpt_adh)
    if RAGBenchFields.GPT35_RELEVANCE_PRED in ds.column_names:
        results['gpt35_relevance_rmse'] = rmse(relevance_gt, ds[RAGBenchFields.GPT35_RELEVANCE_PRED])
    if RAGBenchFields.GPT35_UTILIZATION_PRED in ds.column_names:
        results['gpt35_utilization_rmse'] = rmse(utilization_gt, ds[RAGBenchFields.GPT35_UTILIZATION_PRED])

    # RAGAS
    if RAGBenchFields.RAGAS_ADHERENCE_PRED in ds.column_names:
        ragas_adh = 1.0 - np.array(ds[RAGBenchFields.RAGAS_ADHERENCE_PRED], dtype=float)
        results['ragas_hallucination_auroc'] = auroc_safe(halluc_gt, ragas_adh)
    if RAGBenchFields.RAGAS_RELEVANCE_PRED in ds.column_names:
        results['ragas_relevance_rmse'] = rmse(relevance_gt, ds[RAGBenchFields.RAGAS_RELEVANCE_PRED])

    # TruLens
    if RAGBenchFields.TRULENS_ADHERENCE_PRED in ds.column_names:
        trulens_adh = 1.0 - np.array(ds[RAGBenchFields.TRULENS_ADHERENCE_PRED], dtype=float)
        results['trulens_hallucination_auroc'] = auroc_safe(halluc_gt, trulens_adh)
    if RAGBenchFields.TRULENS_RELEVANCE_PRED in ds.column_names:
        results['trulens_relevance_rmse'] = rmse(relevance_gt, ds[RAGBenchFields.TRULENS_RELEVANCE_PRED])

    return results


# TRACe distributions and failure mode classification

def trace_distributions(datasets_dict):
    """Compute per-dataset TRACe metric distributions."""
    rows = []
    for name, ds in datasets_dict.items():
        for metric_field, metric_name in [
            (RAGBenchFields.RELEVANCE, 'Relevance'),
            (RAGBenchFields.UTILIZATION, 'Utilization'),
            (RAGBenchFields.COMPLETENESS, 'Completeness'),
        ]:
            if metric_field in ds.column_names:
                vals = np.array(ds[metric_field], dtype=float)
                vals = vals[~np.isnan(vals)]
                for v in vals:
                    rows.append({'dataset': name, 'metric': metric_name, 'value': v})
        if RAGBenchFields.ADHERENCE in ds.column_names:
            vals = np.array(ds[RAGBenchFields.ADHERENCE], dtype=float)
            vals = vals[~np.isnan(vals)]
            for v in vals:
                rows.append({'dataset': name, 'metric': 'Adherence', 'value': v})
    return pd.DataFrame(rows)


def classify_failure_modes(ds, name):
    """Classify each example into a failure mode based on TRACe thresholds."""
    failures = defaultdict(int)
    total = ds.num_rows
    rel = np.array(ds[RAGBenchFields.RELEVANCE], dtype=float)
    util = np.array(ds[RAGBenchFields.UTILIZATION], dtype=float)
    comp = np.array(ds[RAGBenchFields.COMPLETENESS], dtype=float)
    adh = np.array(ds[RAGBenchFields.ADHERENCE], dtype=float)

    for i in range(total):
        r, u, c, a = rel[i], util[i], comp[i], adh[i]
        if any(np.isnan(x) for x in [r, u, c, a]):
            continue
        if r < 0.3:
            failures['Irrelevant Retrieval'] += 1
        elif u < 0.3:
            failures['Context Underutilization'] += 1
        elif a < 0.5:
            failures['Hallucination'] += 1
        elif c < 0.5:
            failures['Incomplete Answer'] += 1
        else:
            failures['Adequate'] += 1
    return dict(failures), total


def metric_correlations(datasets_dict):
    """Compute pairwise Pearson correlations between TRACe metrics."""
    all_rel, all_util, all_comp, all_adh = [], [], [], []
    for ds in datasets_dict.values():
        all_rel.extend(ds[RAGBenchFields.RELEVANCE])
        all_util.extend(ds[RAGBenchFields.UTILIZATION])
        all_comp.extend(ds[RAGBenchFields.COMPLETENESS])
        all_adh.extend(ds[RAGBenchFields.ADHERENCE])
    df = pd.DataFrame({
        'Relevance': np.array(all_rel, dtype=float),
        'Utilization': np.array(all_util, dtype=float),
        'Completeness': np.array(all_comp, dtype=float),
        'Adherence': np.array(all_adh, dtype=float),
    }).dropna()
    return df.corr()


def find_representative_examples(ds, name, n=3):
    """Find illustrative examples of each failure mode."""
    examples = {}
    rel = np.array(ds[RAGBenchFields.RELEVANCE], dtype=float)
    util = np.array(ds[RAGBenchFields.UTILIZATION], dtype=float)
    adh = np.array(ds[RAGBenchFields.ADHERENCE], dtype=float)
    comp = np.array(ds[RAGBenchFields.COMPLETENESS], dtype=float)

    idx = np.argsort(rel)[:n]
    examples['low_relevance'] = [
        {'idx': int(i), 'relevance': float(rel[i]),
         'question': ds[int(i)]['question'][:200],
         'response': ds[int(i)]['response'][:200]}
        for i in idx if not np.isnan(rel[i])
    ]
    mask = rel > 0.5
    util_masked = np.where(mask, util, 999)
    idx = np.argsort(util_masked)[:n]
    examples['low_utilization'] = [
        {'idx': int(i), 'utilization': float(util[i]), 'relevance': float(rel[i]),
         'question': ds[int(i)]['question'][:200]}
        for i in idx if util_masked[i] < 999
    ]
    idx = np.argsort(adh)[:n]
    examples['hallucination'] = [
        {'idx': int(i), 'adherence': float(adh[i]),
         'question': ds[int(i)]['question'][:200],
         'response': ds[int(i)]['response'][:200]}
        for i in idx if not np.isnan(adh[i])
    ]
    return examples


# Plotting -- all the figures for the report

def plot_trace_distributions(dist_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.ravel(), ['Relevance', 'Utilization', 'Completeness', 'Adherence']):
        sub = dist_df[dist_df['metric'] == metric]
        if sub.empty:
            continue
        sns.boxplot(data=sub, x='dataset', y='value', ax=ax, palette='Set2')
        ax.set_title(f'{metric} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=35)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'trace_distributions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_failure_modes(failure_data):
    categories = ['Irrelevant Retrieval', 'Context Underutilization',
                   'Hallucination', 'Incomplete Answer', 'Adequate']
    ds_names = list(failure_data.keys())
    data = {cat: [] for cat in categories}
    for name in ds_names:
        modes, total = failure_data[name]
        for cat in categories:
            data[cat].append(modes.get(cat, 0) / total * 100)
    x = np.arange(len(ds_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c', '#f39c12', '#e67e22', '#3498db', '#2ecc71']
    for i, (cat, color) in enumerate(zip(categories, colors)):
        ax.bar(x + i * width, data[cat], width, label=cat, color=color)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(ds_names, rotation=30)
    ax.set_ylabel('Percentage of Examples (%)')
    ax.set_title('Failure Mode Breakdown by Dataset', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'failure_modes.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_baseline_comparison(baseline_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ds_names = list(baseline_results.keys())
    for method, label, color in [
        ('gpt35_hallucination_auroc', 'GPT-3.5', '#3498db'),
        ('ragas_hallucination_auroc', 'RAGAS', '#e74c3c'),
        ('trulens_hallucination_auroc', 'TruLens', '#2ecc71'),
    ]:
        vals = [baseline_results[d].get(method, float('nan')) for d in ds_names]
        ax1.plot(ds_names, vals, 'o-', label=label, color=color, markersize=6)
    ax1.set_ylabel('Hallucination AUROC')
    ax1.set_title('Hallucination Detection (AUROC)', fontweight='bold')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=30)
    ax1.set_ylim(0.0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    for method, label, color in [
        ('gpt35_relevance_rmse', 'GPT-3.5', '#3498db'),
        ('ragas_relevance_rmse', 'RAGAS', '#e74c3c'),
        ('trulens_relevance_rmse', 'TruLens', '#2ecc71'),
    ]:
        vals = [baseline_results[d].get(method, float('nan')) for d in ds_names]
        ax2.plot(ds_names, vals, 's-', label=label, color=color, markersize=6)
    ax2.set_ylabel('Relevance RMSE')
    ax2.set_title('Context Relevance Estimation (RMSE)', fontweight='bold')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'baseline_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_heatmap(corr_matrix):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                vmin=-1, vmax=1, center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('TRACe Metric Correlations', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'metric_correlations.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_adherence_by_dataset(datasets_dict):
    names, rates = [], []
    for name, ds in datasets_dict.items():
        adh = np.array(ds[RAGBenchFields.ADHERENCE], dtype=float)
        adh = adh[~np.isnan(adh)]
        halluc_rate = (adh < 1.0).mean() * 100
        names.append(name)
        rates.append(halluc_rate)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_list = sns.color_palette('Set2', len(names))
    ax.barh(names, rates, color=colors_list)
    ax.set_xlabel('Hallucination Rate (%)')
    ax.set_title('Hallucination Rate by Dataset', fontweight='bold')
    for i, v in enumerate(rates):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'hallucination_rates.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_retrieval_comparison(retrieval_results):
    """Compare BM25 vs dense retrieval relevance and utilization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ds_names = list(retrieval_results.keys())
    configs = ['bm25_k2', 'bm25_k4', 'dense_k2', 'dense_k4']
    labels = ['BM25 k=2', 'BM25 k=4', 'Dense k=2', 'Dense k=4']
    colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
    markers = ['o', 's', 'o', 's']

    # Relevance
    for config, label, color, marker in zip(configs, labels, colors, markers):
        vals = []
        for ds_name in ds_names:
            examples = retrieval_results[ds_name]
            mean_rel = np.mean([ex[config]['relevance'] for ex in examples if config in ex])
            vals.append(mean_rel)
        ax1.plot(ds_names, vals, f'{marker}-', label=label, color=color, markersize=6)

    # Add original retrieval line
    orig_vals = []
    for ds_name in ds_names:
        examples = retrieval_results[ds_name]
        mean_rel = np.mean([ex['original']['relevance'] for ex in examples])
        orig_vals.append(mean_rel)
    ax1.plot(ds_names, orig_vals, 'D--', label='Original', color='gray', markersize=6)

    ax1.set_ylabel('Mean Relevance')
    ax1.set_title('Retrieval Relevance by Method', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='x', rotation=30)

    # Utilization
    for config, label, color, marker in zip(configs, labels, colors, markers):
        vals = []
        for ds_name in ds_names:
            examples = retrieval_results[ds_name]
            mean_util = np.mean([ex[config]['utilization'] for ex in examples if config in ex])
            vals.append(mean_util)
        ax2.plot(ds_names, vals, f'{marker}-', label=label, color=color, markersize=6)

    orig_vals = []
    for ds_name in ds_names:
        examples = retrieval_results[ds_name]
        mean_util = np.mean([ex['original']['utilization'] for ex in examples])
        orig_vals.append(mean_util)
    ax2.plot(ds_names, orig_vals, 'D--', label='Original', color='gray', markersize=6)

    ax2.set_ylabel('Mean Utilization')
    ax2.set_title('Context Utilization by Method', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'retrieval_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_k_sensitivity(retrieval_results):
    """Shows how retrieval depth affects relevance andutilization for BM25 vs dense retrieval"""
    bm25_k2_rel, bm25_k4_rel = [], []
    dense_k2_rel, dense_k4_rel = [], []
    bm25_k2_util, bm25_k4_util = [], []
    dense_k2_util, dense_k4_util = [], []

    for ds_name, examples in retrieval_results.items():
        for ex in examples:
            if 'bm25_k2' in ex:
                bm25_k2_rel.append(ex['bm25_k2']['relevance'])
                bm25_k2_util.append(ex['bm25_k2']['utilization'])
            if 'bm25_k4' in ex:
                bm25_k4_rel.append(ex['bm25_k4']['relevance'])
                bm25_k4_util.append(ex['bm25_k4']['utilization'])
            if 'dense_k2' in ex:
                dense_k2_rel.append(ex['dense_k2']['relevance'])
                dense_k2_util.append(ex['dense_k2']['utilization'])
            if 'dense_k4' in ex:
                dense_k4_rel.append(ex['dense_k4']['relevance'])
                dense_k4_util.append(ex['dense_k4']['utilization'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ks = ['k=2', 'k=4']
    bm25_rel_means = [np.mean(bm25_k2_rel), np.mean(bm25_k4_rel)]
    dense_rel_means = [np.mean(dense_k2_rel), np.mean(dense_k4_rel)]

    ax1.plot(ks, bm25_rel_means, 'o-', label='BM25', color='#3498db', markersize=8, linewidth=2)
    ax1.plot(ks, dense_rel_means, 's-', label='Dense (MiniLM)', color='#e74c3c', markersize=8, linewidth=2)
    ax1.set_ylabel('Mean Relevance')
    ax1.set_title('Relevance vs. Retrieval Depth', fontweight='bold')
    ax1.legend()

    bm25_util_means = [np.mean(bm25_k2_util), np.mean(bm25_k4_util)]
    dense_util_means = [np.mean(dense_k2_util), np.mean(dense_k4_util)]

    ax2.plot(ks, bm25_util_means, 'o-', label='BM25', color='#3498db', markersize=8, linewidth=2)
    ax2.plot(ks, dense_util_means, 's-', label='Dense (MiniLM)', color='#e74c3c', markersize=8, linewidth=2)
    ax2.set_ylabel('Mean Utilization')
    ax2.set_title('Utilization vs. Retrieval Depth', fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'k_sensitivity.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_trace_summary_table(datasets_dict):
    """Renders a clean table showing mean TRACe scores per dataset"""
    rows = []
    for name, ds in datasets_dict.items():
        rel = np.nanmean(np.array(ds[RAGBenchFields.RELEVANCE], dtype=float))
        util = np.nanmean(np.array(ds[RAGBenchFields.UTILIZATION], dtype=float))
        comp = np.nanmean(np.array(ds[RAGBenchFields.COMPLETENESS], dtype=float))
        adh = np.nanmean(np.array(ds[RAGBenchFields.ADHERENCE], dtype=float))
        n = ds.num_rows
        rows.append({
            'Dataset': name,
            'N': n,
            'Relevance': f'{rel:.3f}',
            'Utilization': f'{util:.3f}',
            'Completeness': f'{comp:.3f}',
            'Adherence': f'{adh:.3f}',
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 2.5 + 0.4 * len(rows)))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(len(df.columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row shading
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')

    ax.set_title('Mean TRACe Metrics by Dataset', fontweight='bold', fontsize=13, pad=20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'trace_summary_table.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")



def main():
    print("=" * 60)
    print("RAGBench Tech Review - Analysis Pipeline")
    print("=" * 60)

    # Pull all six dataset subsets from HuggingFace
    print("\nLoading RAGBench datasets...")
    datasets_dict = load_all_datasets(DATASETS)
    if not datasets_dict:
        print("ERROR: No datasets loaded.")
        return

    # MiniLM model for dense retrieval
    print("\nLoading sentence-transformer model (all-MiniLM-L6-v2)...")
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("    Model loaded.")

    # Run BM25 and dense retrieval on a sample from each dataset
    print("\nRunning retrieval experiments (BM25 + Dense/FAISS)...")
    retrieval_results = {}
    for ds_name, ds in datasets_dict.items():
        retrieval_results[ds_name] = run_retrieval_experiment(
            ds, ds_name, dense_model, sample_size=50
        )

    # Aggregate and save retrieval numbers
    retrieval_summary = {}
    for ds_name, examples in retrieval_results.items():
        summary = {}
        for config in ['original', 'bm25_k2', 'bm25_k4', 'dense_k2', 'dense_k4']:
            rels = [ex[config]['relevance'] for ex in examples if config in ex]
            utils = [ex[config]['utilization'] for ex in examples if config in ex]
            summary[config] = {
                'mean_relevance': float(np.mean(rels)) if rels else None,
                'mean_utilization': float(np.mean(utils)) if utils else None,
                'n': len(rels),
            }
        retrieval_summary[ds_name] = summary

    with open(os.path.join(FIGURES_DIR, 'retrieval_summary.json'), 'w') as f:
        json.dump(retrieval_summary, f, indent=2)
    print("\n  Retrieval summary:")
    for ds_name, summary in retrieval_summary.items():
        print(f"\n  {ds_name}:")
        for config, vals in summary.items():
            if vals['mean_relevance'] is not None:
                print(f"    {config:12s}: relevance={vals['mean_relevance']:.3f}, utilization={vals['mean_utilization']:.3f}")

    # Generate answers under each retrieval config and compare adherence/completeness
    generation_results = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if OPENAI_AVAILABLE and api_key:
        print("\nGenerating answers with GPT-3.5-turbo...")
        openai_client = OpenAI(api_key=api_key)

        for ds_name in ['hotpotqa', 'msmarco', 'covidqa']:
            generation_results[ds_name] = run_llm_generation_experiment(
                datasets_dict[ds_name], ds_name, dense_model,
                openai_client, sample_size=20
            )

        with open(os.path.join(FIGURES_DIR, 'generation_results.json'), 'w') as f:
            json.dump(generation_results, f, indent=2, default=str)

        print("\n  Adherence and completeness proxies by retrieval config:")
        for ds_name, examples in generation_results.items():
            orig_adh = np.nanmean([ex.get('original_adherence_proxy', np.nan) for ex in examples])
            bm25_adh = np.nanmean([ex.get('bm25_k2_adherence_proxy', np.nan) for ex in examples])
            dense_adh = np.nanmean([ex.get('dense_k2_adherence_proxy', np.nan) for ex in examples])
            orig_comp = np.nanmean([ex.get('original_completeness_proxy', np.nan) for ex in examples])
            bm25_comp = np.nanmean([ex.get('bm25_k2_completeness_proxy', np.nan) for ex in examples])
            dense_comp = np.nanmean([ex.get('dense_k2_completeness_proxy', np.nan) for ex in examples])
            print(f"\n  {ds_name}:")
            print(f"    Adherence proxy  -- All docs: {orig_adh:.3f}, BM25 k=2: {bm25_adh:.3f}, Dense k=2: {dense_adh:.3f}")
            print(f"    Completeness proxy -- All docs: {orig_comp:.3f}, BM25 k=2: {bm25_comp:.3f}, Dense k=2: {dense_comp:.3f}")

        print("\n  Sample generated answers:")
        for ds_name, examples in generation_results.items():
            ex = examples[0]
            print(f"\n  {ds_name} - Q: {ex['question'][:80]}...")
            print(f"    Original response: {ex['original_response'][:100]}...")
            if ex.get('original_generated'):
                print(f"    Generated (all docs): {ex['original_generated'][:100]}...")
            if ex.get('bm25_k2_generated'):
                print(f"    Generated (BM25 k=2): {ex['bm25_k2_generated'][:100]}...")
            if ex.get('dense_k2_generated'):
                print(f"    Generated (Dense k=2): {ex['dense_k2_generated'][:100]}...")
    else:
        if not OPENAI_AVAILABLE:
            print("\nSkipping LLM generation: openai package not installed.")
        elif not api_key:
            print("\nSkipping LLM generation: OPENAI_API_KEY not set.")
            print("    Set it with: export OPENAI_API_KEY='your-key-here'")
        print("    Everything else still runs fine without it.")

    # Reproduce the baseline evaluator metrics from the paper (GPT-3.5, RAGAS, TruLens)
    print("\nComputing baseline evaluation metrics...")
    baseline_results = {}
    for name, ds in datasets_dict.items():
        baseline_results[name] = compute_baseline_metrics(ds)
    with open(os.path.join(FIGURES_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)

    print("\nAnalyzing TRACe metric distributions...")
    dist_df = trace_distributions(datasets_dict)
    summary_stats = dist_df.groupby(['dataset', 'metric'])['value'].describe()
    summary_stats.to_csv(os.path.join(FIGURES_DIR, 'trace_summary_stats.csv'))

    print("\nClassifying failure modes...")
    failure_data = {}
    for name, ds in datasets_dict.items():
        modes, total = classify_failure_modes(ds, name)
        failure_data[name] = (modes, total)
        print(f"  {name} (n={total}): {modes}")

    print("\nComputing cross-metric correlations...")
    corr = metric_correlations(datasets_dict)
    print(corr.to_string())

    print("\nFinding representative failure examples...")
    all_examples = {}
    for name, ds in datasets_dict.items():
        all_examples[name] = find_representative_examples(ds, name)
    with open(os.path.join(FIGURES_DIR, 'representative_examples.json'), 'w') as f:
        json.dump(all_examples, f, indent=2, default=str)

    # Generate all figures
    print("\nGenerating figures...")
    print("=" * 60)
    plot_trace_distributions(dist_df)
    plot_failure_modes(failure_data)
    plot_baseline_comparison(baseline_results)
    plot_correlation_heatmap(corr)
    plot_adherence_by_dataset(datasets_dict)
    plot_retrieval_comparison(retrieval_results)
    plot_k_sensitivity(retrieval_results)
    plot_trace_summary_table(datasets_dict)

    if generation_results:
        plot_generation_comparison(generation_results)

    print("\n" + "=" * 60)
    print("Done. All figures saved to:", FIGURES_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()