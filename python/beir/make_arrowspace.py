#!/usr/bin/env python3
"""
SciFact → embeddings → ArrowSpace manifold (build_and_store) → eigenmaps search tests.

Install:
  pip install beir sentence-transformers numpy tqdm scikit-learn arrowspace

Run:
  python scifact_arrowspace_eigenmaps_trial.py --workdir ./runs/scifact_trial

Notes:
- Downloads SciFact from BEIR mirror using the same pattern as your BEIR scripts. [file:24]
- Builds embeddings with SentenceTransformer.
- Builds and stores ArrowSpace manifold using pyarrowspace (arrowspace pip package). [file:42]
- Runs retrieval metrics (MRR@10, Recall@10, nDCG@10) comparing:
    (a) pure cosine (tau=1.0)
    (b) eigenmaps / spectral blend (tau < 1.0)
"""

import os
import json
import time
import argparse
import pathlib
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from sentence_transformers import SentenceTransformer

from sklearn.metrics import ndcg_score

# pyarrowspace
from arrowspace import ArrowSpaceBuilder, set_debug


# ----------------------------
# BEIR download + load
# ----------------------------
def download_beir_dataset(dataset: str, out_dir: str) -> str:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, out_dir)
    return data_path


def load_beir_split(data_path: str, split: str):
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    return corpus, queries, qrels


# ----------------------------
# Embeddings
# ----------------------------
def build_text_for_doc(doc: Dict) -> str:
    """
    SciFact docs typically have title + text; keep a stable concatenation.
    """
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine-friendly baseline
    )
    # ArrowSpace scales embeddings to keep eps/sigma meaningful.
    # Keep it explicit and configurable:
    return emb.astype(np.float64) * 12.0  


def encode_texts_cached(
    model: SentenceTransformer,
    texts: List[str],
    cache_path: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Loads embeddings from cache_path if present and shape matches.
    Otherwise computes, saves, and returns.
    """
    if os.path.exists(cache_path):
        X = np.load(cache_path)
        if X.shape[0] == len(texts):
            print(f"Loaded cached embeddings: {cache_path} shape={X.shape}")
            return X
        print(
            f"Cache mismatch for {cache_path}: "
            f"cache has {X.shape[0]} rows but texts={len(texts)}. Regenerating..."
        )

    X = encode_texts(model, texts, batch_size=batch_size)
    np.save(cache_path, X)
    print(f"Saved embeddings: {cache_path} shape={X.shape}")
    return X



# ----------------------------
# Ranking + metrics
# ----------------------------
def mrr_at_k(ranked_doc_ids: List[str], rel_set: set, k: int) -> float:
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_doc_ids: List[str], rel_set: set, k: int) -> float:
    if not rel_set:
        return 0.0
    hit = sum(1 for d in ranked_doc_ids[:k] if d in rel_set)
    return hit / float(len(rel_set))


def ndcg_at_k(ranked_doc_ids: List[str], rel_map: Dict[str, int], k: int) -> float:
    """
    Uses sklearn.ndcg_score with relevance labels aligned to predicted ranking.
    """
    if not rel_map:
        return 0.0
    # relevance in predicted order
    y_true = np.array([[rel_map.get(d, 0) for d in ranked_doc_ids[:k]]], dtype=np.float64)
    # use monotonically decreasing scores to represent ranking positions
    y_score = np.array([[1.0 / (i + 1) for i in range(min(k, len(ranked_doc_ids)))]], dtype=np.float64)
    if y_true.sum() == 0:
        return 0.0
    return float(ndcg_score(y_true, y_score, k=min(k, y_true.shape[1])))


def evaluate_queries(
    aspace,
    gl,
    query_ids: List[str],
    query_emb: np.ndarray,
    qrels: Dict[str, Dict[str, int]],
    doc_ids_order: List[str],
    tau: float,
    k_eval: int = 10,
) -> Dict[str, float]:
    """
    Runs ArrowSpace search for each query embedding and computes avg MRR/Recall/nDCG.
    """
    mrrs, recalls, ndcgs = [], [], []
    n = len(query_ids)

    for i, qid in enumerate(tqdm(query_ids, desc=f"Searching tau={tau}")):
        print(f"Query embeddings -> ", query_emb[i])
        res = aspace.search(query_emb[i], gl, tau=tau)  # returns [(index, score), ...] [file:42]

        # map ArrowSpace row indices → doc ids
        ranked_doc_ids = [doc_ids_order[idx] for idx, _ in res]

        rels = qrels.get(qid, {})
        rel_set = {doc_id for doc_id, rel in rels.items() if rel > 0}
        # graded relevance map (keep BEIR label if provided)
        rel_map = {doc_id: int(rel) for doc_id, rel in rels.items() if rel > 0}

        mrrs.append(mrr_at_k(ranked_doc_ids, rel_set, k_eval))
        recalls.append(recall_at_k(ranked_doc_ids, rel_set, k_eval))
        ndcgs.append(ndcg_at_k(ranked_doc_ids, rel_map, k_eval))

    return {
        "n_queries": n,
        "MRR@10": float(np.mean(mrrs)) if mrrs else 0.0,
        "Recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "nDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


# ----------------------------
# Main pipeline
# ----------------------------
def main(workdir: str, model_name: str, tau_cosine: float, tau_eigen: float):
    set_debug(True)

    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    dataset = "scifact"
    datasets_dir = os.path.join(workdir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    data_path = os.path.join(datasets_dir, dataset)

    # If already downloaded, skip download_and_unzip.
    # Otherwise download from BEIR mirror. [file:24]
    if not os.path.isdir(data_path):
        print(f"[1] Downloading BEIR dataset: {dataset}")
        data_path = download_beir_dataset(dataset, datasets_dir)
        print(f"    data_path={data_path}")
    else:
        print(f"[1] Using existing dataset at: {data_path}")

    print("[1b] Loading splits")
    corpus, queries, qrels = load_beir_split(data_path, split="test")

    doc_ids = list(corpus.keys())
    doc_texts = [build_text_for_doc(corpus[doc_id]) for doc_id in doc_ids]

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"    docs={len(doc_ids)} queries={len(query_ids)} qrels={len(qrels)}")

    print("[2] Creating / loading embeddings")
    model = SentenceTransformer(model_name)

    doc_cache = os.path.join(workdir, "scifact_doc_embeddings.npy")
    query_cache = os.path.join(workdir, "scifact_query_embeddings.npy")

    X_docs = encode_texts_cached(model, doc_texts, cache_path=doc_cache, batch_size=64)
    X_q = encode_texts_cached(model, query_texts, cache_path=query_cache, batch_size=64)

    emb_path = os.path.join(workdir, "scifact_doc_embeddings.npy")
    np.save(emb_path, X_docs)
    print(f"    Saved embeddings: {emb_path} shape={X_docs.shape}")
    print("Sample: ", X_docs[0])

    print("[3] Build + store ArrowSpace manifold (eigenmaps)")
    # Start from reasonably safe graph params; SciFact is small so higher k is fine.
    graph_params = {
        "eps": 0.95,    # Start near 0.25 and grow according to scale
        "k": 25,        # higher to improve connectivity
        "topk": 15,
        "p": 1.5,       # Kernel sharpness
        "sigma": None
    }

    t0 = time.perf_counter()

    aspace, gl = ArrowSpaceBuilder.build_and_store(
        graph_params=graph_params,
        items=X_docs,
    )

    dt = time.perf_counter() - t0
    print(f"    Build+store done in {dt:.2f}s -> storage/ directory")

    print("[4] Tests + metrics: eigenmaps search behavior")

    # Test 1: basic sanity: search returns non-empty and indices in range
    sample_res = aspace.search(X_q[0], gl, tau=tau_cosine)
    assert len(sample_res) > 0, "Search returned empty result list"
    assert 0 <= sample_res[0][0] < len(doc_ids), "Top result index out of range"

    # Test 2: cosine vs eigen differ (usually not identical if tau differs)
    res_cos = aspace.search(X_q[0], gl, tau=tau_cosine)
    res_eig = aspace.search(X_q[0], gl, tau=tau_eigen)
    top10_cos = [i for i, _ in res_cos[:10]]
    top10_eig = [i for i, _ in res_eig[:10]]
    jaccard = len(set(top10_cos) & set(top10_eig)) / float(len(set(top10_cos) | set(top10_eig)))
    print(f"    Sanity: Top10 Jaccard(cos,eigen)={jaccard:.3f}")

    # IR metrics on the full query set
    metrics_cos = evaluate_queries(
        aspace=aspace,
        gl=gl,
        query_ids=query_ids,
        query_emb=X_q,
        qrels=qrels,
        doc_ids_order=doc_ids,
        tau=tau_cosine,
        k_eval=10,
    )
    metrics_eig = evaluate_queries(
        aspace=aspace,
        gl=gl,
        query_ids=query_ids,
        query_emb=X_q,
        qrels=qrels,
        doc_ids_order=doc_ids,
        tau=tau_eigen,
        k_eval=10,
    )

    print("\n=== Metrics (test split) ===")
    print(json.dumps({"tau=1.0(cosine)": metrics_cos, f"tau={tau_eigen}(eigen)": metrics_eig}, indent=2))

    # Persist metrics
    with open(os.path.join(workdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "model": model_name,
                "graph_params": graph_params,
                "tau_cosine": tau_cosine,
                "tau_eigen": tau_eigen,
                "metrics_cosine": metrics_cos,
                "metrics_eigen": metrics_eig,
                "top10_jaccard_first_query": jaccard,
            },
            f,
            indent=2,
        )

    print(f"\nWrote {os.path.join(workdir, 'metrics.json')}")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="./runs/scifact_trial", help="Output working directory.")
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or path.",
    )
    ap.add_argument("--tau_cosine", type=float, default=1.0, help="tau=1.0 means pure cosine.")
    ap.add_argument(
        "--tau_eigen",
        type=float,
        default=0.52,
        help="Spectral blend tau (smaller => more spectral). Matches your CVE-style usage.",
    )
    args = ap.parse_args()

    try:
        main(
            workdir=args.workdir,
            model_name=args.model,
            tau_cosine=args.tau_cosine,
            tau_eigen=args.tau_eigen,
        )
    except KeyboardInterrupt:
        import sys
        sys.exit(0)

