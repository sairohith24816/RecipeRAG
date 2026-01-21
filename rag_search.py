from __future__ import annotations
import os
import sys
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

from sentence_transformers import SentenceTransformer
from src.utils import load_dataset, combine_fields, load_config, ensure_dir
from src.generation import generate_answer, rank_recipes


EMBEDDINGS_DIR = "embeddings"
FULL_DENSE_EMBED_FILE = os.path.join(EMBEDDINGS_DIR, "full_recipe_dense_embeddings.npy")
FULL_SPARSE_VECTORIZER_FILE = os.path.join(EMBEDDINGS_DIR, "full_recipe_tfidf_vectorizer.pkl")
FULL_SPARSE_MATRIX_FILE = os.path.join(EMBEDDINGS_DIR, "full_recipe_tfidf_matrix.npz")
MODEL_NAME_KEY = "embedding_model"


_model: SentenceTransformer | None = None
_dense_embeddings: np.ndarray | None = None
_sparse_matrix = None
_tfidf_vectorizer: TfidfVectorizer | None = None


def _load_model(model_name: str) -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def _build_corpus_texts(df) -> List[str]:
    # Use the existing structured combiner so downstream pretty printing / LLM sees fields.
    return [combine_fields(row) for _, row in df.iterrows()]


def build_or_load_full_embeddings(force: bool = False):
    global _dense_embeddings, _sparse_matrix, _tfidf_vectorizer

    config = load_config()
    model_name = config.get(MODEL_NAME_KEY, "all-MiniLM-L6-v2")
    df = load_dataset()
    texts = _build_corpus_texts(df)

    ensure_dir(EMBEDDINGS_DIR)
    retrain = config.get("retrain", False) or force
    model = _load_model(model_name)

    # Build or load DENSE embeddings (normalized)
    if _dense_embeddings is None:
        need_rebuild_dense = retrain or not os.path.exists(FULL_DENSE_EMBED_FILE)
        if not need_rebuild_dense:
            try:
                loaded = np.load(FULL_DENSE_EMBED_FILE)
                if loaded.shape[0] != len(texts):
                    need_rebuild_dense = True
                else:
                    _dense_embeddings = loaded
            except Exception:
                need_rebuild_dense = True
        
        if need_rebuild_dense:
            print("Building dense full recipe embeddings (normalized)...")
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=config.get("embedding_batch_size", 32),
                normalize_embeddings=True,
            )
            np.save(FULL_DENSE_EMBED_FILE, embeddings)
            print(f"Saved dense embeddings to {FULL_DENSE_EMBED_FILE}")
            _dense_embeddings = embeddings

    # Build or load SPARSE embeddings (TF-IDF)
    if _sparse_matrix is None or _tfidf_vectorizer is None:
        need_rebuild_sparse = retrain or not (os.path.exists(FULL_SPARSE_VECTORIZER_FILE) and os.path.exists(FULL_SPARSE_MATRIX_FILE))
        if not need_rebuild_sparse:
            try:
                loaded_vectorizer = joblib.load(FULL_SPARSE_VECTORIZER_FILE)
                loaded_matrix = sparse.load_npz(FULL_SPARSE_MATRIX_FILE)
                if loaded_matrix.shape[0] != len(texts):
                    need_rebuild_sparse = True
                else:
                    _tfidf_vectorizer = loaded_vectorizer
                    _sparse_matrix = loaded_matrix
            except Exception:
                need_rebuild_sparse = True
        
        if need_rebuild_sparse:
            print("Building sparse full recipe embeddings (TF-IDF)...")
            max_features = config.get("tfidf_max_features", 5000)
            stop_words = config.get("tfidf_stop_words", "english")
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
            tfidf_matrix = vectorizer.fit_transform(texts)
            joblib.dump(vectorizer, FULL_SPARSE_VECTORIZER_FILE)
            sparse.save_npz(FULL_SPARSE_MATRIX_FILE, tfidf_matrix)
            print(f"Saved TF-IDF vectorizer to {FULL_SPARSE_VECTORIZER_FILE}")
            print(f"Saved TF-IDF matrix to {FULL_SPARSE_MATRIX_FILE}")
            _tfidf_vectorizer = vectorizer
            _sparse_matrix = tfidf_matrix

    return _dense_embeddings, _sparse_matrix, model  # type: ignore


def search_full(query: str, top_k: int = 5, method: str = "dense") -> List[Tuple[int, float]]:
    """Search full recipes. method: 'dense' for semantic search or 'sparse' for TF-IDF."""
    dense_emb, sparse_mat, model = build_or_load_full_embeddings()
    
    if method == "dense":
        if dense_emb is None:
            raise RuntimeError("Dense embeddings not initialized.")
        q_vec = model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=1,
            normalize_embeddings=True,
        )[0]
        scores = dense_emb @ q_vec
    else:  # sparse (TF-IDF)
        if sparse_mat is None or _tfidf_vectorizer is None:
            raise RuntimeError("Sparse embeddings not initialized.")
        q_vec = _tfidf_vectorizer.transform([query])
        # Compute similarity: (n_docs, n_features) @ (n_features, 1) = (n_docs, 1)
        scores_sparse = sparse_mat @ q_vec.T  # type: ignore
        scores = np.asarray(scores_sparse.todense()).flatten()
    
    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def print_titles(df, indices_with_scores: List[Tuple[int, float]]):
    print("Top 5 candidate recipes:")
    for rank, (i, s) in enumerate(indices_with_scores, start=1):
        title = str(df.iloc[i]["title"]).strip()
        print(f"  {rank}. {title} (Similarity: {s:.4f})")


def ask_llm(query: str, df, indices_with_scores: List[Tuple[int, float]]) -> str:
    config = load_config()
    contexts: List[str] = []
    for i, _ in indices_with_scores:
        row = df.iloc[i]
        # Provide full combined text (title+ingredients+directions)
        contexts.append(combine_fields(row))
    
    print(f"[DEBUG] Built {len(contexts)} context snippets for LLM.")
    
    try:
        # Step 1: Ask LLM to rank the recipes
        print("[INFO] Asking LLM to rank the retrieved recipes...")
        best_idx = rank_recipes(query, contexts, config)
        best_recipe = contexts[best_idx]
        best_recipe_num = indices_with_scores[best_idx][0]
        best_title = str(df.iloc[best_recipe_num]["title"]).strip()
        
        print(f"[INFO] LLM selected: '{best_title}' (Recipe #{best_recipe_num + 1})")
        
        # Step 2: Generate refined answer using only the top-ranked recipe
        answer = generate_answer(query, [best_recipe], config)
        if not answer or not answer.strip():
            return (
                "No response from LLM. Possible causes:\n"
                "- Model name invalid or unavailable (try gemini-1.5-flash-latest)\n"
                "- Prompt too large (reduce rag_max_context_chars in config.yaml)\n"
                "- Safety/policy block\n"
                "- API quota or auth issue"
            )
        return answer
    except Exception as e:
        # Surface a concise message if LLM config/API isn't set up.
        return (
            "LLM generation failed. Ensure GOOGLE_API_KEY is set and configuration is valid.\n"
            f"Error: {e}"
        )


def run_query(query: str, top_k: int = 5, method: str = "dense") -> None:
    df = load_dataset()
    matches = search_full(query, top_k=top_k, method=method)
    if not matches:
        print("No matches found.")
        return
    print_titles(df, matches)
    print()
    print("Refined answer:")
    print(ask_llm(query, df, matches))


def interactive_loop():
    print("Full recipe RAG ready. Type a query (blank to exit).")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        run_query(query)


def main(argv: List[str]):
    # Pre-build or load embeddings for fast first response
    build_or_load_full_embeddings()
    if len(argv) > 1:
        query = " ".join(argv[1:]).strip()
        run_query(query)
    else:
        interactive_loop()


if __name__ == "__main__":
    main(sys.argv)
