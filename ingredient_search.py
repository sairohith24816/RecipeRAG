from __future__ import annotations
import os
import sys
from typing import List, Tuple, Set
import numpy as np
import re
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

from sentence_transformers import SentenceTransformer
from src.utils import load_dataset, combine_fields, pretty_print_recipe, load_config, ensure_dir


EMBEDDINGS_DIR = "embeddings"
ING_DENSE_EMBED_FILE = os.path.join(EMBEDDINGS_DIR, "ingredient_dense_embeddings.npy")
ING_SPARSE_VECTORIZER_FILE = os.path.join(EMBEDDINGS_DIR, "ingredient_tfidf_vectorizer.pkl")
ING_SPARSE_MATRIX_FILE = os.path.join(EMBEDDINGS_DIR, "ingredient_tfidf_matrix.npz")
MODEL_NAME_KEY = "embedding_model"


_model: SentenceTransformer | None = None
_dense_embeddings: np.ndarray | None = None
_sparse_matrix = None
_tfidf_vectorizer: TfidfVectorizer | None = None
_ingredient_vocab: Set[str] | None = None


def _load_model(model_name: str) -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def build_or_load_ingredient_embeddings(force: bool = False):
    global _ing_embeddings, _ingredient_vocab

    config = load_config()
    model_name = config.get(MODEL_NAME_KEY, "all-MiniLM-L6-v2")
    df = load_dataset()
    raw_ingredients = df["ingredients"].astype(str).tolist()

    # Normalize and ensure JSON lists are flattened when possible
    ingredients: List[str] = []
    vocab: Set[str] = set()
    for entry in raw_ingredients:
        cleaned = entry.strip()
        if not cleaned:
            continue
        # Attempt JSON parse
        try:
            import json
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                # Build vocab from individual items
                for item in parsed:
                    item_str = str(item).strip().lower()
                    if item_str:
                        vocab.add(item_str)
                # Keep original list string representation for embedding consistency
                ingredients.append(cleaned)
                continue
        except Exception:
            pass
        # Fallback: treat as plain string (may already be a comma separated list)
        ingredients.append(cleaned)
        # Tokenize plaintext for vocab enrichment
        for token in re.split(r",|;|\n|\band\b", cleaned.lower()):
            tok = token.strip()
            if tok:
                vocab.add(tok)

    _ingredient_vocab = vocab

    ensure_dir(EMBEDDINGS_DIR)
    retrain = config.get("retrain", False) or force
    model = _load_model(model_name)

    # Build or load DENSE embeddings (normalized)
    global _dense_embeddings, _sparse_matrix, _tfidf_vectorizer
    if _dense_embeddings is None:
        need_rebuild_dense = retrain or not os.path.exists(ING_DENSE_EMBED_FILE)
        if not need_rebuild_dense:
            try:
                loaded = np.load(ING_DENSE_EMBED_FILE)
                if loaded.shape[0] != len(ingredients):
                    need_rebuild_dense = True
                else:
                    _dense_embeddings = loaded
            except Exception:
                need_rebuild_dense = True
        
        if need_rebuild_dense:
            print("Building dense ingredient embeddings (normalized)...")
            embeddings = model.encode(
                ingredients,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=config.get("embedding_batch_size", 32),
                normalize_embeddings=True,
            )
            np.save(ING_DENSE_EMBED_FILE, embeddings)
            print(f"Saved dense embeddings to {ING_DENSE_EMBED_FILE}")
            _dense_embeddings = embeddings

    # Build or load SPARSE embeddings (TF-IDF)
    if _sparse_matrix is None or _tfidf_vectorizer is None:
        need_rebuild_sparse = retrain or not (os.path.exists(ING_SPARSE_VECTORIZER_FILE) and os.path.exists(ING_SPARSE_MATRIX_FILE))
        if not need_rebuild_sparse:
            try:
                loaded_vectorizer = joblib.load(ING_SPARSE_VECTORIZER_FILE)
                loaded_matrix = sparse.load_npz(ING_SPARSE_MATRIX_FILE)
                if loaded_matrix.shape[0] != len(ingredients):
                    need_rebuild_sparse = True
                else:
                    _tfidf_vectorizer = loaded_vectorizer
                    _sparse_matrix = loaded_matrix
            except Exception:
                need_rebuild_sparse = True
        
        if need_rebuild_sparse:
            print("Building sparse ingredient embeddings (TF-IDF)...")
            max_features = config.get("tfidf_max_features", 5000)
            stop_words = config.get("tfidf_stop_words", "english")
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
            tfidf_matrix = vectorizer.fit_transform(ingredients)
            joblib.dump(vectorizer, ING_SPARSE_VECTORIZER_FILE)
            sparse.save_npz(ING_SPARSE_MATRIX_FILE, tfidf_matrix)
            print(f"Saved TF-IDF vectorizer to {ING_SPARSE_VECTORIZER_FILE}")
            print(f"Saved TF-IDF matrix to {ING_SPARSE_MATRIX_FILE}")
            _tfidf_vectorizer = vectorizer
            _sparse_matrix = tfidf_matrix

    return _dense_embeddings, _sparse_matrix, model  # type: ignore


def _ensure_vocab():
    if _ingredient_vocab is None:
        # Trigger building embeddings (which also builds vocab)
        build_or_load_ingredient_embeddings()
    return _ingredient_vocab or set()


def extract_ingredients(query: str, max_matches: int = 15) -> List[str]:
    """Heuristic NER-style extraction of ingredients from a free-form query.

    Strategy:
    1. Lowercase & remove filler phrases ("i have", "what can i make", etc.).
    2. Split on punctuation, conjunctions.
    3. Fuzzy match tokens against known ingredient vocabulary from dataset.
    4. Return unique matched ingredient phrases.
    """
    vocab = _ensure_vocab()
    if not vocab:
        return []

    q = query.lower()
    # Remove common filler phrases but keep the query intact
    q = re.sub(r"\b(i\s+have|what\s+can\s+i\s+make|what\s+can\s+i\s+cook|suggest|give\s+me|recipe|recipes)\b", " ", q)
    q = re.sub(r"[^a-z0-9,;\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    
    # Split into candidate phrases by major separators
    rough_parts = [p.strip() for p in re.split(r",|;|\band\b", q) if p.strip()]
    candidates: List[str] = []
    
    for part in rough_parts:
        part = part.strip()
        if not part or len(part) < 3:
            continue
        
        # Keep the full phrase as primary candidate
        candidates.append(part)
        
        # Only add individual words if phrase is longer than 3 words
        words = [w for w in re.split(r"\s+", part) if w and len(w) > 2]
        if len(words) > 3:
            candidates.extend(words)

    # Deduplicate
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return []

    # Fuzzy match each candidate against vocab with stricter threshold
    matches: List[str] = []
    seen = set()
    
    for cand in candidates:
        if cand in seen:
            continue
            
        res = process.extract(
            cand,
            vocab,
            scorer=fuzz.token_sort_ratio,
            limit=1,
        )
        if res:
            best, score, _ = res[0]
            # Stricter threshold to avoid false matches
            if score >= 90:
                if best not in seen:
                    matches.append(best)
                    seen.add(best)

    # Return unique, limited list
    return matches[:max_matches]


def extract_ingredients_with_negation(query: str, max_matches: int = 15) -> Tuple[List[str], List[str]]:
    """Extract included and excluded ingredients using fuzzy matching and simple negation detection.

    Returns (include_list, exclude_list).
    """
    vocab = _ensure_vocab()
    if not vocab:
        return [], []

    q = query.lower()
    # Remove filler phrases
    q = re.sub(r"\b(i\s+have|what\s+can\s+i\s+make|what\s+can\s+i\s+cook|suggest|give\s+me|recipe|recipes)\b", " ", q)
    q = re.sub(r"[^a-z0-9,;\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # Split into fragments by major separators
    parts = [p.strip() for p in re.split(r",|;", q) if p.strip()]

    # Negation cues
    neg_cue = re.compile(r"\b(no|without|except|avoid|exclude|don'?t|do\s+not|not\s+with|free\s+of)\b")

    pos_candidates: List[str] = []
    neg_candidates: List[str] = []

    for part in parts:
        part = part.strip()
        if not part or len(part) < 3:
            continue
            
        has_negation = neg_cue.search(part)
        target = neg_candidates if has_negation else pos_candidates
        
        # Remove negation words from the part
        clean_part = neg_cue.sub(" ", part).strip()
        clean_part = re.sub(r"\s+", " ", clean_part)
        
        if not clean_part:
            continue
        
        # Add the full cleaned phrase
        target.append(clean_part)
        
        # Split on 'and' for multiple ingredients
        sub_parts = [sp.strip() for sp in re.split(r"\band\b", clean_part) if sp.strip()]
        if len(sub_parts) > 1:
            target.extend(sub_parts)

    # Deduplicate
    pos_candidates = list(dict.fromkeys(pos_candidates))
    neg_candidates = list(dict.fromkeys(neg_candidates))

    def _fuzzy_pick(cands: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for cand in cands:
            if cand in seen or len(cand) < 3:
                continue
            res = process.extract(cand, vocab, scorer=fuzz.token_sort_ratio, limit=1)
            if res:
                best, score, _ = res[0]
                # Stricter threshold for more accurate matches
                if score >= 90 and best not in seen:
                    out.append(best)
                    seen.add(best)
        return out[:max_matches]

    include = _fuzzy_pick(pos_candidates)
    exclude = _fuzzy_pick(neg_candidates)
    # If user only expressed negatives, it's okay to have empty include
    return include, exclude


def search_ingredients(query: str, top_k: int = 5, method: str = "dense") -> List[Tuple[int, float]]:
    """Search ingredients. method: 'dense' for semantic search or 'sparse' for TF-IDF."""
    dense_emb, sparse_mat, model = build_or_load_ingredient_embeddings()
    
    include, exclude = extract_ingredients_with_negation(query)
    effective_query = ", ".join(include) if include else query

    if method == "dense":
        if dense_emb is None:
            raise RuntimeError("Dense embeddings not initialized.")
        q_vec = model.encode(
            [effective_query],
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=1,
            normalize_embeddings=True,
        )[0]
        scores = dense_emb @ q_vec
    else:  # sparse (TF-IDF)
        if sparse_mat is None or _tfidf_vectorizer is None:
            raise RuntimeError("Sparse embeddings not initialized.")
        q_vec = _tfidf_vectorizer.transform([effective_query])
        # Compute similarity: (n_docs, n_features) @ (n_features, 1) = (n_docs, 1)
        scores_sparse = sparse_mat @ q_vec.T  # type: ignore
        scores = np.asarray(scores_sparse.todense()).flatten()
    # Sort all indices by score descending
    sorted_idx = np.argsort(-scores)

    # Filtering helper: check if a row contains any excluded ingredient
    def _row_has_excluded(row_txt: str, excluded: List[str]) -> bool:
        # Parse ingredients if in JSON format
        ingredients_list = []
        try:
            import json
            parsed = json.loads(row_txt)
            if isinstance(parsed, list):
                ingredients_list = [str(item).lower().strip() for item in parsed]
        except:
            # Fallback to treating as plain text
            ingredients_list = [item.strip().lower() for item in re.split(r',|;|\n', row_txt) if item.strip()]
        
        # Check each excluded ingredient against the list
        for ex in excluded:
            ex_l = ex.lower().strip()
            if not ex_l:
                continue
            
            # Check for exact or partial matches in ingredient list
            for ing in ingredients_list:
                # Check if excluded ingredient is in this ingredient
                # Use fuzzy matching for better accuracy
                if ex_l in ing or ing in ex_l:
                    return True
                # Also check with fuzzy matching for close matches
                score = fuzz.partial_ratio(ex_l, ing)
                if score >= 85:  # High threshold for exclusion matching
                    return True
        return False

    # Build filtered top_k considering exclusions
    df = load_dataset()
    results: List[Tuple[int, float]] = []
    for i in sorted_idx:
        if len(results) >= top_k:
            break
        row = df.iloc[int(i)]
        # Use combined ingredients text for substring check
        raw_ing = str(row.get("ingredients", ""))
        if exclude and _row_has_excluded(raw_ing, exclude):
            continue
        results.append((int(i), float(scores[int(i)])))

    return results


def print_results(query: str, top_k: int = 5) -> None:
    df = load_dataset()
    include, exclude = extract_ingredients_with_negation(query)
    if include:
        print(f"[INFO] Include ingredients: {include}")
    if exclude:
        print(f"[INFO] Exclude ingredients: {exclude}")
    matches = search_ingredients(query, top_k=top_k)
    if not matches:
        print("No matches found.")
        return
    for rank, (row_idx, score) in enumerate(matches, start=1):
        row = df.iloc[row_idx]
        text_block = combine_fields(row)
        pretty_print_recipe(text_block, score=score, rank=rank)


def interactive_loop():
    print("Ingredient search ready. Type a query (blank to exit).")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        print_results(query)


def main(argv: List[str]):
    build_or_load_ingredient_embeddings()
    if len(argv) > 1:
        query = " ".join(argv[1:]).strip()
        print_results(query)
    else:
        interactive_loop()


if __name__ == "__main__":
    main(sys.argv)
