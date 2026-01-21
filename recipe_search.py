from __future__ import annotations

import os
import sys
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

from src.utils import load_dataset, combine_fields, pretty_print_recipe, load_config, ensure_dir
from sentence_transformers import SentenceTransformer


EMBEDDINGS_DIR = "embeddings"
TITLE_DENSE_EMBED_FILE = os.path.join(EMBEDDINGS_DIR, "title_dense_embeddings.npy")
TITLE_SPARSE_VECTORIZER_FILE = os.path.join(EMBEDDINGS_DIR, "title_tfidf_vectorizer.pkl")
TITLE_SPARSE_MATRIX_FILE = os.path.join(EMBEDDINGS_DIR, "title_tfidf_matrix.npz")
MODEL_NAME_KEY = "embedding_model"  # reuse existing config key


_model: SentenceTransformer | None = None
_dense_embeddings: np.ndarray | None = None
_sparse_matrix = None
_tfidf_vectorizer: TfidfVectorizer | None = None
_titles: List[str] | None = None


def _load_model(model_name: str) -> SentenceTransformer:
	global _model
	if _model is None:
		_model = SentenceTransformer(model_name)
	return _model


def build_or_load_title_embeddings(force = False):
	global _dense_embeddings, _sparse_matrix, _tfidf_vectorizer, _titles

	config = load_config()
	model_name = config.get(MODEL_NAME_KEY, "all-MiniLM-L6-v2")
	retrain = config.get("retrain", False) or force
	df = load_dataset()
	titles = df["title"].astype(str).tolist()

	# Ensure directory
	ensure_dir(EMBEDDINGS_DIR)
	model = _load_model(model_name)

	# Build or load DENSE embeddings (normalized)
	if _dense_embeddings is None:
		need_rebuild_dense = retrain or not os.path.exists(TITLE_DENSE_EMBED_FILE)
		if not need_rebuild_dense:
			try:
				loaded = np.load(TITLE_DENSE_EMBED_FILE)
				if loaded.shape[0] != len(titles):
					need_rebuild_dense = True
				else:
					_dense_embeddings = loaded
					_titles = titles
			except Exception:
				need_rebuild_dense = True
		
		if need_rebuild_dense:
			print("Building dense title embeddings (normalized)...")
			embeddings = model.encode(
				titles,
				show_progress_bar=True,
				convert_to_numpy=True,
				batch_size=config.get("embedding_batch_size", 32),
				normalize_embeddings=True,
			)
			np.save(TITLE_DENSE_EMBED_FILE, embeddings)
			print(f"Saved dense embeddings to {TITLE_DENSE_EMBED_FILE}")
			_dense_embeddings = embeddings
			_titles = titles

	# Build or load SPARSE embeddings (TF-IDF)
	if _sparse_matrix is None or _tfidf_vectorizer is None:
		need_rebuild_sparse = retrain or not (os.path.exists(TITLE_SPARSE_VECTORIZER_FILE) and os.path.exists(TITLE_SPARSE_MATRIX_FILE))
		if not need_rebuild_sparse:
			try:
				loaded_vectorizer = joblib.load(TITLE_SPARSE_VECTORIZER_FILE)
				loaded_matrix = sparse.load_npz(TITLE_SPARSE_MATRIX_FILE)
				if loaded_matrix.shape[0] != len(titles):
					need_rebuild_sparse = True
				else:
					_tfidf_vectorizer = loaded_vectorizer
					_sparse_matrix = loaded_matrix
					_titles = titles
			except Exception:
				need_rebuild_sparse = True
		
		if need_rebuild_sparse:
			print("Building sparse title embeddings (TF-IDF)...")
			max_features = config.get("tfidf_max_features", 5000)
			stop_words = config.get("tfidf_stop_words", "english")
			vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
			tfidf_matrix = vectorizer.fit_transform(titles)
			joblib.dump(vectorizer, TITLE_SPARSE_VECTORIZER_FILE)
			sparse.save_npz(TITLE_SPARSE_MATRIX_FILE, tfidf_matrix)
			print(f"Saved TF-IDF vectorizer to {TITLE_SPARSE_VECTORIZER_FILE}")
			print(f"Saved TF-IDF matrix to {TITLE_SPARSE_MATRIX_FILE}")
			_tfidf_vectorizer = vectorizer
			_sparse_matrix = tfidf_matrix
			_titles = titles

	return _dense_embeddings, _sparse_matrix, _titles, model  # type: ignore


def search_titles(query: str, top_k: int = 5, method: str = "dense") -> List[Tuple[int, float]]:
	"""Return list of (row_index, similarity) for top_k matches.
	method: 'dense' for semantic search or 'sparse' for TF-IDF search.
	"""
	dense_emb, sparse_mat, titles, model = build_or_load_title_embeddings()
	
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


def print_results(query: str, top_k: int = 5) -> None:
	df = load_dataset()
	matches = search_titles(query, top_k=top_k)
	if not matches:
		print("No matches found.")
		return
	for rank, (row_idx, score) in enumerate(matches, start=1):
		row = df.iloc[row_idx]
		text_block = combine_fields(row)
		pretty_print_recipe(text_block, score=score, rank=rank)


def interactive_loop():
	print("Recipe title search ready. Type a query (blank to exit).")
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
	build_or_load_title_embeddings()
	if len(argv) > 1:
		query = " ".join(argv[1:]).strip()
		print_results(query)
	else:
		interactive_loop()


if __name__ == "__main__":
	main(sys.argv)

