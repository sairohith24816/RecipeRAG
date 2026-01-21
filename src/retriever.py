import numpy as np
import pandas as pd
from typing import List, Tuple, Union
import yaml

from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from src.embedding import embed_chunks, create_dense_embeddings, create_sparse_embeddings, load_sparse_embeddings
from src.vector_db import build_index, search_index, load_faiss_index, load_sparse_index
from src.utils import load_dataset, prepare_documents, pretty_print_recipe

import faiss



def load_config(config_path= "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def embed_query(query,method= "dense",model_name = "all-MiniLM-L6-v2",dense_model=None,vectorizer=None):
    if method == "dense":
        if dense_model is None:
            dense_model, embeddings = create_dense_embeddings([query], model_name)
            return embeddings[0]
        else:
            return dense_model.encode([query], convert_to_numpy=True)[0]
    elif method == "sparse":
        if vectorizer is None:
            raise ValueError("For sparse method, provide a fitted vectorizer.")
        return vectorizer.transform([query]).toarray()[0]
    else:
        raise ValueError("Invalid method, choose 'dense' or 'sparse'.")


def retrieve(query,chunks_df,index,method = "dense",top_k = 5,dense_model=None,vectorizer=None):
    query_emb = embed_query(query, method, dense_model=dense_model, vectorizer=vectorizer)
    query_emb = np.array([query_emb])  # shape (1, dim)

    scores, indices = search_index(index, query_emb, method=method, top_k=top_k)

    # Flatten results (since 1 query)
    scores = scores.flatten()
    indices = indices.flatten()

    # Filter out invalid indices (FAISS returns -1 for missing)
    valid_mask = indices >= 0
    filtered_scores = scores[valid_mask]
    filtered_indices = indices[valid_mask]

    results = chunks_df.iloc[filtered_indices].copy()
    results["score"] = filtered_scores

    return results.reset_index(drop=True)


if __name__ == "__main__":
    # Load configuration and optional metadata to determine method
    config = load_config()
    prefix = config["vector_db_path"]
    meta_path = prefix + "_meta.yaml"

    embedding_method = config.get("embedding_method", "dense")
    try:
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f) or {}
            embedding_method = meta.get("embedding_method", embedding_method)
    except FileNotFoundError:
        pass

    # Load documents (no chunking)
    df = load_dataset()
    documents = prepare_documents(df)
    docs_df = pd.DataFrame({
        'doc_id': range(len(documents)),
        'text': documents,
        'source': 'recipe_dataset'
    })

    # Load index and query encoder (no corpus recomputation)
    if embedding_method == "dense":
        index = load_faiss_index(prefix + "_index.faiss")
        dense_model = SentenceTransformer(config.get("embedding_model", "all-MiniLM-L6-v2"))
        vectorizer = None
    else:
        index = load_sparse_index(prefix + "_index.pkl")
        vectorizer, _ = load_sparse_embeddings()
        dense_model = None

    # Test retrieval
    query = "chicken cutlet with onions and egg"
    results = retrieve(
        query,
        docs_df,
        index,
        method=embedding_method,
        top_k=5,
        dense_model=dense_model,
        vectorizer=vectorizer,
    )
    
    print("\n" + "="*60)
    print("🔍 SEARCH RESULTS")
    print("="*60)
    
    for idx, row in results.iterrows():
        pretty_print_recipe(row['text'], score=row['score'], rank=idx + 1)
