import os
import pickle
import numpy as np
from typing import Tuple, List, Union
from sklearn.neighbors import NearestNeighbors
from src.utils import ensure_dir

import faiss


# DENSE VECTOR INDEX (FAISS)
def build_faiss_index(embeddings):      # shape=no. of items, embed_dim
    embeddings = np.ascontiguousarray(embeddings.astype('float32'))
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product ≈ cosine similarity after normalization
    index.add(embeddings)  # type: ignore
    return index


def save_faiss_index(index, path):
    ensure_dir(os.path.dirname(path))
    faiss.write_index(index, path)
    print(f"Saved FAISS index to {path}")


def load_faiss_index(path):
    return faiss.read_index(path)


def faiss_search(index, query_vecs, top_k = 5):
    faiss.normalize_L2(query_vecs)
    scores, indices = index.search(query_vecs.astype('float32'), top_k)
    return scores, indices


# SPARSE VECTOR INDEX (TF-IDF) 
def build_sparse_index(embeddings, metric = "cosine"):
    nn = NearestNeighbors(metric=metric)
    nn.fit(embeddings)
    return nn


def save_sparse_index(nn, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(nn, f)

def load_sparse_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sparse_search(index, query_vecs, top_k: int = 5):
    distances, indices = index.kneighbors(query_vecs, n_neighbors=top_k)
    scores = 1 - distances  # convert distance → similarity
    return scores, indices


# Wrapper
def build_index(embeddings, method="dense"):
    if method == "dense":
        return build_faiss_index(embeddings)
    elif method == "sparse":
        return build_sparse_index(embeddings)
    else:
        raise ValueError("Invalid method. Choose either 'dense' or 'sparse'.")


def search_index(index, query_vecs, method = "dense", top_k = 5):
    if method == "dense":
        return faiss_search(index, query_vecs, top_k)
    elif method == "sparse":
        return sparse_search(index, query_vecs, top_k)
    else:
        raise ValueError("Invalid method. Choose either 'dense' or 'sparse'.")


# if __name__ == "__main__":
#     embeddings = np.random.rand(100, 768).astype('float32')

#     # index = build_faiss_index(embeddings)
#     # print(index.is_trained)
#     # query = np.random.rand(1, 768).astype('float32')
#     # faiss.normalize_L2(query)
#     # distances, indices = index.search(query, k=5)
#     # print(indices)  # indices of the 5 most similar embeddings
#     # print(distances) 

#     # save_faiss_index(index, "test_faiss.index")
#     # print(load_faiss_index("test_faiss.index").search(query, k=5))

#     # nn=build_sparse_index(embeddings)
#     # print(nn)



#     nn = build_sparse_index(embeddings, metric="cosine")
#     query = np.array([embeddings[0] + 0.01]).astype('float32')
#     distances, indices = nn.kneighbors(query, n_neighbors=3)
#     print("Indices of nearest neighbors:", indices)
#     print("Cosine distances:", distances)

#     save_sparse_index(nn, "test_sparse.index")
#     print(load_sparse_index("test_sparse.index").kneighbors(query, n_neighbors=3))


