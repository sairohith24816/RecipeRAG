import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Union, List
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
from sentence_transformers import SentenceTransformer
from scipy import sparse
import joblib

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir

    
def load_config(config_path= "config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        # Fallback: read raw bytes and decode with replacement for invalid bytes
        with open(config_path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        return yaml.safe_load(text) or {}


def create_sparse_embeddings(texts, max_features = 5000, stop_words = "english"):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def save_sparse_embeddings(vectorizer, matrix, path_prefix = None):
    if path_prefix is None:
        config = load_config()
        path_prefix = config["vector_db_path"]
    
    ensure_dir(os.path.dirname(path_prefix))
    joblib.dump(vectorizer, f"{path_prefix}_vectorizer.pkl")
    sparse.save_npz(f"{path_prefix}_matrix.npz", matrix)
    print(f"[INFO] Saved sparse embeddings to {path_prefix}_matrix.npz")


def load_sparse_embeddings(path_prefix = None):
    if path_prefix is None:
        config = load_config()
        path_prefix = config["vector_db_path"]
    
    vectorizer = joblib.load(f"{path_prefix}_vectorizer.pkl")
    matrix = sparse.load_npz(f"{path_prefix}_matrix.npz")
    return vectorizer, matrix



def create_dense_embeddings(texts: List[str], model_name = "all-MiniLM-L6-v2", batch_size= 32, normalize= True):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts,
                            show_progress_bar=True,
                            convert_to_numpy=True,
                            batch_size=batch_size,
                            normalize_embeddings=normalize
    )
    return model, embeddings


def save_dense_embeddings(embeddings, path_prefix= None):
    if path_prefix is None:
        config = load_config()
        path_prefix = config["vector_db_path"]
    
    ensure_dir(os.path.dirname(path_prefix))
    np.save(f"{path_prefix}_embeddings.npy", embeddings)


def load_dense_embeddings(path_prefix = None):
    if path_prefix is None:
        config = load_config()
        path_prefix = config["vector_db_path"]
    
    return np.load(f"{path_prefix}_embeddings.npy")



def embed_chunks(chunks_df,method= None,model_name= None,batch_size= None,max_features= None,stop_words= None,config_path= "config.yaml"):
    config = load_config(config_path)

    requested_method = method or config.get("embedding_method", "dense")
    method = requested_method
    model_name = model_name or config.get("embedding_model", "all-MiniLM-L6-v2")
    batch_size = batch_size or config.get("embedding_batch_size", 32)
    max_features = max_features or config.get("tfidf_max_features", 5000)
    stop_words = stop_words if stop_words is not None else config.get("tfidf_stop_words", "english")
    normalize = config.get("normalize_embeddings", True)

    texts = chunks_df["text"].tolist()

    if method == "dense":
        model, embeddings = create_dense_embeddings(texts, model_name, batch_size, normalize)
        return model, embeddings, method

    if method == "sparse":
        vectorizer, embeddings = create_sparse_embeddings(texts, max_features, stop_words)
        return vectorizer, embeddings, method

    raise ValueError(f"Invalid embedding method '{requested_method}'. Choose either 'dense' or 'sparse'.")



# if __name__ == "__main__":
#     config = load_config()
#     sample_texts = ["welcome to IR....", "IR is good!!!."]

#     # vectorizer, sparse_matrix = create_sparse_embeddings(sample_texts)
#     # print("Sparse Embeddings Shape:", sparse_matrix.shape)
#     # print("Feature Names:", vectorizer.get_feature_names_out())
#     # print(sparse_matrix.toarray()) # type: ignore

#     # save_sparse_embeddings(vectorizer, sparse_matrix)

#     # loaded_vectorizer, loaded_sparse_matrix = load_sparse_embeddings()

#     # print("Loaded Sparse Embeddings Shape:", loaded_sparse_matrix.shape)
#     # print("Loaded Feature Names:", loaded_vectorizer.get_feature_names_out())
#     # print(loaded_sparse_matrix.toarray()) # type: ignore






#     model, dense_embeddings = create_dense_embeddings(sample_texts,config.get("embedding_model"))
#     print("Dense Embeddings Shape:", dense_embeddings.shape)
#     # print(dense_embeddings)
#     print(model)

#     # save_dense_embeddings(dense_embeddings)


#     loaded_dense_embeddings = load_dense_embeddings()
#     print("Dense Embeddings Shape:", loaded_dense_embeddings.shape)
