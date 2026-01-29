# Information Retrieval System

A complete **Information Retrieval and RAG pipeline** supporting both **dense** and  **sparse embeddings** , designed for recipe-based search and question answering.

## Project Structure

```
.
├── config.yaml              # Central configuration file
├── pipeline.py              # End-to-end pipeline script
├── README.md                # Project documentation
├── data/
│   ├── full_dataset.csv     # Complete dataset
│   ├── small.csv            # Small dataset for testing
│   └── processed/           # Processed chunks
├── src/
│   ├── utils.py             # Utility functions
│   ├── chunking.py          # Text chunking logic
│   ├── embedding.py         # Embedding generation
│   ├── vector_db.py         # Vector database operations
│   └── retriever.py         # Retrieval logic
└── embeddings/              # Stored embeddings and indices

```

## Quick Start

### Streamlit UI (Recommended)

Run an interactive UI for Title, Ingredient, and RAG search:

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Notes:

- Configure dataset and models via `config.yaml` (e.g., `dataset_path`, `embedding_model`).
- For RAG answers, set `GOOGLE_API_KEY` in your environment or paste it in the sidebar.

### 1. Install Dependencies

```bash
pip install pandas numpy pyyaml scikit-learn sentence-transformers faiss-cpu
```

### 2. Configure Settings

Edit `config.yaml` to customize:

* Chunk size and overlap
* Embedding type (dense or sparse)
* Model names
* Dataset and output paths

### 3. Run the Pipeline

**Build and query the system:**

```bash
python pipeline.py --mode both --query "chocolate cake recipe"
```

**Only build the system:**

```bash
python pipeline.py --mode build
```

**Only query (after building):**

```bash
python pipeline.py --mode query --query "your query here" --top-k 10
```

## Configuration File

The `config.yaml` file contains all settings:

```yaml
# Chunking settings
chunk_size: 500
chunk_overlap: 50

# Embedding settings
embedding_method: dense  # or 'sparse'
embedding_model: all-MiniLM-L6-v2
embedding_batch_size: 32
normalize_embeddings: true

# TF-IDF settings (for sparse)
tfidf_max_features: 10000
tfidf_stop_words: english

# Paths
dataset_path: data/full_dataset.csv
processed_chunks_path: data/processed/chunks.csv
vector_db_path: embeddings/storage/faiss_index
output_json_path: data/output.json

# Retrieval settings
top_k: 10
```

## Module Usage

### Individual Module Examples

#### 1. Load Configuration and Dataset

```python
from src.utils import load_config, load_dataset, prepare_documents

config = load_config()
df = load_dataset()  # Uses path from config
documents = prepare_documents(df)
```

#### 2. Chunk Documents

```python
from src.chunking import chunk_documents, save_chunks

chunks = chunk_documents(documents, config)
chunk_dicts = [
    {"chunk_id": i, "text": text}
    for i, text in enumerate(chunks)
]

save_chunks(chunk_dicts)
```

#### 3. Create Embeddings

```python
from src.embedding import embed_chunks, save_dense_embeddings
import pandas as pd

chunks_df = pd.DataFrame(chunk_dicts)
model, embeddings = embed_chunks(chunks_df)
save_dense_embeddings(embeddings)  # Uses path from config
```

#### 4. Build Vector Index

```python
from src.vector_db import build_index, save_faiss_index

index = build_index(embeddings, method="dense")
save_faiss_index(index, "embeddings/storage/faiss_index_index.faiss")
```

#### 5. Query the System

```python
from src.retriever import retrieve
from src.vector_db import load_faiss_index
from src.chunking import load_chunks

chunks_df = load_chunks()  # Uses path from config
index = load_faiss_index("embeddings/storage/faiss_index_index.faiss")

results = retrieve(
    query="chocolate cake", 
    chunks_df=chunks_df, 
    index=index,
    method="dense",
    top_k=5,
    dense_model=model
)
```

## Pipeline Workflow

The complete pipeline follows these steps:

1. **Load Configuration** - Read settings from `config.yaml`
2. **Load Dataset** - Load CSV data
3. **Prepare & Chunk** - Process documents and split into chunks
4. **Create Embeddings** - Generate dense or sparse embeddings
5. **Build Index** - Create vector search index (FAISS or sklearn)
6. **Save Artifacts** - Save chunks, embeddings, and index
7. **Query System** - Retrieve relevant chunks for queries

## Features

- **Flexible Configuration**: All settings in one YAML file
- **Multiple Embedding Methods**: Dense (Sentence Transformers) and Sparse (TF-IDF)
- **Efficient Vector Search**: FAISS for dense, sklearn for sparse
- **Modular Design**: Use individual modules or the complete pipeline
- **Easy Querying**: Simple interface for retrieval
- **Persistent Storage**: Save and load embeddings/indices

## Example Queries

```bash
# Food-related
python pipeline.py --mode query --query "spicy chicken curry"

# Ingredients-based
python pipeline.py --mode query --query "recipes with tomatoes and basil"

# Cooking method
python pipeline.py --mode query --query "baking desserts"

# Custom top-k
python pipeline.py --mode query --query "healthy salads" --top-k 10
```

## Troubleshooting

### Common Issues

1. **Config file not found**

   - Make sure `config.yaml` exists in the project root
   - Check the path if running from a different directory
2. **Dataset not found**

   - Verify the `dataset_path` in `config.yaml`
   - Ensure the CSV file exists at that location
3. **Import errors**

   - Make sure all dependencies are installed
   - Run: `pip install -r requirements.txt` (if available)
4. **FAISS installation**

   - For CPU: `pip install faiss-cpu`
   - For GPU: `pip install faiss-gpu`

## Output Files

After running the pipeline, you'll find:

- `data/processed/chunks.csv` - Processed text chunks
- `embeddings/storage/faiss_index_embeddings.npy` - Dense embeddings
- `embeddings/storage/faiss_index_index.faiss` - FAISS index
- `embeddings/storage/faiss_index_vectorizer.pkl` - TF-IDF vectorizer (sparse)
- `embeddings/storage/faiss_index_matrix.npy` - Sparse matrix (sparse)

## Next Steps

- Experiment with different embedding models
- Try both dense and sparse methods
- Adjust chunk size and overlap for better results
- Integrate with your own datasets
- Add hybrid retrieval (combine dense + sparse)

## License

This project is for educational purposes.
