import os
import yaml
import pandas as pd
import json
import re


def load_config(config_path = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    # Force UTF-8 to support extended punctuation in YAML on Windows
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# print(load_config('config.yaml'))

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def load_dataset():
    config = load_config()
    csv_path = config.get("dataset_path", "data/small.csv")
    max_docs = config.get("max_docs", None)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at path: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # drop unused column
    df = df.dropna(subset=["title", "ingredients", "directions"])  # ensure valid rows

    # Apply document cap if configured (deterministic: first N rows)
    if isinstance(max_docs, int) and max_docs > 0:
        df = df.head(max_docs)
    
    return df



def preprocess_text(text: str) -> str:
    # text = str(text).lower()
    # text = re.sub(r"[^a-z0-9\s.,:;!?'-]", " ", text)
    # text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_fields(row: pd.Series) -> str:
    """Combine recipe fields with structured formatting for better readability."""
    title = preprocess_text(row.get('title', ''))
    ingredients = preprocess_text(row.get('ingredients', ''))
    directions = preprocess_text(row.get('directions', ''))
    
    combined = f"title: {title}\ningredients: {ingredients}\ndirections: {directions}"
    return combined


def prepare_documents(df: pd.DataFrame) -> list:
    return [combine_fields(row) for _, row in df.iterrows()]


def pretty_print_recipe(text: str, score = None, rank = None):
    """Pretty print a recipe with structured formatting."""
    import textwrap
    
    lines = text.split('\n')
    
    # Parse structured fields dynamically
    fields = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key.strip()] = value.strip()
    
    title = fields.get('title', '')
    ingredients = fields.get('ingredients', '')
    directions = fields.get('directions', '')
    
    # Print formatted output
    if rank:
        print(f"\n{'='*60}")
        print(f"RESULT #{rank}")
        print(f"{'='*60}")
    
    # Title with similarity score
    print(f"\nTITLE: {title.title()}", end="")
    if score is not None:
        print(f"  (Similarity: {score:.4f})")
    else:
        print()
    
    # Parse and display ingredients
    print(f"\nINGREDIENTS:")
    try:
        ingredient_items = json.loads(ingredients)
        if isinstance(ingredient_items, list):
            for i, item in enumerate(ingredient_items, 1):
                print(f"   {i}. {item}")
        else:
            print(f"   {ingredients}")
    except (json.JSONDecodeError, ValueError):
        print(f"   {ingredients}")
    
    # Parse and display directions
    print(f"\nDIRECTIONS:")
    try:
        direction_steps = json.loads(directions)
        if isinstance(direction_steps, list):
            for i, step in enumerate(direction_steps, 1):
                # Wrap long steps to 70 characters
                wrapped = textwrap.fill(step, width=70, initial_indent='   ', 
                                       subsequent_indent='      ')
                print(f"   {i}. {step}")
        else:
            wrapped = textwrap.fill(directions, width=70, initial_indent='   ',
                                   subsequent_indent='      ')
            print(wrapped)
    except (json.JSONDecodeError, ValueError):
        wrapped = textwrap.fill(directions, width=70, initial_indent='   ',
                               subsequent_indent='      ')
        print(wrapped)
    
    print(f"\n{'-'*60}")


# print(prepare_documents(load_dataset())[0])

def load_json(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


























