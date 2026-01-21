from __future__ import annotations

import os
import json
from typing import List, Tuple, Any

import streamlit as st

from src.utils import load_dataset, combine_fields, load_config
from recipe_search import (
    search_titles,
    build_or_load_title_embeddings,
)
from ingredient_search import (
    search_ingredients,
    build_or_load_ingredient_embeddings,
    extract_ingredients_with_negation,
)
from rag_search import (
    search_full,
    build_or_load_full_embeddings,
    ask_llm,
)


st.set_page_config(page_title="Recipe Search", layout="wide")


def _parse_combined_text(text: str) -> dict[str, Any]:
    """Parse the combined recipe text produced by combine_fields.

    The original parser only captured the first line of multi-line fields
    (especially directions). This version uses a regex to capture each
    labeled block (title / ingredients / directions) including embedded
    newlines. If ingredients or directions parse as JSON lists they are
    returned as lists; otherwise multi-line directions are split into
    individual steps (non-empty lines).
    """
    import re

    # Initialize with empty defaults
    out: dict[str, Any] = {"title": "", "ingredients": "", "directions": ""}

    pattern = re.compile(r"(title|ingredients|directions):\s*(.*?)(?=\n(?:title|ingredients|directions):|$)", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        key = match.group(1).lower()
        value = match.group(2).strip()
        out[key] = value

    # Attempt JSON parsing for list-like fields
    for key in ("ingredients", "directions"):
        raw = out.get(key, "")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                out[key] = parsed
                continue
        except Exception:
            pass
        # If not JSON list and contains multiple lines, treat each line as a step
        if "\n" in raw:
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            if lines:
                out[key] = lines
    return out


@st.cache_data(show_spinner=False)
def _get_df():
    return load_dataset()


def _warmup_embeddings(mode: str):
    if mode == "Title Search":
        with st.spinner("Loading title embeddings…"):
            build_or_load_title_embeddings()
    elif mode == "Ingredient Search":
        with st.spinner("Loading ingredient embeddings…"):
            build_or_load_ingredient_embeddings()
    elif mode == "RAG Search":
        with st.spinner("Loading full recipe embeddings…"):
            build_or_load_full_embeddings()


def _render_recipe(rank: int, row_idx: int, score: float):
    df = _get_df()
    row = df.iloc[row_idx]
    combined = combine_fields(row)
    fields = _parse_combined_text(combined)

    title = str(fields.get("title", "")).strip()
    st.markdown(f"### {rank}. {title}  ")
    st.caption(f"Similarity: {score:.4f} | Row #{row_idx}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ingredients")
        ings = fields.get("ingredients", "")
        if isinstance(ings, list):
            for i, item in enumerate(ings, 1):
                st.write(f"{i}. {item}")
        else:
            st.write(ings)
    with col2:
        st.subheader("Directions")
        dirs = fields.get("directions", "")
        if isinstance(dirs, list):
            for i, step in enumerate(dirs, 1):
                st.write(f"{i}. {step}")
        else:
            st.write(dirs)

    st.divider()


def ui_title_search(query: str, top_k: int, method: str):
    matches = search_titles(query, top_k=top_k, method=method)
    if not matches:
        st.info("No matches found.")
        return
    for rank, (idx, score) in enumerate(matches, start=1):
        _render_recipe(rank, idx, score)


def ui_ingredient_search(query: str, top_k: int, method: str):
    include, exclude = extract_ingredients_with_negation(query)
    with st.expander("Query parsing", expanded=True):
        if include:
            st.write({"include": include})
        if exclude:
            st.write({"exclude": exclude})
        if not include and not exclude:
            st.caption("No distinct ingredients extracted; using full query.")
    matches = search_ingredients(query, top_k=top_k, method=method)
    if not matches:
        st.info("No matches found.")
        return
    for rank, (idx, score) in enumerate(matches, start=1):
        _render_recipe(rank, idx, score)


def ui_rag_search(query: str, top_k: int, method: str):
    df = _get_df()
    matches = search_full(query, top_k=top_k, method=method)
    if not matches:
        st.info("No matches found.")
        return

    st.subheader("Top candidates")
    for i, (row_idx, s) in enumerate(matches, start=1):
        title = str(df.iloc[row_idx]["title"]).strip()
        st.write(f"{i}. {title} (Similarity: {s:.4f})")

    st.divider()
    st.subheader("Refined answer")
    with st.spinner("Asking LLM to refine the answer…"):
        answer = ask_llm(query, df, matches)
    st.write(answer)

    with st.expander("Show retrieved contexts"):
        for i, (row_idx, s) in enumerate(matches, start=1):
            row = df.iloc[row_idx]
            st.markdown(f"#### Context {i}")
            st.code(combine_fields(row))


def sidebar_controls():
    st.sidebar.title("Recipe Search")
    mode = st.sidebar.radio(
        "Search Type",
        ["Title Search", "Ingredient Search", "RAG Search"],
    )
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=5)
    
    st.sidebar.markdown("---")
    embedding_method = st.sidebar.radio(
        "Embedding Method",
        ["Dense (Semantic)", "Sparse (TF-IDF)"],
        help="Dense: Sentence transformer embeddings for semantic similarity.\nSparse: TF-IDF vectors for term-based matching."
    )
    method = "dense" if embedding_method == "Dense (Semantic)" else "sparse"

    if mode == "RAG Search":
        st.sidebar.markdown("---")
        st.sidebar.subheader("LLM Settings (Gemini)")
        api_key = st.sidebar.text_input(
            "GOOGLE_API_KEY",
            type="password",
            help="Required for RAG answer refinement.",
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.sidebar.success("API key set in-session.")
        else:
            if not os.getenv("GOOGLE_API_KEY"):
                st.sidebar.warning("No GOOGLE_API_KEY detected. RAG answer may fail.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Dataset and model are configured via config.yaml")
    return mode, top_k, method


def main():
    st.title("Recipe Retrieval Explorer")
    st.caption("Search by title, ingredients, or use RAG refinement.")

    mode, top_k, method = sidebar_controls()
    _warmup_embeddings(mode)

    query = st.text_input("Enter your query")
    do_search = st.button("Search", type="primary")

    if do_search and not query.strip():
        st.warning("Please enter a query.")
        return

    if do_search:
        if mode == "Title Search":
            ui_title_search(query, top_k, method)
        elif mode == "Ingredient Search":
            ui_ingredient_search(query, top_k, method)
        else:
            ui_rag_search(query, top_k, method)


if __name__ == "__main__":
    main()
