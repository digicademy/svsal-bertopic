#!/usr/bin/env python3
"""Create an Embedding Atlas-ready dataset from aggregated embedding outputs."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
from typing import Optional

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export embeddings + metadata + BERTopic annotations for Embedding Atlas."
    )
    parser.add_argument(
        "--input-dir",
        default="./out-data",
        help="Directory containing *_all_docs.parquet files.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Explicit input parquet path. If omitted, latest *_all_docs.parquet from --input-dir is used.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output parquet path. Defaults to <input-dir>/embeddings_atlas_<provider>.parquet.",
    )
    parser.add_argument(
        "--embedding-column",
        default=None,
        help="Embedding column to use (e.g., embeddings_openai_text-embedding-3-small). Defaults to first embeddings_* column.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Text column to use for BERTopic/snippets. Auto-detected from content/passage/text if omitted.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=280,
        help="Maximum number of characters for text_snippet.",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=20,
        help="BERTopic minimum topic size.",
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=8,
        help="Top keywords to extract per topic.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP neighbors for projection and BERTopic dimensionality reduction.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum distance for 2D projection.",
    )
    parser.add_argument(
        "--also-jsonl",
        action="store_true",
        help="Also write JSONL file next to parquet output.",
    )
    parser.add_argument(
        "--also-csv",
        action="store_true",
        help="Also write CSV file next to parquet output.",
    )
    parser.add_argument(
        "--launch-atlas",
        action="store_true",
        help="Launch local Embedding Atlas UI via `embedding-atlas <output-file>` if installed.",
    )
    return parser.parse_args()


def find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Return newest file path matching pattern in directory, or None."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def detect_text_column(df, explicit_column: Optional[str]) -> str:
    """Resolve text column from explicit name or common defaults."""
    if explicit_column:
        if explicit_column not in df.columns:
            raise ValueError(f"Text column '{explicit_column}' not found in input data")
        return explicit_column
    for candidate in ["content", "passage", "text"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not auto-detect text column. Please pass --text-column explicitly."
    )


def detect_embedding_column(df, explicit_column: Optional[str]) -> str:
    """Resolve embedding column from explicit name or first embeddings_* column."""
    if explicit_column:
        if explicit_column not in df.columns:
            raise ValueError(
                f"Embedding column '{explicit_column}' not found in input data"
            )
        return explicit_column
    embedding_columns = [c for c in df.columns if c.startswith("embeddings_")]
    if not embedding_columns:
        raise ValueError("No embeddings_* columns found in input data")
    return embedding_columns[0]


def topic_keywords_map(topic_model, top_n_words: int) -> dict[int, str]:
    """Build topic_id -> comma-separated top keywords mapping."""
    info = topic_model.get_topic_info()
    topic_ids = info["Topic"].tolist()
    mapping: dict[int, str] = {}
    for topic_id in topic_ids:
        if topic_id == -1:
            mapping[topic_id] = ""
            continue
        words = topic_model.get_topic(topic_id) or []
        mapping[topic_id] = ", ".join(w for w, _ in words[:top_n_words])
    return mapping


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    try:
        import numpy as np
        import polars as pl
        import umap
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for atlas export. Install project dependencies first "
            "(e.g. `pip install numpy polars umap-learn bertopic hdbscan`)."
        ) from exc

    input_file = args.input_file or find_latest_file(args.input_dir, "*_all_docs.parquet")
    if not input_file:
        raise FileNotFoundError(
            f"No input file found. Provide --input-file or place *_all_docs.parquet in {args.input_dir}"
        )
    logger.info("Loading docs from %s", input_file)
    docs = pl.read_parquet(input_file)

    embedding_col = detect_embedding_column(docs, args.embedding_column)
    text_col = detect_text_column(docs, args.text_column)
    provider = embedding_col.removeprefix("embeddings_")
    logger.info("Using embedding column: %s", embedding_col)
    logger.info("Using text column: %s", text_col)

    docs_subset = docs.filter(pl.col(embedding_col).is_not_null())
    if docs_subset.height == 0:
        raise ValueError(f"No rows with non-null embeddings in column {embedding_col}")
    embeddings_array = np.array(docs_subset[embedding_col].to_list(), dtype=np.float32)
    logger.info("Prepared %s embeddings (dim=%s)", embeddings_array.shape[0], embeddings_array.shape[1])

    logger.info("Running UMAP 2D projection")
    projection_model = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    projection_2d = projection_model.fit_transform(embeddings_array)

    topic_texts = (
        docs_subset[text_col].fill_null("").cast(pl.Utf8).to_list()
    )
    topic_texts = [
        doc_text if doc_text else f"[EMPTY:{idx}]"
        for idx, doc_text in enumerate(topic_texts)
    ]

    logger.info("Fitting BERTopic")
    topic_model = BERTopic(
        umap_model=umap.UMAP(
            n_neighbors=args.umap_n_neighbors,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=args.min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(topic_texts, embeddings_array)
    document_info = topic_model.get_document_info(topic_texts)
    topic_keywords = topic_keywords_map(topic_model, args.top_n_words)

    metadata_cols = [c for c in docs_subset.columns if not c.startswith("embeddings_")]
    atlas_df = (
        docs_subset.select(metadata_cols)
        .with_columns(
            pl.Series("atlas_x", projection_2d[:, 0]),
            pl.Series("atlas_y", projection_2d[:, 1]),
            pl.Series("topic_id", topics),
            pl.Series("topic_label", document_info["Name"].tolist()),
            pl.Series("topic_probability", document_info["Probability"].tolist()),
            pl.Series(
                "topic_keywords",
                [topic_keywords.get(topic_id, "") for topic_id in topics],
            ),
            pl.Series("embedding_provider", [provider] * len(topics)),
        )
        .with_columns(
            pl.col(text_col).cast(pl.Utf8).str.slice(0, args.snippet_chars).alias("text_snippet")
        )
    )

    output_file = args.output_file or os.path.join(
        args.input_dir, f"embeddings_atlas_{provider}.parquet"
    )
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    atlas_df.write_parquet(output_file)
    logger.info("Wrote atlas parquet: %s", output_file)

    topics_output = os.path.splitext(output_file)[0] + "_topics.parquet"
    topic_info = topic_model.get_topic_info()
    topic_info["topic_keywords"] = topic_info["Topic"].map(
        lambda t: topic_keywords.get(int(t), "")
    )
    pl.from_pandas(topic_info).write_parquet(topics_output)
    logger.info("Wrote topic summary parquet: %s", topics_output)

    if args.also_jsonl:
        jsonl_path = os.path.splitext(output_file)[0] + ".jsonl"
        atlas_df.write_ndjson(jsonl_path)
        logger.info("Wrote atlas JSONL: %s", jsonl_path)
    if args.also_csv:
        csv_path = os.path.splitext(output_file)[0] + ".csv"
        atlas_df.write_csv(csv_path)
        logger.info("Wrote atlas CSV: %s", csv_path)

    if args.launch_atlas:
        try:
            subprocess.run(["embedding-atlas", output_file], check=True)
        except FileNotFoundError:
            logger.error(
                "embedding-atlas CLI not found. Install with: pip install embedding-atlas"
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Embedding Atlas launcher exited with status code %d",
                exc.returncode,
            )


if __name__ == "__main__":
    main()
