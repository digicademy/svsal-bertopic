"""
Create embeddings for text passages using a locally hosted Ollama server.
Designed to be run inside a SLURM job via run_embeddings.slurm.

The script mirrors the output format of the interactive notebook
01-embeddings-create.*.ipynb so that downstream notebooks that consume
those files work without modification.

Output files (written to --output-dir, prefixed with today's date):
  <date>_all_docs.parquet              — document metadata
  <date>_all_embeddings.parquet        — per-document embedding vectors
  <date>_all_embeddings.pkl            — numpy array + doc_id list
  <date>_all_embeddings.jsonl          — vector-DB payload (id + vector + metadata)
  <date>_all_processing_metadata.json  — run configuration
  <date>_all_embedding_statistics.json — success / failure counters
  embeddings_cache.pkl                 — incremental cache for resume

Usage (from repo root):
    uv run python slurm/create_embeddings_ollama.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import openai
import orjson
import polars as pl
import tiktoken


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create embeddings via a local Ollama server (HPC/SLURM version).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument(
        "--input",
        default="./in-data/corpus_20260111.csv",
        help="Path to the input CSV file.",
    )
    p.add_argument(
        "--text-column",
        default="text",
        help=(
            "CSV column containing the text to embed. "
            "Run once with a wrong name to see all available columns."
        ),
    )
    p.add_argument(
        "--id-column",
        default=None,
        help="CSV column to use as document ID (defaults to the first column).",
    )
    p.add_argument(
        "--output-dir",
        default="./out-data",
        help="Directory for all output files.",
    )
    # Model / server
    p.add_argument(
        "--model",
        default="nomic-embed-text",
        help=(
            "Ollama embedding model. Good choices for Spanish/Latin text: "
            "nomic-embed-text (768d), mxbai-embed-large (1024d), bge-m3 (1024d, multilingual)."
        ),
    )
    p.add_argument(
        "--url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible API base URL of the Ollama server.",
    )
    p.add_argument(
        "--api-key",
        default="ollama",
        help="Placeholder API key (local Ollama servers accept any non-empty string).",
    )
    # Filtering
    p.add_argument(
        "--max-documents",
        type=int,
        default=-1,
        help="Maximum number of documents to process (-1 = all).",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=10,
        help="Minimum token count for a passage to be included.",
    )
    p.add_argument(
        "--context-limit",
        type=int,
        default=8192,
        help="Token context limit of the embedding model (texts are truncated to this).",
    )
    # Throughput
    p.add_argument(
        "--concurrent-requests",
        type=int,
        default=5,
        help="Number of concurrent embedding API calls.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of texts sent in a single embedding API call.",
    )
    # Reliability
    p.add_argument(
        "--cache-save-interval",
        type=int,
        default=7200,
        help="Seconds between periodic cache saves (default: 2 h).",
    )
    p.add_argument(
        "--retry-max",
        type=int,
        default=5,
        help="Maximum retries per batch before marking it as failed.",
    )
    return p.parse_args()


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(output_dir, f"{ts}_embeddings_ollama.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    for noisy in ("httpx", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


# ── Statistics ─────────────────────────────────────────────────────────────────

class EmbeddingStatistics:
    """Track counts and timing for the embedding run (mirrors the notebook class)."""

    def __init__(self, model: str):
        self.model = model
        self.total = 0
        self.success = 0
        self.failed = 0
        self.skipped_short = 0
        self.skipped_cached = 0
        self.failed_docs: List[Dict] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def complete(self):
        self.end_time = time.time()

    def record_success(self, doc_id: str):
        self.total += 1
        self.success += 1

    def record_failure(self, doc_id: str, error: str):
        self.total += 1
        self.failed += 1
        self.failed_docs.append({"doc_id": doc_id, "error": error})

    def processing_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def success_rate(self) -> float:
        return (self.success / self.total * 100.0) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped_short": self.skipped_short,
            "skipped_cached": self.skipped_cached,
            "success_rate": round(self.success_rate(), 2),
            "processing_time_s": round(self.processing_time(), 2),
            "failed_docs": self.failed_docs,
        }

    def print_summary(self):
        r = self.success_rate()
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  Embedding statistics — {self.model}")
        print(sep)
        print(f"  Total processed : {self.total}")
        print(f"  Success         : {self.success}  ({r:.1f} %)")
        print(f"  Failed          : {self.failed}")
        print(f"  Skipped (short) : {self.skipped_short}")
        print(f"  Skipped (cache) : {self.skipped_cached}")
        print(f"  Processing time : {self.processing_time():.1f} s")
        print(f"{sep}\n")


# ── Utilities ──────────────────────────────────────────────────────────────────

def batched(iterable, n: int):
    """Yield successive n-sized chunks (last chunk may be shorter)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while chunk := tuple(itertools.islice(it, n)):
        yield chunk


def count_tokens(text: str, encoding) -> int:
    """Count tokens; fall back to whitespace splitting if tiktoken is unavailable."""
    try:
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())


def atomic_save_pickle(data, filepath: str):
    """Write to a .tmp file then atomically rename — safe even if interrupted."""
    tmp = filepath + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp, filepath)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_save_json(data: dict, filepath: str):
    """Atomically write a pretty-printed JSON file via orjson."""
    tmp = filepath + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(
                orjson.dumps(
                    data,
                    option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
                )
            )
        os.replace(tmp, filepath)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_cache(cache_path: str) -> dict:
    """Load the embedding cache from disk (returns empty dict if not found)."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            print(f"Warning: could not load cache from {cache_path}: {exc}")
    return {}


# ── Embedding engine ───────────────────────────────────────────────────────────

async def embed_batch_with_retry(
    client: openai.AsyncOpenAI,
    model: str,
    texts: List[str],
    logger: logging.Logger,
    retry_max: int,
) -> Optional[List[List[float]]]:
    """Call the /v1/embeddings endpoint with exponential-backoff retries."""
    for attempt in range(retry_max):
        try:
            resp = await client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            wait = min(60, 2 ** attempt)
            logger.warning(
                f"Embed attempt {attempt + 1}/{retry_max} failed: {exc!r}. "
                f"Retrying in {wait} s …"
            )
            await asyncio.sleep(wait)
    logger.error(f"All {retry_max} attempts failed for a batch of {len(texts)} texts.")
    return None


async def run_embedding_pipeline(
    records: List[dict],
    client: openai.AsyncOpenAI,
    model: str,
    cache: dict,
    stats: EmbeddingStatistics,
    logger: logging.Logger,
    concurrent_requests: int,
    batch_size: int,
    cache_path: str,
    cache_save_interval: int,
    retry_max: int,
) -> dict:
    """
    Embed all records (skipping those already in cache).
    Returns a merged dict: {doc_id: {"embedding": [...], "metadata": {...}}}.
    """
    results: dict = dict(cache)
    to_embed = [r for r in records if r["doc_id"] not in cache]
    stats.skipped_cached = len(records) - len(to_embed)
    logger.info(
        f"Records to embed : {len(to_embed)}  "
        f"({stats.skipped_cached} already cached, skipping)"
    )

    semaphore = asyncio.Semaphore(concurrent_requests)
    last_save = time.time()
    batches = list(batched(to_embed, batch_size))
    completed = 0

    async def process_batch(batch: list):
        nonlocal completed, last_save
        texts = [r["text"] for r in batch]
        async with semaphore:
            embeddings = await embed_batch_with_retry(
                client, model, texts, logger, retry_max
            )

        if embeddings is None:
            for r in batch:
                stats.record_failure(r["doc_id"], "All retries exhausted")
        else:
            for r, emb in zip(batch, embeddings):
                results[r["doc_id"]] = {
                    "embedding": emb,
                    "metadata": {k: v for k, v in r.items() if k != "text"},
                }
                stats.record_success(r["doc_id"])

        completed += 1
        # Periodic cache save (guards against job walltime kills)
        if time.time() - last_save > cache_save_interval:
            logger.info(f"Periodic cache save ({len(results)} entries) …")
            atomic_save_pickle(results, cache_path)
            last_save = time.time()
        if completed % 100 == 0 or completed == len(batches):
            logger.info(
                f"Progress: {completed}/{len(batches)} batches  "
                f"({stats.success} ok, {stats.failed} failed)"
            )

    await asyncio.gather(*[process_batch(list(b)) for b in batches])

    # Final cache save
    atomic_save_pickle(results, cache_path)
    logger.info(f"Cache saved: {cache_path}  ({len(results)} total entries)")
    return results


# ── Output helpers ─────────────────────────────────────────────────────────────

def save_outputs(
    results: dict,
    output_dir: str,
    stats: EmbeddingStatistics,
    config: dict,
    logger: logging.Logger,
):
    """Write all output files (format matches the interactive notebook)."""
    prefix = datetime.now().strftime("%Y-%m-%d")

    doc_rows: list = []
    emb_ids: list = []
    emb_vectors: list = []
    for doc_id, entry in results.items():
        doc_rows.append({"doc_id": doc_id, **entry["metadata"]})
        emb_ids.append(doc_id)
        emb_vectors.append(entry["embedding"])

    emb_array = np.array(emb_vectors, dtype=np.float32)

    # ── docs parquet ──────────────────────────────────────────────────────────
    path = os.path.join(output_dir, f"{prefix}_all_docs.parquet")
    pl.DataFrame(doc_rows).write_parquet(path)
    logger.info(f"Saved docs parquet       : {path}")

    # ── embeddings parquet ───────────────────────────────────────────────────
    path = os.path.join(output_dir, f"{prefix}_all_embeddings.parquet")
    pl.DataFrame({"doc_id": emb_ids, "embedding": emb_vectors}).write_parquet(path)
    logger.info(f"Saved embeddings parquet : {path}")

    # ── embeddings pickle (numpy array) ──────────────────────────────────────
    path = os.path.join(output_dir, f"{prefix}_all_embeddings.pkl")
    atomic_save_pickle({"doc_ids": emb_ids, "embeddings": emb_array}, path)
    logger.info(f"Saved embeddings pickle  : {path}")

    # ── JSONL (vector-DB payload format) ─────────────────────────────────────
    path = os.path.join(output_dir, f"{prefix}_all_embeddings.jsonl")
    with open(path, "wb") as f:
        for doc_id, entry in results.items():
            payload = {"id": doc_id, "vector": entry["embedding"], **entry["metadata"]}
            f.write(orjson.dumps(payload) + b"\n")
    logger.info(f"Saved JSONL              : {path}")

    # ── processing metadata ───────────────────────────────────────────────────
    path = os.path.join(output_dir, f"{prefix}_all_processing_metadata.json")
    atomic_save_json(config, path)
    logger.info(f"Saved config JSON        : {path}")

    # ── embedding statistics ──────────────────────────────────────────────────
    path = os.path.join(output_dir, f"{prefix}_all_embedding_statistics.json")
    atomic_save_json(stats.to_dict(), path)
    logger.info(f"Saved statistics JSON    : {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    sep = "=" * 52
    logger.info(sep)
    logger.info("  SVSAL Embeddings — Ollama / HPC version")
    logger.info(sep)
    logger.info(f"  Model        : {args.model}")
    logger.info(f"  Server URL   : {args.url}")
    logger.info(f"  Input        : {args.input}")
    logger.info(f"  Output dir   : {args.output_dir}")
    logger.info(f"  Concurrency  : {args.concurrent_requests} requests")
    logger.info(f"  Batch size   : {args.batch_size} texts / call")
    logger.info(f"  Min tokens   : {args.min_tokens}")
    logger.info(f"  Context limit: {args.context_limit} tokens")
    logger.info(sep)

    # ── Load CSV ─────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.input} …")
    df = pl.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows  |  columns: {df.columns}")

    id_col = args.id_column or df.columns[0]
    text_col = args.text_column

    if text_col not in df.columns:
        raise SystemExit(
            f"ERROR: text column '{text_col}' not found in the CSV.\n"
            f"Available columns: {df.columns}\n"
            f"Re-run with --text-column <correct-name>."
        )
    if id_col not in df.columns:
        raise SystemExit(
            f"ERROR: ID column '{id_col}' not found in the CSV.\n"
            f"Available columns: {df.columns}\n"
            f"Re-run with --id-column <correct-name>."
        )

    logger.info(f"ID column    : {id_col}")
    logger.info(f"Text column  : {text_col}")

    # ── Token filtering ──────────────────────────────────────────────────────
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
        logger.warning(
            "tiktoken unavailable — falling back to whitespace token count."
        )

    stats = EmbeddingStatistics(model=args.model)
    records: List[dict] = []

    for row in df.iter_rows(named=True):
        raw_text = row.get(text_col)
        if raw_text is None:
            stats.skipped_short += 1
            continue
        text = str(raw_text)
        n_tok = count_tokens(text, enc)
        if n_tok < args.min_tokens:
            stats.skipped_short += 1
            continue
        # Truncate to model context limit
        if enc and n_tok > args.context_limit:
            text = enc.decode(enc.encode(text)[: args.context_limit])
        records.append({"doc_id": str(row[id_col]), "text": text, **row})

    if args.max_documents > 0:
        records = records[: args.max_documents]

    logger.info(
        f"After filtering: {len(records)} records kept  "
        f"({stats.skipped_short} skipped as too short)"
    )

    # ── Load cache ───────────────────────────────────────────────────────────
    cache_path = os.path.join(args.output_dir, "embeddings_cache.pkl")
    cache = load_cache(cache_path)
    logger.info(f"Cache: {len(cache)} existing entries loaded from {cache_path}")

    # ── Embed ────────────────────────────────────────────────────────────────
    client = openai.AsyncOpenAI(base_url=args.url, api_key=args.api_key)
    stats.start()
    results = await run_embedding_pipeline(
        records=records,
        client=client,
        model=args.model,
        cache=cache,
        stats=stats,
        logger=logger,
        concurrent_requests=args.concurrent_requests,
        batch_size=args.batch_size,
        cache_path=cache_path,
        cache_save_interval=args.cache_save_interval,
        retry_max=args.retry_max,
    )
    stats.complete()
    stats.print_summary()

    # ── Save ─────────────────────────────────────────────────────────────────
    config = {
        "model": args.model,
        "server_url": args.url,
        "input_file": str(args.input),
        "text_column": text_col,
        "id_column": id_col,
        "min_tokens": args.min_tokens,
        "context_limit": args.context_limit,
        "batch_size": args.batch_size,
        "concurrent_requests": args.concurrent_requests,
        "max_documents": args.max_documents,
        "total_input_rows": len(df),
        "total_embedded": len(results),
        "timestamp": datetime.now().isoformat(),
    }
    logger.info("Saving output files …")
    save_outputs(results, args.output_dir, stats, config, logger)
    logger.info("=== All done. ===")


if __name__ == "__main__":
    asyncio.run(main())
