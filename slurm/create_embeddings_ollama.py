"""
Create embeddings for text passages using a locally hosted Ollama server.
Designed to be run inside a SLURM job via run_embeddings_slurm.sh.

Multi-model: the script can embed the same corpus with several Ollama
embedding models in one run, in parallel against the same server. Its
output format matches the interactive notebook
``01-embeddings-create.*.ipynb`` so that all downstream notebooks
(``02-…``, ``03-…``, ``04-…``) consume the files without modification.

Model discovery
---------------
The script can ask the Ollama server which locally-cached models support
embeddings (via the ``/api/tags`` + ``/api/show`` REST endpoints). Two
special model names trigger discovery:

* ``auto`` (default) — use the first available embedding model
  (alphabetical order, deterministic across clusters)
* ``all``  — use every locally available embedding model concurrently

You can also pass one or more concrete model names instead, in which
case discovery is skipped and the named models are used as-is. Model
names can be repeated via ``--model`` or supplied as a comma-separated
list to a single ``--model`` flag::

    --model auto
    --model all
    --model bge-m3 --model nomic-embed-text
    --model bge-m3,nomic-embed-text,mxbai-embed-large

Output files (written to ``--output-dir``)
------------------------------------------
Dated final outputs (prefix ``<YYYY-MM-DD>_``)::

    <date>_all_docs.parquet              -- docs DataFrame with one
                                            ``embeddings_<provider_id>``
                                            column per model
    <date>_all_docs.pkl                  -- pickle of the same DataFrame
    <date>_all_docs.csv                  -- CSV without embedding columns
    <date>_all_embeddings.pkl            -- nested dict pickle:
                                            ``{provider_id: {doc_id: vec}}``
    <date>_all_embeddings.parquet        -- same shape, parquet
    <date>_all_embeddings.jsonl          -- one record per (doc_id, model):
                                            ``{id, model, vector, ...meta}``
    <date>_all_processing_metadata.json  -- per-provider run config
    <date>_all_embedding_statistics.json -- per-provider stats

Per-model parquet files (one each, written when a model finishes)::

    <provider_id>_<YYYYmmdd_HHMMSS>.parquet

Resume state (rewritten incrementally)::

    embeddings_cache.pkl                 -- ``{provider_id: {doc_id: vec}}``
    embeddings_manifest.json             -- ``{providers: {provider_id: …}}``

The cache and manifest are loaded at startup. Models already listed in the
manifest are skipped wholesale; partially-cached models continue from where
they left off.

Usage (from the repo root)::

    uv run python slurm/create_embeddings_ollama.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
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
    # Models / server
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Ollama embedding model name(s). May be passed multiple times, or "
            "given as a comma-separated list. Special values: "
            "'auto' = first available locally (default if --model omitted); "
            "'all'  = every locally available embedding model."
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
        default=None,
        help=(
            "Optional user-supplied cap on tokens per text. If unset (the "
            "default), each model's native context length is used (queried "
            "via /api/show). If set, the effective limit per model is "
            "min(model_native, --context-limit)."
        ),
    )
    # Throughput
    p.add_argument(
        "--concurrent-requests",
        type=int,
        default=5,
        help="Number of concurrent embedding API calls per model.",
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
        help="Seconds between periodic cache checkpoints (default: 2 h).",
    )
    p.add_argument(
        "--retry-max",
        type=int,
        default=5,
        help="Maximum retries per batch before marking it as failed.",
    )
    # Discovery
    p.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "Discover available embedding models on the server, print them, "
            "and exit without embedding anything."
        ),
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
    """Per-model counts and timing (mirrors the notebook's class)."""

    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        self.total = 0
        self.success = 0
        self.failed = 0
        self.skipped_short = 0
        self.skipped_cached = 0
        self.failed_docs: List[Dict] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> None:
        self.start_time = time.time()

    def complete(self) -> None:
        self.end_time = time.time()

    def record_success(self, doc_id: str) -> None:
        self.total += 1
        self.success += 1

    def record_failure(self, doc_id: str, error: str) -> None:
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
            "provider_id": self.provider_id,
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped_short": self.skipped_short,
            "skipped_cached": self.skipped_cached,
            "success_rate": round(self.success_rate(), 2),
            "processing_time_s": round(self.processing_time(), 2),
            "failed_docs": self.failed_docs,
        }

    def print_summary(self) -> None:
        r = self.success_rate()
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  {self.provider_id}")
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


def atomic_save_pickle(data, filepath: str) -> None:
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


def atomic_save_json(data: dict, filepath: str) -> None:
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


def atomic_save_parquet(df: pl.DataFrame, filepath: str) -> None:
    """Atomically write a polars DataFrame to a parquet file."""
    tmp = filepath + ".tmp"
    try:
        df.write_parquet(tmp)
        os.replace(tmp, filepath)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_pickle(filepath: str, default: Any) -> Any:
    """Load a pickle file; return ``default`` if missing or unreadable."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            print(f"Warning: could not load {filepath}: {exc}", file=sys.stderr)
    return default


def load_json(filepath: str, default: Any) -> Any:
    """Load a JSON file; return ``default`` if missing or unreadable."""
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                return json.load(f)
        except Exception as exc:
            print(f"Warning: could not load {filepath}: {exc}", file=sys.stderr)
    return default


def provider_id_for(model: str) -> str:
    """Build the provider identifier used by the notebook (``localhost_<model>``)."""
    return f"localhost_{model}"


# ── Ollama model discovery ────────────────────────────────────────────────────

def _native_base_url(openai_base_url: str) -> str:
    """Convert the OpenAI-compatible URL (``…/v1``) to the native Ollama base.

    Ollama's native REST endpoints (``/api/tags``, ``/api/show``) live one
    level above the OpenAI-compatible ``/v1`` prefix.
    """
    url = openai_base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def _model_info_value(model_info: dict, suffix: str) -> Optional[int]:
    """Return the int value of the first ``model_info`` key ending in ``suffix``.

    Ollama's ``/api/show`` ``model_info`` block uses GGUF metadata keys
    prefixed by the model's architecture (e.g. ``bert.context_length``,
    ``nomic-bert.embedding_length``). Since the architecture prefix varies
    per model, we scan for any key whose suffix matches the GGUF
    convention.
    """
    for key, value in (model_info or {}).items():
        if key.endswith(suffix):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _extract_model_config(name: str, show: dict) -> Dict[str, Any]:
    """Pull the values we care about out of a ``/api/show`` response.

    ``context_length`` and ``embedding_length`` come from GGUF metadata in
    ``model_info``; family / parameter_size / quantization come from the
    ``details`` block. Anything missing is left as ``None`` so the caller
    can fall back gracefully.
    """
    model_info = show.get("model_info") or {}
    details = show.get("details") or {}
    return {
        "name": name,
        "context_length": _model_info_value(model_info, ".context_length"),
        "embedding_length": _model_info_value(model_info, ".embedding_length"),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }


async def discover_embedding_models(
    openai_base_url: str,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Return the alphabetically-sorted list of locally-cached embedding models.

    Each entry is a config dict produced by :func:`_extract_model_config`.
    A model is considered an embedding model iff ``/api/show`` reports
    ``"embedding"`` in its ``capabilities`` array (added to Ollama in PR
    #10066, Mar 2025; derived from the GGUF ``pooling_type`` metadata).
    Network errors and capability misses are logged and skipped.
    """
    base = _native_base_url(openai_base_url)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            tags_resp = await client.get(f"{base}/api/tags")
            tags_resp.raise_for_status()
        except Exception as exc:
            logger.error(f"Failed to call {base}/api/tags: {exc!r}")
            return []
        names = [m["name"] for m in tags_resp.json().get("models", [])]
        configs: List[Dict[str, Any]] = []
        for name in names:
            try:
                show_resp = await client.post(
                    f"{base}/api/show", json={"model": name}
                )
                show_resp.raise_for_status()
                show = show_resp.json()
                caps = show.get("capabilities") or []
                if "embedding" not in caps:
                    continue
                configs.append(_extract_model_config(name, show))
            except Exception as exc:
                logger.warning(f"/api/show failed for '{name}': {exc!r} — skipping")
    return sorted(configs, key=lambda c: c["name"])


async def fetch_model_config(
    name: str, openai_base_url: str, logger: logging.Logger,
) -> Dict[str, Any]:
    """Query ``/api/show`` for one specific model the user named explicitly.

    Used when the user passes a model name that isn't in the discovery
    list (or when discovery was skipped). Returns a config dict with
    ``None`` fields if the call fails, so the caller can still proceed
    with whatever defaults the user supplied.
    """
    base = _native_base_url(openai_base_url)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{base}/api/show", json={"model": name})
            resp.raise_for_status()
            return _extract_model_config(name, resp.json())
        except Exception as exc:
            logger.warning(
                f"/api/show failed for explicitly-requested '{name}': {exc!r}. "
                f"Proceeding without per-model context limit; the user-supplied "
                f"--context-limit will be used as the cap."
            )
            return {"name": name, "context_length": None,
                    "embedding_length": None, "family": None,
                    "parameter_size": None, "quantization_level": None}


async def resolve_model_configs(
    requested: Optional[List[str]],
    available: List[Dict[str, Any]],
    openai_base_url: str,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Resolve the user's ``--model`` arguments to a list of config dicts.

    ``requested`` is the post-parse, post-comma-split list of model names
    or special tokens (``auto`` / ``all``). When ``None``, the default is
    ``auto``. Returns deduplicated, order-preserving config dicts. Exits
    with a clear error if no model can be chosen.
    """
    if not requested:
        requested = ["auto"]
    by_name = {c["name"]: c for c in available}
    resolved: List[Dict[str, Any]] = []

    for token in requested:
        if token == "all":
            if not available:
                logger.error(
                    "--model all was requested but no embedding models are "
                    "available on the server. Pre-download at least one model "
                    "(see slurm/README.md, section 'Pre-download models')."
                )
                sys.exit(2)
            resolved.extend(available)
        elif token == "auto":
            if not available:
                logger.error(
                    "--model auto was requested (or defaulted to) but no "
                    "embedding models are available on the server. "
                    "Pre-download at least one model (see slurm/README.md, "
                    "section 'Pre-download models')."
                )
                sys.exit(2)
            resolved.append(available[0])
            logger.info(
                f"--model auto -> picked '{available[0]['name']}' "
                f"(first of {len(available)} available embedding models)"
            )
        else:
            # The user named a specific model. If it's already in our
            # discovery results, reuse that config; otherwise ask the
            # server about it directly (covers e.g. tag-mismatch cases
            # like 'bge-m3' vs the cached 'bge-m3:latest').
            cfg = by_name.get(token)
            if cfg is None:
                # Try matching by ignoring the tag.
                base_name = token.split(":")[0]
                for c in available:
                    if c["name"].split(":")[0] == base_name:
                        cfg = c
                        break
            if cfg is None:
                cfg = await fetch_model_config(token, openai_base_url, logger)
            resolved.append(cfg)

    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for cfg in resolved:
        if cfg["name"] not in seen:
            seen.add(cfg["name"])
            deduped.append(cfg)
    return deduped


def effective_context_limit(
    cfg: Dict[str, Any],
    user_cap: Optional[int],
    fallback: int,
    logger: logging.Logger,
) -> int:
    """Decide the per-model truncation threshold.

    Precedence: the model's native ``context_length`` (queried from
    ``/api/show``) is the canonical answer. If the user explicitly set
    ``--context-limit``, we treat that as a cap (``min(native, user)``).
    If neither is available, ``fallback`` is used and a warning logged.
    """
    native = cfg.get("context_length")
    if native is None and user_cap is None:
        logger.warning(
            f"[{cfg['name']}] no native context_length in /api/show "
            f"response and no --context-limit set; falling back to "
            f"{fallback} tokens. Server-side truncate=true is still on."
        )
        return fallback
    if native is None:
        return user_cap  # type: ignore[return-value]
    if user_cap is None:
        return native
    return min(native, user_cap)


# ── Embedding engine (per-model) ───────────────────────────────────────────────

async def embed_batch_with_retry(
    client: openai.AsyncOpenAI,
    model: str,
    texts: List[str],
    logger: logging.Logger,
    retry_max: int,
) -> Optional[List[List[float]]]:
    """Call ``/v1/embeddings`` with exponential-backoff retries.

    Returns ``None`` if every retry fails. Successful calls return one
    embedding per input text, in the original input order.
    """
    for attempt in range(retry_max):
        try:
            resp = await client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            wait = min(60, 2 ** attempt)
            logger.warning(
                f"[{model}] batch attempt {attempt + 1}/{retry_max} failed: "
                f"{exc!r}. Retrying in {wait} s …"
            )
            await asyncio.sleep(wait)
    logger.error(f"[{model}] all {retry_max} retries failed for batch of {len(texts)}.")
    return None


async def process_single_model(
    model: str,
    records: List[dict],
    client: openai.AsyncOpenAI,
    cache: Dict[str, Dict[str, List[float]]],
    cache_lock: asyncio.Lock,
    manifest: dict,
    manifest_lock: asyncio.Lock,
    last_cache_save: Dict[str, float],
    cache_path: str,
    manifest_path: str,
    output_dir: str,
    stats: EmbeddingStatistics,
    logger: logging.Logger,
    concurrent_requests: int,
    batch_size: int,
    cache_save_interval: int,
    retry_max: int,
    context_limit: int,
    encoding,
) -> None:
    """Embed ``records`` with one model. Updates the shared cache/manifest in place.

    Mirrors the notebook's ``process_single_provider`` design: load the
    per-provider cache slice, embed only the missing docs in batches under
    a per-model semaphore, checkpoint periodically, then write a
    per-provider parquet file and append an entry to the manifest.

    ``context_limit`` is the per-model effective truncation threshold
    (resolved by :func:`effective_context_limit`). Each text is truncated
    to this many tokens before being sent. Ollama's server-side
    ``truncate: true`` (the /v1/embeddings default) is the safety net for
    cases where our token count is off.
    """
    pid = provider_id_for(model)
    stats.start()

    # ── Manifest shortcut: a model already marked completed is skipped ───────
    async with manifest_lock:
        completed = (
            pid in manifest["providers"]
            and manifest["providers"][pid].get("num_embeddings", 0) > 0
        )
    if completed:
        logger.info(
            f"[{pid}] already completed per manifest "
            f"({manifest['providers'][pid]['num_embeddings']} embeddings); skipping."
        )
        async with cache_lock:
            stats.skipped_cached = len(cache.get(pid, {}))
            for doc_id in cache.get(pid, {}):
                stats.record_success(doc_id)
        stats.complete()
        return

    # ── Identify which docs still need embeddings ────────────────────────────
    async with cache_lock:
        provider_cache = cache.setdefault(pid, {})
        needed_ids = {r["doc_id"] for r in records}
        cached_ids = set(provider_cache) & needed_ids
        missing_ids = needed_ids - cached_ids
        for doc_id in cached_ids:
            stats.record_success(doc_id)
        stats.skipped_cached = len(cached_ids)

    if not missing_ids:
        logger.info(f"[{pid}] all {len(cached_ids)} docs already cached; finalising.")
        await _finalise_provider(
            pid, cache, cache_lock, manifest, manifest_lock,
            output_dir, manifest_path, logger,
        )
        stats.complete()
        return

    logger.info(
        f"[{pid}] embedding {len(missing_ids)} docs "
        f"(cached: {len(cached_ids)}; concurrency={concurrent_requests}; "
        f"batch={batch_size}; context_limit={context_limit} tokens)"
    )

    # ── Embed missing docs in concurrent batches ─────────────────────────────
    missing_records = [r for r in records if r["doc_id"] in missing_ids]
    semaphore = asyncio.Semaphore(concurrent_requests)
    batches = list(batched(missing_records, batch_size))
    completed_batches = 0

    def _truncate(text: str) -> str:
        """Per-model truncation. The token count is a heuristic (tiktoken's
        cl100k_base, not the model's own tokenizer) — Ollama's server-side
        truncate=true is the authoritative safety net.
        """
        if encoding is None:
            # No tokenizer at all: be conservative and cap by characters
            # (~4 chars/token is a common rule of thumb).
            max_chars = context_limit * 4
            return text if len(text) <= max_chars else text[:max_chars]
        toks = encoding.encode(text)
        if len(toks) <= context_limit:
            return text
        return encoding.decode(toks[:context_limit])

    async def process_batch(batch: tuple) -> None:
        nonlocal completed_batches
        texts = [_truncate(r["text"]) for r in batch]
        async with semaphore:
            embeddings = await embed_batch_with_retry(
                client, model, texts, logger, retry_max
            )
        async with cache_lock:
            if embeddings is None:
                for r in batch:
                    stats.record_failure(r["doc_id"], "All retries exhausted")
            else:
                for r, emb in zip(batch, embeddings):
                    provider_cache[r["doc_id"]] = emb
                    stats.record_success(r["doc_id"])
            completed_batches += 1
            # Periodic cache checkpoint — guards against walltime kills.
            now = time.time()
            if now - last_cache_save["time"] >= cache_save_interval:
                logger.info(
                    f"[{pid}] cache checkpoint after {completed_batches}/"
                    f"{len(batches)} batches "
                    f"({stats.success} ok, {stats.failed} failed) …"
                )
                atomic_save_pickle(cache, cache_path)
                last_cache_save["time"] = now
        if completed_batches % 100 == 0 or completed_batches == len(batches):
            logger.info(
                f"[{pid}] progress: {completed_batches}/{len(batches)} batches "
                f"({stats.success} ok, {stats.failed} failed)"
            )

    await asyncio.gather(*(process_batch(b) for b in batches))

    # ── Finalise: per-provider parquet + manifest entry ──────────────────────
    await _finalise_provider(
        pid, cache, cache_lock, manifest, manifest_lock,
        output_dir, manifest_path, logger,
    )
    stats.complete()


async def _finalise_provider(
    pid: str,
    cache: Dict[str, Dict[str, List[float]]],
    cache_lock: asyncio.Lock,
    manifest: dict,
    manifest_lock: asyncio.Lock,
    output_dir: str,
    manifest_path: str,
    logger: logging.Logger,
) -> None:
    """Write the per-provider parquet file and update the manifest atomically."""
    async with cache_lock:
        snapshot = dict(cache.get(pid, {}))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{pid}_{ts}.parquet"
    filepath = os.path.join(output_dir, filename)
    df = pl.DataFrame(
        [{"doc_id": doc_id, "embedding": vec} for doc_id, vec in snapshot.items()]
    )
    atomic_save_parquet(df, filepath)
    async with manifest_lock:
        manifest["providers"][pid] = {
            "file": filename,
            "filepath": filepath,
            "completed_at": datetime.now().isoformat(),
            "num_embeddings": len(snapshot),
        }
        atomic_save_json(manifest, manifest_path)
    logger.info(f"[{pid}] wrote {len(snapshot)} embeddings -> {filename}")


# ── Output assembly (notebook-compatible) ──────────────────────────────────────

def assemble_outputs(
    df_docs: pl.DataFrame,
    id_col: str,
    models: List[str],
    model_configs: Dict[str, Dict[str, Any]],
    context_limits: Dict[str, int],
    cache: Dict[str, Dict[str, List[float]]],
    stats_by_pid: Dict[str, EmbeddingStatistics],
    config: dict,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Write the dated, final output files in the notebook's shapes."""
    prefix = datetime.now().strftime("%Y-%m-%d") + "_"

    docs_pkl = os.path.join(output_dir, f"{prefix}all_docs.pkl")
    docs_parquet = os.path.join(output_dir, f"{prefix}all_docs.parquet")
    docs_csv = os.path.join(output_dir, f"{prefix}all_docs.csv")
    emb_pkl = os.path.join(output_dir, f"{prefix}all_embeddings.pkl")
    emb_parquet = os.path.join(output_dir, f"{prefix}all_embeddings.parquet")
    emb_jsonl = os.path.join(output_dir, f"{prefix}all_embeddings.jsonl")
    config_path = os.path.join(output_dir, f"{prefix}all_processing_metadata.json")
    stats_path = os.path.join(output_dir, f"{prefix}all_embedding_statistics.json")

    # ── Docs DataFrame: one ``embeddings_<provider_id>`` column per model ───
    enriched = df_docs
    for model in models:
        pid = provider_id_for(model)
        col = f"embeddings_{pid}"
        provider_cache = cache.get(pid, {})

        def _lookup(doc_id: str, _pc=provider_cache) -> Optional[list]:
            return _pc.get(str(doc_id))

        enriched = enriched.with_columns(
            pl.col(id_col)
            .map_elements(_lookup, return_dtype=pl.List(pl.Float32))
            .alias(col)
        )

    # Polars 1.x doesn't support pickling LazyFrame, but eager DataFrames pickle
    # fine via ``pickle.dumps`` — same behaviour the notebook relies on.
    atomic_save_pickle(enriched, docs_pkl)
    atomic_save_parquet(enriched, docs_parquet)
    docs_no_emb = enriched.drop([c for c in enriched.columns if c.startswith("embeddings_")])
    docs_no_emb.write_csv(docs_csv)
    logger.info(f"Wrote {docs_pkl}")
    logger.info(f"Wrote {docs_parquet}")
    logger.info(f"Wrote {docs_csv}")

    # ── Nested-dict embeddings (matches notebook's cache_data shape) ────────
    cache_for_models: Dict[str, Dict[str, List[float]]] = {
        provider_id_for(m): cache.get(provider_id_for(m), {}) for m in models
    }
    atomic_save_pickle(cache_for_models, emb_pkl)
    # Parquet: match the notebook exactly. ``pl.DataFrame(d).write_parquet``
    # on a dict-of-dicts produces a one-row table where each provider_id is
    # one column whose cell holds that provider's ``{doc_id: vector}`` map.
    pl.DataFrame(cache_for_models).write_parquet(emb_parquet)
    logger.info(f"Wrote {emb_pkl}")
    logger.info(f"Wrote {emb_parquet}")

    # ── JSONL: one record per (doc_id, model) for vector-DB upload ──────────
    # Pull metadata from the docs DataFrame so each line carries the original
    # columns alongside the embedding.
    meta_lookup = {
        str(row[id_col]): {k: v for k, v in row.items() if k != id_col}
        for row in enriched.drop([c for c in enriched.columns if c.startswith("embeddings_")]).iter_rows(named=True)
    }
    with open(emb_jsonl, "wb") as f:
        for model in models:
            pid = provider_id_for(model)
            for doc_id, vector in cache_for_models[pid].items():
                payload = {
                    "id": doc_id,
                    "model": model,
                    "provider_id": pid,
                    "vector": vector,
                    **meta_lookup.get(str(doc_id), {}),
                }
                f.write(orjson.dumps(payload) + b"\n")
    logger.info(f"Wrote {emb_jsonl}")

    # ── Per-provider config / stats JSON ─────────────────────────────────────
    config_export = {
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": config["input_file"],
        "id_column": config["id_column"],
        "text_column": config["text_column"],
        "documents": {
            "total_available": config["total_input_rows"],
            "processed": config["max_documents"] if config["max_documents"] > 0 else config["total_input_rows"],
            "min_tokens_threshold": config["min_tokens"],
        },
        "providers": {
            provider_id_for(model): {
                "provider": "localhost",
                "model": model,
                "api_spec": "openai",
                "base_url": config["server_url"],
                "effective_context_limit": context_limits[model],
                "native_context_length": model_configs[model].get("context_length"),
                "embedding_length": model_configs[model].get("embedding_length"),
                "family": model_configs[model].get("family"),
                "parameter_size": model_configs[model].get("parameter_size"),
                "quantization_level": model_configs[model].get("quantization_level"),
                "user_context_cap": config["context_limit"],
                "concurrent_requests": config["concurrent_requests"],
                "batch_size": config["batch_size"],
                "documents_processed": len(cache_for_models[provider_id_for(model)]),
            }
            for model in models
        },
    }
    atomic_save_json(config_export, config_path)
    logger.info(f"Wrote {config_path}")

    # Stats: one provider-keyed sub-dict per model, plus an overall summary.
    failed_by_pid = {
        pid: {f["doc_id"] for f in s.failed_docs}
        for pid, s in stats_by_pid.items()
    }
    all_doc_ids = {str(row[id_col]) for row in df_docs.iter_rows(named=True)}
    critical_failures = (
        set.intersection(*failed_by_pid.values())
        if failed_by_pid and all(failed_by_pid.values())
        else set()
    )
    partial_failures = {
        doc_id for doc_id in all_doc_ids
        if 0 < sum(1 for s in failed_by_pid.values() if doc_id in s) < len(failed_by_pid)
    }
    stats_export = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_documents": len(all_doc_ids),
        "providers": {pid: s.to_dict() for pid, s in stats_by_pid.items()},
        "summary": {
            "critical_failures": len(critical_failures),
            "partial_success": len(partial_failures),
            "full_success": len(all_doc_ids) - len(critical_failures) - len(partial_failures),
            "critical_failure_docs": sorted(critical_failures),
        },
    }
    atomic_save_json(stats_export, stats_path)
    logger.info(f"Wrote {stats_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def split_model_args(raw: Optional[List[str]]) -> Optional[List[str]]:
    """Split comma-separated ``--model`` values so ``a,b,c`` is the same as ``-m a -m b -m c``."""
    if raw is None:
        return None
    out: List[str] = []
    for token in raw:
        out.extend(part.strip() for part in token.split(",") if part.strip())
    return out or None


async def amain() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    # ── Discover what's available on the server (with config) ────────────────
    available_configs = await discover_embedding_models(args.url, logger)
    if args.list_models:
        if available_configs:
            print("Available embedding models on " + args.url + ":")
            for cfg in available_configs:
                ctx = cfg["context_length"] or "?"
                dim = cfg["embedding_length"] or "?"
                fam = cfg["family"] or "?"
                print(
                    f"  - {cfg['name']:40s}  context={ctx:>6}  "
                    f"dim={dim:>5}  family={fam}"
                )
        else:
            print("No embedding models found on " + args.url + ".")
        return

    # ── Resolve the requested model list (rich configs) ──────────────────────
    requested = split_model_args(args.model)
    model_configs = await resolve_model_configs(
        requested, available_configs, args.url, logger
    )
    models = [cfg["name"] for cfg in model_configs]

    # ── Per-model effective context limits ───────────────────────────────────
    FALLBACK_CONTEXT = 8192  # last-resort default if discovery + flag both empty
    context_limits = {
        cfg["name"]: effective_context_limit(
            cfg, args.context_limit, FALLBACK_CONTEXT, logger
        )
        for cfg in model_configs
    }

    sep = "=" * 52
    logger.info(sep)
    logger.info("  SVSAL Embeddings — Ollama / HPC version")
    logger.info(sep)
    logger.info(f"  Server URL   : {args.url}")
    logger.info(f"  Input        : {args.input}")
    logger.info(f"  Output dir   : {args.output_dir}")
    logger.info(f"  Concurrency  : {args.concurrent_requests} requests/model")
    logger.info(f"  Batch size   : {args.batch_size} texts / call")
    logger.info(f"  Min tokens   : {args.min_tokens}")
    user_cap_str = (
        f"{args.context_limit} tokens (user cap)"
        if args.context_limit is not None else "model-native"
    )
    logger.info(f"  Context limit: {user_cap_str}")
    logger.info("  Models:")
    for cfg in model_configs:
        native = cfg["context_length"] or "?"
        dim = cfg["embedding_length"] or "?"
        logger.info(
            f"    - {cfg['name']:40s}  effective_ctx="
            f"{context_limits[cfg['name']]:>6}  native_ctx={native:>6}  dim={dim}"
        )
    if available_configs:
        logger.info(
            f"  Discovered   : {', '.join(c['name'] for c in available_configs)}"
        )
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

    # ── Token-count filter (pre-embedding) ───────────────────────────────────
    # We use tiktoken (cl100k_base) as a rough token-count heuristic for the
    # min_tokens filter only — it's not any of these models' actual tokenizer,
    # but it's a fine approximation for "is this text too short to bother".
    # Real per-model truncation happens at batch time inside process_single_model
    # against each model's effective context limit, with Ollama's server-side
    # truncate=true as the authoritative safety net.
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
        logger.warning("tiktoken unavailable — falling back to whitespace token count.")

    skipped_short = 0
    records: List[dict] = []
    for row in df.iter_rows(named=True):
        raw_text = row.get(text_col)
        if raw_text is None:
            skipped_short += 1
            continue
        text = str(raw_text)
        n_tok = count_tokens(text, enc)
        if n_tok < args.min_tokens:
            skipped_short += 1
            continue
        # No pre-truncation here -- each model truncates to its own limit.
        records.append({"doc_id": str(row[id_col]), "text": text, **row})
    if args.max_documents > 0:
        records = records[: args.max_documents]
    logger.info(
        f"After filtering: {len(records)} records kept "
        f"({skipped_short} skipped as too short)"
    )

    # Keep only the kept rows in the docs DataFrame so output files only
    # contain rows that actually have embeddings.
    kept_ids = [r["doc_id"] for r in records]
    df_docs = df.with_columns(pl.col(id_col).cast(pl.Utf8)).filter(
        pl.col(id_col).is_in(kept_ids)
    )

    # ── Load cache & manifest ────────────────────────────────────────────────
    cache_path = os.path.join(args.output_dir, "embeddings_cache.pkl")
    manifest_path = os.path.join(args.output_dir, "embeddings_manifest.json")
    cache: Dict[str, Dict[str, List[float]]] = load_pickle(cache_path, {})
    manifest: dict = load_json(manifest_path, {"providers": {}})
    if "providers" not in manifest:
        manifest["providers"] = {}
    for pid, doc_map in cache.items():
        if doc_map:
            logger.info(f"Cache: {pid} -> {len(doc_map)} existing embeddings")

    # ── Embed (all models concurrently against the shared server) ───────────
    cache_lock = asyncio.Lock()
    manifest_lock = asyncio.Lock()
    last_cache_save = {"time": time.time()}
    stats_by_pid: Dict[str, EmbeddingStatistics] = {
        provider_id_for(m): EmbeddingStatistics(provider_id_for(m)) for m in models
    }

    client = openai.AsyncOpenAI(base_url=args.url, api_key=args.api_key)

    tasks = [
        process_single_model(
            model=model,
            records=records,
            client=client,
            cache=cache,
            cache_lock=cache_lock,
            manifest=manifest,
            manifest_lock=manifest_lock,
            last_cache_save=last_cache_save,
            cache_path=cache_path,
            manifest_path=manifest_path,
            output_dir=args.output_dir,
            stats=stats_by_pid[provider_id_for(model)],
            logger=logger,
            concurrent_requests=args.concurrent_requests,
            batch_size=args.batch_size,
            cache_save_interval=args.cache_save_interval,
            retry_max=args.retry_max,
            context_limit=context_limits[model],
            encoding=enc,
        )
        for model in models
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for model, result in zip(models, results):
        if isinstance(result, BaseException):
            logger.error(f"[{provider_id_for(model)}] failed with: {result!r}")

    # Final cache save (the per-model finalisers already wrote parquet+manifest).
    atomic_save_pickle(cache, cache_path)
    logger.info(f"Final cache saved to {cache_path}")

    for stats in stats_by_pid.values():
        stats.print_summary()

    # ── Save final outputs ───────────────────────────────────────────────────
    config = {
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
    }
    logger.info("Assembling final output files …")
    assemble_outputs(
        df_docs=df_docs,
        id_col=id_col,
        models=models,
        model_configs={cfg["name"]: cfg for cfg in model_configs},
        context_limits=context_limits,
        cache=cache,
        stats_by_pid=stats_by_pid,
        config=config,
        output_dir=args.output_dir,
        logger=logger,
    )
    logger.info("=== All done. ===")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
