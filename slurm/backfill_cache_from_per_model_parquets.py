"""Backfill ``embeddings_cache.pkl`` + ``embeddings_manifest.json`` from per-model parquet files.

Use this when you have per-model parquet snapshots from earlier
single-model SLURM runs (filenames like
``localhost_<model>_<YYYYmmdd_HHMMSS>.parquet``) but the cache/manifest
are missing, stale, or were never accumulated -- typically because the
runs happened before the cross-run accumulation in
``create_embeddings_ollama.py`` landed.

After running this, the next SLURM job will see all previously-embedded
providers in its cache and produce a dated docs parquet whose columns
span every model that has ever been embedded against this corpus.

Usage::

    uv run python slurm/backfill_cache_from_per_model_parquets.py \\
        --output-dir out-data

Defaults are conservative: a backup of any pre-existing cache/manifest
is written next to the originals before they are overwritten. Pass
``--dry-run`` to preview what would happen without touching any file.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl


# Filenames look like: localhost_<model>_<YYYYmmdd_HHMMSS>.parquet
# The model name itself may contain colons (e.g. 'bge-m3:latest'), so we
# capture greedily up to the trailing _<timestamp>.parquet suffix.
PER_MODEL_RE = re.compile(
    r"^(?P<provider_id>localhost_.+)_(?P<ts>\d{8}_\d{6})\.parquet$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default="./out-data",
        help="Directory containing per-model parquet files, the cache, and the manifest.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen, do not write or overwrite anything.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the .bak copy of any pre-existing cache/manifest.",
    )
    return p.parse_args()


def find_per_model_parquets(out_dir: Path) -> Dict[str, Path]:
    """Group per-model parquets by provider_id, picking the most recent
    timestamp per provider when several exist.
    """
    by_provider: Dict[str, Path] = {}
    by_provider_ts: Dict[str, str] = {}
    for path in sorted(out_dir.glob("localhost_*.parquet")):
        m = PER_MODEL_RE.match(path.name)
        if not m:
            continue
        pid = m.group("provider_id")
        ts = m.group("ts")
        if pid not in by_provider_ts or ts > by_provider_ts[pid]:
            by_provider[pid] = path
            by_provider_ts[pid] = ts
    return by_provider


def load_parquet_as_doc_map(path: Path) -> Dict[str, List[float]]:
    """Read a per-model parquet (columns ``doc_id``, ``embedding``) into a dict."""
    df = pl.read_parquet(str(path))
    if not {"doc_id", "embedding"}.issubset(df.columns):
        raise ValueError(
            f"{path.name} doesn't have the expected (doc_id, embedding) columns; "
            f"got {df.columns}"
        )
    out: Dict[str, List[float]] = {}
    for row in df.iter_rows(named=True):
        out[str(row["doc_id"])] = list(row["embedding"])
    return out


def safe_backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    if not out_dir.is_dir():
        print(f"ERROR: {out_dir} is not a directory.", file=sys.stderr)
        return 2

    cache_path = out_dir / "embeddings_cache.pkl"
    manifest_path = out_dir / "embeddings_manifest.json"

    per_model = find_per_model_parquets(out_dir)
    if not per_model:
        print(
            f"No per-model parquet files found in {out_dir} matching "
            f"localhost_<model>_<timestamp>.parquet. Nothing to backfill.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(per_model)} per-model parquet file(s) in {out_dir}:")
    for pid, path in per_model.items():
        print(f"  - {pid:48s} <- {path.name}")

    # Load existing cache + manifest (if any) so we merge rather than overwrite.
    cache: Dict[str, Dict[str, List[float]]]
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
            print(f"Loaded existing cache with {len(cache)} provider(s).")
        except Exception as exc:
            print(
                f"WARNING: could not load {cache_path.name} ({exc!r}); "
                f"a fresh cache will be created.",
                file=sys.stderr,
            )
            cache = {}
    else:
        cache = {}

    manifest: dict
    if manifest_path.exists():
        try:
            with manifest_path.open() as f:
                manifest = json.load(f)
        except Exception as exc:
            print(
                f"WARNING: could not load {manifest_path.name} ({exc!r}); "
                f"a fresh manifest will be created.",
                file=sys.stderr,
            )
            manifest = {"providers": {}}
    else:
        manifest = {"providers": {}}
    manifest.setdefault("providers", {})

    # Merge each per-model parquet into the cache + manifest.
    updates: List[str] = []
    for pid, path in per_model.items():
        doc_map = load_parquet_as_doc_map(path)
        before = len(cache.get(pid, {}))
        cache.setdefault(pid, {}).update(doc_map)
        after = len(cache[pid])
        added = after - before
        manifest["providers"][pid] = {
            "file": path.name,
            "filepath": str(path),
            "completed_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "num_embeddings": after,
            "backfilled": True,
            "backfilled_at": datetime.now().isoformat(),
        }
        updates.append(
            f"  {pid:48s} -> {after:>6} embeddings (added {added:>5})"
        )

    print()
    print("Backfill summary:")
    for u in updates:
        print(u)

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return 0

    if not args.no_backup:
        for path in (cache_path, manifest_path):
            backup = safe_backup(path)
            if backup is not None:
                print(f"Backed up {path.name} -> {backup.name}")

    tmp_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_cache.open("wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp_cache, cache_path)

    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_manifest.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_manifest, manifest_path)

    print(f"\nWrote {cache_path}")
    print(f"Wrote {manifest_path}")
    print(
        f"\nNext step: re-run the SLURM job with any single model "
        f"(e.g. `sbatch slurm/run_embeddings_slurm.sh bge-m3`) and the "
        f"resulting `<date>_all_docs.parquet` will contain "
        f"`embeddings_<provider_id>` columns for every provider listed above."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
