# HPC/SLURM Embedding Pipeline

This directory contains everything needed to create embeddings for the
Salamanca corpus on an HPC cluster using [Ollama](https://ollama.com/) as a
locally-hosted embedding server.

The pipeline is an alternative to running the interactive notebook
`01-embeddings-create.*.ipynb` and is intended for situations where the
number of passages is too large to embed via rate-limited online providers
within a reasonable time. It produces the same output files as the notebook
so that all downstream notebooks (`02-…`, `03-…`, `04-…`) work without
modification.

---

## Directory layout

```
slurm/
├── README.md                                       # this file
├── run_embeddings_slurm.sh                         # SLURM batch script — create embeddings (entry point)
├── run_atlas_export_slurm.sh                       # SLURM batch script — Embedding Atlas export (CPU job)
├── create_embeddings_ollama.py                     # Python embedding script
├── wait_for_server.py                              # server-readiness probe (used by the batch script)
├── backfill_cache_from_per_model_parquets.py       # one-shot utility for cross-run accumulation
└── ollama.sif                                      # Apptainer image — created by you (see below, not in git)
```

> `ollama.sif` is **not** tracked by git (see `.gitignore`).

---

## Prerequisites

### 1 — Apptainer module

The batch script assumes `apptainer/1.4.3` is available as a module.  
If your cluster uses a different version, edit the `module load` line in
`run_embeddings_slurm.sh`.

### 2 — `uv` package manager

The Python scripts are run via `uv run python …` so that the project
virtual environment is created automatically from `pyproject.toml` /
`uv.lock`.

Check whether `uv` is available in your login environment:

```bash
uv --version
```

If it is not, install it into your home directory (no root required):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then add `$HOME/.local/bin` to your `PATH` (the installer will tell you how).

### 3 — Pull the Ollama container image (once per cluster)

Run this **once** from the repo root (not as a SLURM job):

> ⚠️ Make sure you are in the **repo root**, not already inside `slurm/`.
> Running the command below from inside `slurm/` resolves the destination to
> `slurm/slurm/ollama.sif` and fails with
> `could not open temporary file for copy ... no such file or directory`.
> If you are already inside `slurm/`, drop the `slurm/` prefix instead
> (`apptainer pull ollama.sif docker://...`).

```bash
module load apptainer/1.4.3
apptainer pull slurm/ollama.sif docker://ollama/ollama
```

**On VIPER (AMD/ROCm GPUs)** pull the ROCm-enabled image instead:

```bash
apptainer pull slurm/ollama.sif docker://ollama/ollama:rocm
```

Then open `run_embeddings_slurm.sh` and change `GPU_FLAG="--nv"` to
`GPU_FLAG="--rocm"`.

The image file is ~1.5 GB and will be stored at `slurm/ollama.sif`.
Pulling only needs to be repeated when you want to upgrade Ollama.

### 4 — Pre-download models (only if your compute nodes have no internet access)

`run_embeddings_slurm.sh` checks whether the requested model is already
cached and, if not, tries to `ollama pull` it **inside the SLURM job** —
i.e. on the compute node. If your compute partition has no outbound
internet access (true for VIPER's `apu`/`apudev` partitions), that pull
will fail. Download the model **once, from the login node**, into the same
persistent `OLLAMA_MODELS` directory the job will later use:

```bash
module load apptainer/1.4.3

# Must match the OLLAMA_MODELS path used in run_embeddings_slurm.sh
export OLLAMA_MODELS="${PTMP:-${TMPDIR:-$HOME/ollama_models}}/ollama_models"
mkdir -p "${OLLAMA_MODELS}"

# Start the server in the background (the login node has internet)
apptainer run --rocm \
    --env "OLLAMA_HOST=127.0.0.1:11434" \
    --env "OLLAMA_MODELS=${OLLAMA_MODELS}" \
    --bind "${OLLAMA_MODELS}:${OLLAMA_MODELS}" \
    slurm/ollama.sif serve &
OLLAMA_PID=$!
sleep 5   # give the server a moment to come up

# Pull every model you might want to use later — repeat as needed
apptainer exec --rocm \
    --env "OLLAMA_HOST=127.0.0.1:11434" \
    --env "OLLAMA_MODELS=${OLLAMA_MODELS}" \
    --bind "${OLLAMA_MODELS}:${OLLAMA_MODELS}" \
    slurm/ollama.sif ollama pull nomic-embed-text

apptainer exec --rocm \
    --env "OLLAMA_HOST=127.0.0.1:11434" \
    --env "OLLAMA_MODELS=${OLLAMA_MODELS}" \
    --bind "${OLLAMA_MODELS}:${OLLAMA_MODELS}" \
    slurm/ollama.sif ollama pull bge-m3

kill "${OLLAMA_PID}"
```

Once a model is cached in `${OLLAMA_MODELS}`, `run_embeddings_slurm.sh`
detects it automatically (via `ollama show`, which only reads the local
manifest) and skips the network-dependent pull step inside the job. See the
[Ollama FAQ](https://docs.ollama.com/faq) for details on `OLLAMA_MODELS` and
why `ollama pull` always needs network access, even for an already-known
model name.

> **Tip:** Pre-download as many embedding models as you want to experiment
> with — then submit jobs with `sbatch slurm/run_embeddings_slurm.sh all`
> to embed against every one of them in a single run, or
> `sbatch slurm/run_embeddings_slurm.sh` (no argument) to let the script
> pick the first one. See *Choosing one or more embedding models* below.

> On NVIDIA login nodes, replace `--rocm` with `--nv`, or drop the flag
> entirely — downloading a model doesn't require a GPU.

---

## Input data

| What | Where |
|---|---|
| Corpus CSV | `in-data/corpus_20260111.csv` (repo root) |
| Expected columns | A text column (default name: `text`) and an ID column (default: first column) |

If your CSV uses different column names, pass `--text-column` and
`--id-column` to `create_embeddings_ollama.py` (see [Configuration](#configuration)).

---

## Choosing one or more embedding models

The script can embed the same corpus with **several models in one run**,
in parallel against the same Ollama server. Pass the model names as
positional arguments to `sbatch`; the script also recognises two special
tokens that trigger server-side discovery:

| Argument | Effect |
|---|---|
| *(no argument)* | Equivalent to `auto`. |
| `auto` | Use the alphabetically first locally-available embedding model. Useful when you don't care which model is picked, or when you've only pulled one. |
| `all` | Run every locally-available embedding model concurrently. |
| `bge-m3` (etc.) | Use the named model. Repeat for several specific models. |

Discovery uses Ollama's `GET /api/tags` (locally cached models) and `POST
/api/show` (per-model `capabilities` array); a model is treated as an
embedding model iff its capabilities include `"embedding"`. The capability
flag was [added to Ollama in PR #10066](https://github.com/ollama/ollama/pull/10066)
and is derived from the GGUF `pooling_type` metadata.

You can also ask the script what's available without launching a job:

```bash
uv run python slurm/create_embeddings_ollama.py --list-models \
    --url http://localhost:11434/v1
```

Models worth considering for the Salamanca corpus:

| Model | Dims | Notes |
|---|---|---|
| `nomic-embed-text` | 768 | Fast, good quality, English-biased. |
| `mxbai-embed-large` | 1024 | High quality, moderate speed. |
| `bge-m3` | 1024 | **Recommended for Spanish / Latin.** Truly multilingual, strong cross-lingual retrieval. |
| `snowflake-arctic-embed2` | 1024 | Strong multilingual support, competitive quality. |
| `multilingual-e5-large` | 1024 | Good multilingual baseline. |

Browse all available models at <https://ollama.com/search?c=embedding>.

---

## Configuration

### SLURM resource directives

Open `run_embeddings_slurm.sh` and adjust the `#SBATCH` block at the top for
your cluster:

```bash
#SBATCH --time=24:00:00       # wall-clock limit — extend for very large corpora
#SBATCH --cpus-per-task=8     # CPUs for the Python worker (4–8 is usually fine)
#SBATCH --mem=32G             # RAM — increase if the CSV is very large
#SBATCH --gres=gpu:1          # request one GPU (required for Ollama)
# #SBATCH --partition=gpu     # uncomment and set the correct partition name
```

### Cluster-specific paths

At the top of `run_embeddings_slurm.sh` there is a *Paths* section:

```bash
OLLAMA_MODELS="${OLLAMA_MODELS:-${PTMP:-${TMPDIR:-${HOME}/ollama_models}}/ollama_models}"
```

This stores downloaded model weights on fast scratch storage (`$PTMP` at MPCDF)
so that they do not count against your home-directory quota. If your cluster
uses a different scratch variable (e.g. `$SCRATCH`, `$WORK`), replace `$PTMP`
accordingly. The path is shared across jobs, so a model downloaded once is
reused by every later run.

### Python script parameters

All options accepted by `create_embeddings_ollama.py` can be overridden in
the `uv run python …` call near the bottom of `run_embeddings_slurm.sh`:

| Flag | Default | Description |
|---|---|---|
| `--input` | `in-data/corpus_20260111.csv` | Path to input CSV (relative to repo root) |
| `--text-column` | `text` | CSV column containing the passage text |
| `--id-column` | *(first column)* | CSV column to use as document ID |
| `--output-dir` | `out-data` | Directory for all output files |
| `--model` | `auto` | Ollama embedding model. Repeatable; accepts `auto` / `all`. Forwarded from the `sbatch` positional arguments. |
| `--min-tokens` | `10` | Skip passages shorter than this many tokens (uses tiktoken `cl100k_base` as a rough heuristic — not any model's exact tokenizer). |
| `--context-limit` | *(unset → model-native)* | Optional **cap** on tokens per text. If unset, each model's native `context_length` (queried from `/api/show`) is used. If set, the effective limit per model is `min(model_native, --context-limit)`. |
| `--concurrent-requests` | `5` | Parallel embedding API calls *per model* |
| `--batch-size` | `32` | Texts per API call |
| `--cache-save-interval` | `7200` | Seconds between incremental cache saves (default: 2 h) |
| `--retry-max` | `5` | Max retries per batch before marking it as failed |
| `--list-models` | – | Print discovered embedding models (with native context, output dim, family) and exit. |

Run `uv run python slurm/create_embeddings_ollama.py --help` locally to see
all options.

### Per-model configuration discovered from Ollama

For each requested model the script queries Ollama's `POST /api/show` and
records the following in the per-provider section of the dated
`*_all_processing_metadata.json` output:

| Field | Source in `/api/show` |
|---|---|
| `native_context_length` | First `model_info` key ending in `.context_length` (e.g. `bert.context_length`) — the model's GGUF-declared max input. |
| `embedding_length` | First `model_info` key ending in `.embedding_length` — the output vector dimension. |
| `family`, `parameter_size`, `quantization_level` | `details.{family,parameter_size,quantization_level}` |
| `effective_context_limit` | `min(native_context_length, --context-limit if set)`. Used for per-model pre-truncation at batch time. |
| `user_context_cap` | The value of `--context-limit` at run time (may be `null` if you didn't set it). |

Per-batch text truncation is performed per model using the *effective*
limit, so a model with a 512-token native context isn't sent 8192-token
inputs that would silently get clipped server-side. As a belt-and-braces
measure Ollama's `/v1/embeddings` keeps its default `truncate: true`, so
any miscount in our heuristic tokenizer still produces a correct
embedding rather than an error.

---

## Submitting the job

Submit from the **repo root** (or from the `slurm/` directory — both work):

```bash
# Default: use the first locally-available embedding model
sbatch slurm/run_embeddings_slurm.sh

# Use every embedding model that's been pre-downloaded
sbatch slurm/run_embeddings_slurm.sh all

# One specific model
sbatch slurm/run_embeddings_slurm.sh bge-m3

# Several specific models, embedded concurrently against the same server
sbatch slurm/run_embeddings_slurm.sh bge-m3 nomic-embed-text mxbai-embed-large
```

Monitor the job:

```bash
squeue --me
# or
squeue -j <job_id>
```

Watch the log in real time:

```bash
tail -f slurm/logs/svsal-embeddings_<job_id>.out
```

Cancel the job:

```bash
scancel <job_id>
```

---

## Output files

All output is written to `out-data/` in the repo root. Dated final
outputs are prefixed with the current date (`YYYY-MM-DD`); per-model
intermediates use the provider identifier `localhost_<model>` and a
timestamp.

The shapes match the interactive notebook `01-embeddings-create.*.ipynb`
exactly, so the downstream notebooks (`02-…`, `03-…`, `04-…`) consume
these files without modification.

| File | Description |
|---|---|
| `YYYY-MM-DD_all_docs.parquet` | Docs DataFrame with one `embeddings_localhost_<model>` column per requested model |
| `YYYY-MM-DD_all_docs.pkl` | Pickle of the same DataFrame |
| `YYYY-MM-DD_all_docs.csv` | Same docs without embedding columns (CSV-friendly) |
| `YYYY-MM-DD_all_embeddings.pkl` | Nested-dict pickle: `{localhost_<model>: {doc_id: vector}}` |
| `YYYY-MM-DD_all_embeddings.parquet` | Same nested shape, parquet (one row per provider) |
| `YYYY-MM-DD_all_embeddings.jsonl` | One line per `(doc_id, model)`: `{id, model, provider_id, vector, ...metadata}` |
| `YYYY-MM-DD_all_processing_metadata.json` | Per-provider run config (model, paths, parameters, timestamps) |
| `YYYY-MM-DD_all_embedding_statistics.json` | Per-provider success / failure / skip counts and processing time |
| `localhost_<model>_<YYYYmmdd_HHMMSS>.parquet` | Per-model parquet, written as each model finishes |
| `embeddings_cache.pkl` | **Incremental cache** — enables resume (see below). Shape: `{localhost_<model>: {doc_id: vector}}`. |
| `embeddings_manifest.json` | Tracks which providers have completed. Used for whole-model skip on re-runs. |
| `YYYY-MM-DD_HH-MM-SS_embeddings_ollama.log` | Full log of the Python script run |
| `slurm/logs/svsal-embeddings_<job_id>.out` | SLURM stdout (combined Python + shell output) |
| `slurm/logs/svsal-embeddings_<job_id>.err` | SLURM stderr |
| `slurm/logs/ollama_server_<job_id>.log` | Ollama server output (useful for GPU/model errors) |

---

## Resuming an interrupted job

The embedding script saves a rolling cache (`out-data/embeddings_cache.pkl`)
every `--cache-save-interval` seconds and again when it finishes. Two levels
of skip make resume robust:

1. **Whole-model skip** — if a model is recorded in `embeddings_manifest.json`
   as having completed, it is skipped entirely on the next run.
2. **Per-doc skip** — for any model not yet completed, already-cached
   documents are skipped and only the missing ones are embedded.

So if a job is cancelled (e.g. it hits the wall-clock limit), simply
re-submit with the same model list:

```bash
sbatch slurm/run_embeddings_slurm.sh bge-m3 nomic-embed-text
```

No manual intervention is required. You can also add a model on the
re-submission — already-completed models are skipped, the new model runs
from scratch:

```bash
# Originally
sbatch slurm/run_embeddings_slurm.sh bge-m3
# Later, add another model — bge-m3 is skipped (manifest), mxbai-embed-large runs
sbatch slurm/run_embeddings_slurm.sh bge-m3 mxbai-embed-large
```

> **Tip:** For very large corpora, set `--time` to the maximum allowed by your
> cluster and rely on the cache for multi-day runs across several jobs.

---

## Cross-run accumulation across models

The dated output files (`<date>_all_docs.parquet`, `<date>_all_embeddings.*`,
`<date>_all_processing_metadata.json`, `<date>_all_embedding_statistics.json`)
reflect the **full set of providers in the cache**, not only the models from
the current run. So a sequence like:

```bash
sbatch slurm/run_embeddings_slurm.sh bge-m3
# ... wait for completion ...
sbatch slurm/run_embeddings_slurm.sh nomic-embed-text
# ... wait ...
sbatch slurm/run_embeddings_slurm.sh mxbai-embed-large
```

ends with a docs parquet whose columns include all three
`embeddings_localhost_*` providers. The single-model SLURM run does the
right thing for incremental work; you don't have to merge anything by hand.

A few notes:

- The dated outputs **overwrite same-day predecessors**. So three sequential
  runs on the same day all write to `out-data/2026-06-25_all_docs.parquet`;
  each rewrite is cumulative, so the file at the end contains all providers.
  Same-day re-runs are fine — they don't lose data, they just rewrite the
  same files cumulatively.
- For providers from earlier runs, the per-provider sub-dict in
  `<date>_all_processing_metadata.json` has `from_previous_run: true` and
  the run-specific fields (concurrency, batch size, native context, etc.)
  are `null`, since that information isn't recoverable. The manifest entry
  contributes `completed_at`, `file`, and `num_embeddings`.

### Backfilling the cache from existing per-model parquets

If you've run the SLURM job several times *before* cross-run accumulation
was added (or your cache was deleted / lost), but you still have the per-
model `localhost_<model>_<timestamp>.parquet` snapshots on disk, you can
rebuild the cache and manifest in one shot:

```bash
uv run python slurm/backfill_cache_from_per_model_parquets.py \
    --output-dir out-data
```

The script picks the most recent parquet per provider, writes a merged
`embeddings_cache.pkl` and `embeddings_manifest.json`, and backs up any
pre-existing copies. Add `--dry-run` to preview without writing.

After backfilling, the next SLURM job (even with a single new model) will
produce a docs parquet that includes every backfilled provider.

---

## Running two projects in parallel

The SLURM script honours these environment variables, which lets you submit
several independent runs against different corpora without editing the script
each time:

| Variable | Default | Purpose |
|---|---|---|
| `INPUT_FILE` | `in-data/corpus_20260621.csv` | Path to the corpus CSV |
| `OUTPUT_DIR` | `out-data/` | Where all output files go (cache, manifest, dated outputs, per-model parquets) |
| `INPUT_TEXT_COLUMN` | `content` | CSV column containing the passage text |
| `INPUT_ID_COLUMN` | `url` | CSV column containing the document ID |
| `OLLAMA_PORT` | `11434` | Port the in-job Ollama server binds to |
| `OLLAMA_MODELS` | `/ptmp/$USER/ollama_models` | Ollama model cache directory |

Combine with `sbatch --export=ALL` so the overrides reach the job:

```bash
INPUT_FILE=$PWD/in-data/projectA.csv \
OUTPUT_DIR=$PWD/out-data/projectA \
    sbatch --export=ALL slurm/run_embeddings_slurm.sh all

INPUT_FILE=$PWD/in-data/projectB.csv \
OUTPUT_DIR=$PWD/out-data/projectB \
OLLAMA_PORT=11435 \
    sbatch --export=ALL slurm/run_embeddings_slurm.sh all
```

Each project gets its own cache, manifest, and dated outputs. The `OLLAMA_PORT`
override is only needed if SLURM happens to land both jobs on the same node
(rare, since each requests `--gres=gpu:1`, but cheap insurance).

---

## Retrieving results from the cluster

If you work on a local machine, copy the output directory back using `rsync`
(adjust paths to your cluster's hostname and scratch layout):

```bash
rsync -avz --progress \
    <user>@raven.mpcdf.mpg.de:/path/to/svsal-bertopic/out-data/ \
    ./out-data/
```

The `embeddings_cache.pkl` file can be left on the cluster to allow further
resume runs; only the dated parquet / pickle / jsonl files are needed locally
for the downstream notebooks.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: 'uv' not found in PATH` | `uv` not installed or not on `PATH` | Install uv (see [Prerequisites](#prerequisites)) |
| `apptainer pull` fails with `could not open temporary file for copy ... no such file or directory` | Command run from inside `slurm/` with a `slurm/...`-prefixed destination path | `cd` to the repo root first, or drop the `slurm/` prefix if already inside that directory |
| `apptainer: command not found` | Wrong or missing module | Change `module load apptainer/1.4.3` in the script |
| Server never comes up (timeout) | Container or GPU init too slow | Increase `--timeout` in `wait_for_server.py` call, or check `ollama_server_*.log` |
| GPU not used / slow inference | `--nv` / `--rocm` mismatch | Check GPU vendor; use `--rocm` for VIPER (AMD), `--nv` for Raven (NVIDIA) |
| `text column 'text' not found` | Different column name in CSV | Pass `--text-column <correct-name>` in the script call |
| Job killed before finishing | Wall-clock limit exceeded | Re-submit — the cache resumes automatically |
| Embeddings all zero / error in log | Model not supported by Ollama's `/v1/embeddings` endpoint | Choose a model from `ollama.com/search?c=embedding` |

---

## Exporting to Embedding Atlas (optional, separate job)

Once a SLURM embedding job has produced a `<date>_all_docs.parquet` file
in `out-data/`, you can run [Embedding Atlas](https://github.com/apple/embedding-atlas)
exports for every embedding column it contains via a second SLURM script:

```bash
sbatch slurm/run_atlas_export_slurm.sh
# or, to use a specific docs parquet:
sbatch slurm/run_atlas_export_slurm.sh out-data/2026-06-20_all_docs.parquet
```

The script:

1. Finds the most recent `*_all_docs.parquet` in `out-data/` (or uses the path
   you pass as the first positional argument).
2. Lists every column starting with `embeddings_` and loops the
   `05-embeddings-export-atlas.py` exporter once per column.
3. Builds a Windows-safe output filename per provider by stripping the
   `:latest` Ollama tag and replacing any other `:` with `-`. So
   `embeddings_localhost_bge-m3:latest` becomes
   `out-data/embeddings_atlas_localhost_bge-m3.parquet`, and
   `embeddings_localhost_some-model:q4_0` becomes
   `out-data/embeddings_atlas_localhost_some-model-q4_0.parquet`.
4. Continues past per-provider failures (so a BERTopic issue on one
   model doesn't abandon the others) and prints a final pass/fail tally.

This job is **CPU-only** — `05-embeddings-export-atlas.py` runs UMAP +
HDBSCAN + BERTopic on pre-computed embeddings, no GPU needed. The
defaults at the top of `run_atlas_export_slurm.sh` request 16 CPUs and
64 GB RAM with a 12 h wall-clock limit; tune these to your corpus size.

> The atlas exporter writes its outputs (`embeddings_atlas_<provider>.parquet`
> and `embeddings_atlas_<provider>_topics.parquet`) alongside the docs
> parquet in `out-data/`. Pass `--launch-atlas` only when running locally
> on a workstation, not inside the SLURM job — the Atlas UI is interactive.
