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
├── README.md                     # this file
├── run_embeddings_slurm.sh       # SLURM batch script (entry point)
├── create_embeddings_ollama.py   # Python embedding script
├── wait_for_server.py            # server-readiness probe (used by the batch script)
└── ollama.sif                    # Apptainer image — created by you (see below, not in git)
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

---

## Input data

| What | Where |
|---|---|
| Corpus CSV | `in-data/corpus_20260111.csv` (repo root) |
| Expected columns | A text column (default name: `text`) and an ID column (default: first column) |

If your CSV uses different column names, pass `--text-column` and
`--id-column` to `create_embeddings_ollama.py` (see [Configuration](#configuration)).

---

## Choosing an embedding model

Pass the model name as an argument to `sbatch` (see [Submitting the job](#submitting-the-job)).
The model will be downloaded automatically on the first run and cached in
`$OLLAMA_MODELS` for subsequent runs.

| Model | Dims | Notes |
|---|---|---|
| `nomic-embed-text` | 768 | Default. Fast, good quality, English-biased. |
| `mxbai-embed-large` | 1024 | High quality, moderate speed. |
| `bge-m3` | 1024 | **Recommended for Spanish / Latin.** Truly multilingual, strong cross-lingual retrieval. |
| `snowflake-arctic-embed2` | 1024 | Strong multilingual support, competitive quality. |
| `multilingual-e5-large` | 1024 | Good multilingual baseline. |

Browse all available models at <https://ollama.com/search?c=embedding>.

---

## Configuration

### SLURM resource directives

Open `run_embeddings.slurm` and adjust the `#SBATCH` block at the top for
your cluster:

```bash
#SBATCH --time=24:00:00       # wall-clock limit — extend for very large corpora
#SBATCH --cpus-per-task=8     # CPUs for the Python worker (4–8 is usually fine)
#SBATCH --mem=32G             # RAM — increase if the CSV is very large
#SBATCH --gres=gpu:1          # request one GPU (required for Ollama)
# #SBATCH --partition=gpu     # uncomment and set the correct partition name
```

### Cluster-specific paths

At the top of `run_embeddings.slurm` there is a *Paths* section:

```bash
OLLAMA_MODELS="${PTMP:-${TMPDIR:-${HOME}/ollama_models}}/ollama_models_${SLURM_JOB_ID}"
```

This stores downloaded model weights on fast scratch storage (`$PTMP` at MPCDF)
so that they do not count against your home-directory quota.  If your cluster
uses a different scratch variable (e.g. `$SCRATCH`, `$WORK`), replace `$PTMP`
accordingly.

### Python script parameters

All options accepted by `create_embeddings_ollama.py` can be overridden in
the `uv run python …` call near the bottom of `run_embeddings.slurm`:

| Flag | Default | Description |
|---|---|---|
| `--input` | `in-data/corpus_20260111.csv` | Path to input CSV (relative to repo root) |
| `--text-column` | `text` | CSV column containing the passage text |
| `--id-column` | *(first column)* | CSV column to use as document ID |
| `--output-dir` | `out-data` | Directory for all output files |
| `--model` | `nomic-embed-text` | Ollama model name (overridden by the `sbatch` argument) |
| `--min-tokens` | `10` | Skip passages shorter than this many tokens |
| `--context-limit` | `8192` | Token limit; longer passages are truncated to this |
| `--concurrent-requests` | `5` | Parallel embedding API calls |
| `--batch-size` | `32` | Texts per API call |
| `--cache-save-interval` | `7200` | Seconds between incremental cache saves (default: 2 h) |
| `--retry-max` | `5` | Max retries per batch before marking it as failed |

Run `uv run python slurm/create_embeddings_ollama.py --help` locally to see
all options.

---

## Submitting the job

Submit from the **repo root** (or from the `slurm/` directory — both work):

```bash
# Default model (nomic-embed-text):
sbatch slurm/run_embeddings.slurm

# Specify a different model:
sbatch slurm/run_embeddings.slurm bge-m3
sbatch slurm/run_embeddings.slurm mxbai-embed-large
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

All output is written to `out-data/` in the repo root.
Files are prefixed with the current date (`YYYY-MM-DD`).

| File | Description |
|---|---|
| `YYYY-MM-DD_all_docs.parquet` | Document metadata (all CSV columns, filtered) |
| `YYYY-MM-DD_all_embeddings.parquet` | Per-document embedding vectors (columns: `doc_id`, `embedding`) |
| `YYYY-MM-DD_all_embeddings.pkl` | Numpy array + doc_id list, compatible with the notebook |
| `YYYY-MM-DD_all_embeddings.jsonl` | Vector-DB upload payload: `{id, vector, ...metadata}` per line |
| `YYYY-MM-DD_all_processing_metadata.json` | Run configuration (model, paths, parameters, timestamp) |
| `YYYY-MM-DD_all_embedding_statistics.json` | Success / failure / skip counts and processing time |
| `embeddings_cache.pkl` | **Incremental cache** — enables resume (see below) |
| `YYYY-MM-DD_HH-MM-SS_embeddings_ollama.log` | Full log of the Python script run |
| `slurm/logs/svsal-embeddings_<job_id>.out` | SLURM stdout (combined Python + shell output) |
| `slurm/logs/svsal-embeddings_<job_id>.err` | SLURM stderr |
| `slurm/logs/ollama_server_<job_id>.log` | Ollama server output (useful for GPU/model errors) |

---

## Resuming an interrupted job

The embedding script saves a rolling cache (`out-data/embeddings_cache.pkl`)
every `--cache-save-interval` seconds and again when it finishes.  If the job
is cancelled (e.g. it hits the wall-clock limit), simply re-submit:

```bash
sbatch slurm/run_embeddings.slurm bge-m3
```

On startup the script detects the existing cache, skips all already-embedded
documents, and only requests embeddings for the remaining ones.  No manual
intervention is required.

> **Tip:** For very large corpora, set `--time` to the maximum allowed by your
> cluster and rely on the cache for multi-day runs across several jobs.

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
| `apptainer: command not found` | Wrong or missing module | Change `module load apptainer/1.4.3` in the script |
| Server never comes up (timeout) | Container or GPU init too slow | Increase `--timeout` in `wait_for_server.py` call, or check `ollama_server_*.log` |
| GPU not used / slow inference | `--nv` / `--rocm` mismatch | Check GPU vendor; use `--rocm` for VIPER (AMD), `--nv` for Raven (NVIDIA) |
| `text column 'text' not found` | Different column name in CSV | Pass `--text-column <correct-name>` in the script call |
| Job killed before finishing | Wall-clock limit exceeded | Re-submit — the cache resumes automatically |
| Embeddings all zero / error in log | Model not supported by Ollama's `/v1/embeddings` endpoint | Choose a model from `ollama.com/search?c=embedding` |
