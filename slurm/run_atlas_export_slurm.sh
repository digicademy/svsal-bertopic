#!/bin/bash
# ============================================================
#  SVSAL Atlas Export — run BERTopic + UMAP per embedding provider
# ============================================================
#
# Loops `05-embeddings-export-atlas.py` over every `embeddings_*`
# column in the latest `*_all_docs.parquet` file, producing one
# Embedding Atlas dataset per model.
#
# Usage (submit from the repo root OR from the slurm/ directory):
#
#   sbatch slurm/run_atlas_export_slurm.sh [INPUT_FILE]
#
# INPUT_FILE is optional — if omitted, the most recently modified
# *_all_docs.parquet in out-data/ is used. Pass an explicit path to
# pick an older file or one from a different directory.
#
# This script has no GPU dependency (UMAP, HDBSCAN and BERTopic
# run on CPU here since embeddings are already computed) and runs
# independently of run_embeddings_slurm.sh.
#
# ------------- SLURM directives — adjust to your cluster --------------------
#SBATCH --mail-type=none
#SBATCH --mail-user=wagner@lhlt.mpg.de
#SBATCH --job-name=svsal-atlas-export
#SBATCH --export=ALL
#SBATCH --get-user-env=L

#SBATCH -D .                    # Initial working directory
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- VIPER default: CPU-only node, several hours walltime ---
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000

# --- VIPER alternative: smaller node for small corpora / debugging ---
# #SBATCH --time=2:00:00
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=32000


set -euo pipefail

# MPCDF injects ARGOS via LD_PRELOAD on the host, which spams errors
# every time we apptainer-exec into the container (where the .so files
# don't exist). Clear it for the duration of this job. SLURM accounting
# is unaffected; only in-container profiling via ARGOS is.
unset LD_PRELOAD

# ── Parameters ────────────────────────────────────────────────────────────────
INPUT_FILE="${1:-}"

# ── Paths ──────────────────────────────────────────────────────────────────────
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if   [ -f "${SUBMIT_DIR}/pyproject.toml" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../pyproject.toml" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: cannot locate repo root from '${SUBMIT_DIR}'." >&2
    exit 1
fi
SCRIPT_DIR="${REPO_ROOT}/slurm"
LOG_DIR="${REPO_ROOT}/logs"
OUTPUT_DIR="${REPO_ROOT}/out-data"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ── Pick the input parquet ────────────────────────────────────────────────────
if [ -z "${INPUT_FILE}" ]; then
    INPUT_FILE="$(ls -1t "${OUTPUT_DIR}"/*_all_docs.parquet 2>/dev/null | head -n1 || true)"
    if [ -z "${INPUT_FILE}" ]; then
        echo "ERROR: no *_all_docs.parquet found in ${OUTPUT_DIR}." >&2
        echo "Run run_embeddings_slurm.sh first, or pass an explicit INPUT_FILE." >&2
        exit 2
    fi
fi
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: input file does not exist: ${INPUT_FILE}" >&2
    exit 2
fi

echo "=================================================="
echo "  SVSAL Atlas Export"
echo "  Job  : ${SLURM_JOB_ID:-<interactive>}"
echo "  Node : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Input: ${INPUT_FILE}"
echo "  Time : $(date)"
echo "  Repo : ${REPO_ROOT}"
echo "=================================================="

cd "${REPO_ROOT}"

# ── Enumerate all embeddings_* columns in the input parquet ───────────────────
# A small inline Python helper because parquet column listing in pure bash is
# not worth the effort. Print one column name per line on stdout.
mapfile -t EMBEDDING_COLUMNS < <(
    uv run python - "${INPUT_FILE}" <<'PY'
import sys
import polars as pl
df = pl.scan_parquet(sys.argv[1]).collect_schema()
cols = [c for c in df.names() if c.startswith("embeddings_")]
for c in cols:
    print(c)
PY
)

if [ "${#EMBEDDING_COLUMNS[@]}" -eq 0 ]; then
    echo "ERROR: no embeddings_* columns found in ${INPUT_FILE}." >&2
    echo "Did the embedding job finish successfully?" >&2
    exit 3
fi

echo "Found ${#EMBEDDING_COLUMNS[@]} embedding column(s):"
for col in "${EMBEDDING_COLUMNS[@]}"; do
    echo "  - ${col}"
done

# ── Sanitiser: build a portable filename component from an embedding column ──
# Strips the leading 'embeddings_' prefix; drops a trailing ':latest' tag (the
# most common case); replaces any remaining ':' with '-' so the result is safe
# on Linux, macOS, and Windows-mounted filesystems.
sanitize_provider() {
    local col="$1"
    local provider="${col#embeddings_}"
    provider="${provider%:latest}"
    provider="${provider//:/-}"
    echo "${provider}"
}

# ── Loop over providers; continue past per-model failures ────────────────────
FAILED=()
SUCCEEDED=()
for col in "${EMBEDDING_COLUMNS[@]}"; do
    provider="$(sanitize_provider "${col}")"
    output_file="${OUTPUT_DIR}/embeddings_atlas_${provider}.parquet"
    echo ""
    echo "─── Atlas export: ${col} ─────────────────────────────"
    echo "    output: ${output_file}"
    if uv run python "${REPO_ROOT}/05-embeddings-export-atlas.py" \
            --input-file       "${INPUT_FILE}" \
            --embedding-column "${col}" \
            --output-file      "${output_file}"; then
        SUCCEEDED+=("${col}")
    else
        rc=$?
        echo "    ⚠  exporter exited with status ${rc} for column '${col}'." >&2
        echo "    Continuing with the remaining columns." >&2
        FAILED+=("${col}")
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Atlas Export summary"
echo "  Succeeded: ${#SUCCEEDED[@]} / ${#EMBEDDING_COLUMNS[@]}"
for c in "${SUCCEEDED[@]}"; do echo "    ✔ ${c}"; done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "  Failed   : ${#FAILED[@]}"
    for c in "${FAILED[@]}"; do echo "    ✘ ${c}"; done
fi
echo "=================================================="

# Non-zero exit if every provider failed; partial success is OK.
if [ "${#SUCCEEDED[@]}" -eq 0 ]; then
    exit 1
fi
