#!/bin/bash
# ============================================================
#  SVSAL Embedding Creation — run Ollama on an HPC GPU node
# ============================================================
#
# Usage (submit from the repo root OR from the slurm/ directory):
#
#   sbatch slurm/run_embeddings_slurm.sh [MODEL ...]
#
# Zero or more MODEL arguments. Each MODEL is an Ollama embedding model
# name; two special tokens are recognised by the Python script:
#
#   auto   (default if no MODEL given) -- pick the first locally-available
#                                         embedding model, alphabetically
#   all                                -- run every locally-available
#                                         embedding model in parallel
#
# Examples:
#
#   sbatch slurm/run_embeddings_slurm.sh
#   sbatch slurm/run_embeddings_slurm.sh all
#   sbatch slurm/run_embeddings_slurm.sh bge-m3
#   sbatch slurm/run_embeddings_slurm.sh bge-m3 nomic-embed-text mxbai-embed-large
#
# Good embedding models for Spanish / Latin texts (check https://ollama.com/search?c=embedding):
#   nomic-embed-text          768 dims   fast, solid quality
#   mxbai-embed-large         1024 dims  high quality
#   bge-m3                    1024 dims  multilingual -- best for non-English corpora
#   snowflake-arctic-embed2   1024 dims  strong multilingual support
#
# ------------- Prerequisites (run ONCE before submitting) --------------------
#
#   module load apptainer/1.4.3
#   apptainer pull slurm/ollama.sif docker://ollama/ollama
#
#   # On VIPER (AMD/ROCm) pull with the rocm tag instead:
#   #   apptainer pull slurm/ollama.sif docker://ollama/ollama:rocm
#   # and change GPU_FLAG below from "--nv" to "--rocm".
#
# ------------- SLURM directives — adjust to your cluster --------------------
#SBATCH --mail-type=none
#SBATCH --mail-user=wagner@lhlt.mpg.de
#SBATCH --job-name=svsal-embeddings
#SBATCH --export=ALL
#SBATCH --get-user-env=L

#SBATCH -D .                    # Initial working directory
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --constraint="apu"

# --- Change the following for testing the workflow/GPU setup ---
#SBATCH --time=22:00:00
#SBATCH --partition=apu
# #SBATCH --partition=apudev      # apudev: for testing, 1 node with 2 MI300, 15 min. walltime

# --- VIPER default case: use a single APU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110000

# --- VIPER alternative case: two APUs on a shared node ---
# #SBATCH --gres=gpu:2
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=32
# #SBATCH --mem=220000

# --- DAIS: H200 on a shared node ---
# #SBATCH --partition="gpu1"
# #SBATCH --gres=gpu:h200:1
# #SBATCH --cpus-per-task=12
# #SBATCH --mem=250000


set -euo pipefail

# MPCDF injects ARGOS via LD_PRELOAD on the host, which spams errors
# every time we apptainer-exec into the container (where the .so files
# don't exist). Clear it for the duration of this job. SLURM accounting
# is unaffected; only in-container profiling via ARGOS is.
unset LD_PRELOAD

# ── Parameters ────────────────────────────────────────────────────────────────
# All positional args after the script name are treated as Ollama embedding
# model names. Special tokens 'auto' (default) and 'all' are forwarded to the
# Python script for server-side discovery (see slurm/README.md).
#
#   sbatch slurm/run_embeddings_slurm.sh                  # auto: first available
#   sbatch slurm/run_embeddings_slurm.sh all              # every cached embedding model
#   sbatch slurm/run_embeddings_slurm.sh bge-m3           # one specific
#   sbatch slurm/run_embeddings_slurm.sh bge-m3 nomic-embed-text mxbai-embed-large
if [ "$#" -eq 0 ]; then
    MODELS=("auto")
else
    MODELS=("$@")
fi
OLLAMA_PORT=11434
OLLAMA_URL="http://localhost:${OLLAMA_PORT}"

# GPU flag: --nv for NVIDIA (Raven), --rocm for AMD (VIPER)
GPU_FLAG="--rocm"

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
INPUT_FILE="${REPO_ROOT}/in-data/corpus_20260621.csv"
INPUT_TEXT_COLUMN="content"
INPUT_ID_COLUMN="url"
SCRIPT_DIR="${REPO_ROOT}/slurm"
OLLAMA_SIF="${SCRIPT_DIR}/ollama.sif"
LOG_DIR="${REPO_ROOT}/logs"
OUTPUT_DIR="${REPO_ROOT}/out-data"

CONCURRENT_REQ=5
BATCH_SIZE=32

# Prefer an externally-supplied OLLAMA_MODELS; otherwise build it from
# the canonical scratch directory layout, since $PTMP is not always
# exported into SLURM jobs on this cluster.
if [ -z "${OLLAMA_MODELS:-}" ]; then
    if   [ -d "/ptmp/${USER}" ];   then OLLAMA_MODELS="/ptmp/${USER}/ollama_models"
    elif [ -d "/scratch/${USER}" ];then OLLAMA_MODELS="/scratch/${USER}/ollama_models"
    elif [ -n "${TMPDIR:-}" ];     then OLLAMA_MODELS="${TMPDIR}/ollama_models"
    else                                OLLAMA_MODELS="${HOME}/ollama_models"
    fi
fi
OLLAMA_SERVER_LOG="${LOG_DIR}/ollama_server_${SLURM_JOB_ID:-local}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${OLLAMA_MODELS}"

echo "=================================================="
echo "  SVSAL Embedding Creation"
echo "  Job   : ${SLURM_JOB_ID:-<interactive>}"
echo "  Node  : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Models: ${MODELS[*]}"
echo "  Time  : $(date)"
echo "  Repo  : ${REPO_ROOT}"
echo "=================================================="

# ── System modules ────────────────────────────────────────────────────────────
module load apptainer/1.4.3

# ── Check for uv ──────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "ERROR: 'uv' not found in PATH."
    echo "Install it (https://docs.astral.sh/uv/getting-started/installation/) or"
    echo "activate the project virtualenv manually and replace 'uv run python' with 'python'."
    exit 1
fi

# ── Ensure the Ollama container image is present ──────────────────────────────
if [ ! -f "${OLLAMA_SIF}" ]; then
    echo "INFO: ${OLLAMA_SIF} not found — pulling from Docker Hub …"
    echo "      For VIPER (AMD/ROCm) use: docker://ollama/ollama:rocm"
    apptainer pull "${OLLAMA_SIF}" docker://ollama/ollama
fi

# ── Start Ollama server inside the container ───────────────────────────────────
echo "Starting Ollama server on port ${OLLAMA_PORT} …"
apptainer run ${GPU_FLAG} \
    --env "OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}" \
    --env "OLLAMA_MODELS=${OLLAMA_MODELS}" \
    --bind "${OLLAMA_MODELS}:${OLLAMA_MODELS}" \
    "${OLLAMA_SIF}" serve \
    > "${OLLAMA_SERVER_LOG}" 2>&1 &

OLLAMA_PID=$!
echo "Ollama server PID: ${OLLAMA_PID}"

# Kill the background server when the SLURM job ends (success or failure)
cleanup() {
    echo "Stopping Ollama server (PID ${OLLAMA_PID}) …"
    kill "${OLLAMA_PID}" 2>/dev/null || true
    wait "${OLLAMA_PID}" 2>/dev/null || true
    echo "Ollama server stopped."
}
trap cleanup EXIT

# ── Wait for server to accept connections ─────────────────────────────────────
echo "Waiting for Ollama server to be ready …"
cd "${REPO_ROOT}"
uv run python "${SCRIPT_DIR}/wait_for_server.py" --url "${OLLAMA_URL}"

# ── Ensure each requested model is available ──────────────────────────────
# `ollama show` only checks the local manifest in OLLAMA_MODELS and never
# touches the network, so this is safe to run even on compute nodes without
# internet access. The network-dependent `ollama pull` is only invoked as a
# fallback when a model is missing -- on offline nodes, pre-download each
# model first (see README: "Pre-download models").
#
# Special tokens 'auto' and 'all' are resolved by the Python script against
# the live server, so we leave them alone here.
for MODEL in "${MODELS[@]}"; do
    if [ "${MODEL}" = "auto" ] || [ "${MODEL}" = "all" ]; then
        echo "Model spec '${MODEL}' will be resolved by the Python script at runtime."
        continue
    fi
    echo "Checking whether model '${MODEL}' is already cached in ${OLLAMA_MODELS} ..."
    if apptainer exec ${GPU_FLAG} \
        --env "OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}" \
        --env "OLLAMA_MODELS=${OLLAMA_MODELS}" \
        --bind "${OLLAMA_MODELS}:${OLLAMA_MODELS}" \
        "${OLLAMA_SIF}" ollama show "${MODEL}" >/dev/null 2>&1; then
        echo "Model '${MODEL}' already present -- skipping pull."
    else
        echo "Model '${MODEL}' not cached locally -- attempting to pull it (requires internet) ..."
        apptainer exec ${GPU_FLAG} \
            --env "OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}" \
            --env "OLLAMA_MODELS=${OLLAMA_MODELS}" \
            --bind "${OLLAMA_MODELS}:${OLLAMA_MODELS}" \
            "${OLLAMA_SIF}" \
            ollama pull "${MODEL}"
    fi
    echo "Model '${MODEL}' is ready."
done

# ── Build the --model arguments for the Python script ─────────────────────
MODEL_ARGS=()
for MODEL in "${MODELS[@]}"; do
    MODEL_ARGS+=(--model "${MODEL}")
done

# ── Create embeddings ─────────────────────────────────────────────────────────
echo "Starting embedding creation …"
uv run python "${SCRIPT_DIR}/create_embeddings_ollama.py" \
    --input              "${INPUT_FILE}" \
    --text-column        "${INPUT_TEXT_COLUMN}" \
    --id-column          "${INPUT_ID_COLUMN}" \
    --output-dir         "${OUTPUT_DIR}" \
    "${MODEL_ARGS[@]}" \
    --url                "${OLLAMA_URL}/v1" \
    --concurrent-requests ${CONCURRENT_REQ} \
    --batch-size         ${BATCH_SIZE}

EXIT_CODE=$?
if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "ERROR: Embedding script exited with code ${EXIT_CODE}."
    exit "${EXIT_CODE}"
fi

echo "=================================================="
echo "  Embedding creation completed successfully."
echo "  Output: ${OUTPUT_DIR}"
echo "  Time  : $(date)"
echo "=================================================="
