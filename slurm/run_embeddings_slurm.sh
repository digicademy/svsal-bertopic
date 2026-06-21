#!/bin/bash
# ============================================================
#  SVSAL Embedding Creation — run Ollama on an HPC GPU node
# ============================================================
#
# Usage (submit from the repo root OR from the slurm/ directory):
#
#   sbatch slurm/run_embeddings.slurm [model]
#
# The MODEL argument is optional; default: nomic-embed-text
#
# Good embedding models for Spanish / Latin texts (check https://ollama.com/search?c=embedding):
#   nomic-embed-text          768 dims   fast, solid quality
#   mxbai-embed-large         1024 dims  high quality
#   bge-m3                    1024 dims  multilingual — best for non-English corpora
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

#SBATCH -D .                    # Initial working directory
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --constraint="apu"

# --- Change the following for testing the workflow/GPU setup ---
#SBATCH --time=22:00:00
# #SBATCH --partition=apu
#SBATCH --partition=apudev      # apudev: for testing, 1 node with 2 MI300, 15 min. walltime

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

# ── Parameters ────────────────────────────────────────────────────────────────
MODEL="${1:-nomic-embed-text}"
OLLAMA_PORT=11434
OLLAMA_URL="http://localhost:${OLLAMA_PORT}"

# GPU flag: --nv for NVIDIA (Raven), --rocm for AMD (VIPER)
GPU_FLAG="--rocm"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
OLLAMA_SIF="${SCRIPT_DIR}/ollama.sif"
LOG_DIR="${SCRIPT_DIR}/logs"
OUTPUT_DIR="${REPO_ROOT}/out-data"

# Model storage: prefer /ptmp (MPCDF fast scratch) > TMPDIR > HOME.
# NOTE: this is a *persistent* path (no SLURM_JOB_ID suffix) so that models
# pre-downloaded once via the "offline model download" steps in the README
# are found again by every later job. Override OLLAMA_MODELS yourself before
# calling sbatch if you need an isolated/job-specific cache instead.
OLLAMA_MODELS="${OLLAMA_MODELS:-${PTMP:-${TMPDIR:-${HOME}/ollama_models}}/ollama_models}"
OLLAMA_SERVER_LOG="${LOG_DIR}/ollama_server_${SLURM_JOB_ID:-local}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${OLLAMA_MODELS}"

echo "=================================================="
echo "  SVSAL Embedding Creation"
echo "  Job  : ${SLURM_JOB_ID:-<interactive>}"
echo "  Node : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Model: ${MODEL}"
echo "  Time : $(date)"
echo "  Repo : ${REPO_ROOT}"
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

# ── Ensure the requested model is available ────────────────────────────────
# `ollama show` only checks the local manifest in OLLAMA_MODELS and never
# touches the network, so this is safe to run even on compute nodes without
# internet access. The network-dependent `ollama pull` is only invoked as a
# fallback when the model is missing -- on offline nodes, pre-download the
# model first (see README: "Downloading models without internet access").
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

# ── Create embeddings ─────────────────────────────────────────────────────────
echo "Starting embedding creation …"
uv run python "${SCRIPT_DIR}/create_embeddings_ollama.py" \
    --input              "${REPO_ROOT}/in-data/corpus_20260111.csv" \
    --output-dir         "${OUTPUT_DIR}" \
    --model              "${MODEL}" \
    --url                "${OLLAMA_URL}/v1" \
    --concurrent-requests 5 \
    --batch-size         32

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
