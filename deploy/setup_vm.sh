#!/usr/bin/env bash
# ============================================================
# Bomberland GCP VM Setup Script
# Tested on: Ubuntu 22.04 LTS (GCE)
#
# Usage:
#   chmod +x deploy/setup_vm.sh
#   bash deploy/setup_vm.sh
# ============================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────
CONDA_ENV_NAME="aic_gdgoc"
PYTHON_VERSION="3.11"
# Detect the project root (parent of deploy/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  Bomberland VM Setup"
echo "  Project root: $PROJECT_ROOT"
echo "============================================"

# ── 1. System packages ────────────────────────────────────
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    python3-dev \
    libffi-dev \
    git \
    curl \
    wget \
    unzip \
    > /dev/null 2>&1
echo "  ✓ System packages installed."

# ── 2. Install Miniconda (if not already present) ─────────
echo "[2/5] Checking for Conda..."
if command -v conda &> /dev/null; then
    echo "  ✓ Conda already installed: $(conda --version)"
else
    echo "  Installing Miniconda..."
    MINICONDA_INSTALLER="/tmp/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_INSTALLER"
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    rm "$MINICONDA_INSTALLER"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    echo "  ✓ Miniconda installed. You may need to restart your shell."
fi

# Ensure conda is active in this script
eval "$(conda shell.bash hook 2>/dev/null || "$HOME/miniconda3/bin/conda" shell.bash hook)"

# ── 3. Create Conda environment ──────────────────────────
echo "[3/5] Setting up Conda environment '$CONDA_ENV_NAME'..."
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  ✓ Environment '$CONDA_ENV_NAME' already exists."
else
    conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -q
    echo "  ✓ Environment '$CONDA_ENV_NAME' created."
fi

conda activate "$CONDA_ENV_NAME"
echo "  Active Python: $(python3 --version) at $(which python3)"

# ── 4. Install Python dependencies ───────────────────────
echo "[4/5] Installing Python packages from requirements.txt..."
pip install -q -r "$PROJECT_ROOT/requirements.txt"
echo "  ✓ All Python dependencies installed."

# ── 5. Install and configure systemd services ────────────
echo "[5/5] Installing systemd services..."

CONDA_ENV_PATH="$(conda info --envs | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')"
CURRENT_USER="$(whoami)"

for SERVICE_FILE in bomberland-web.service bomberland-worker.service; do
    SRC="$PROJECT_ROOT/deploy/$SERVICE_FILE"
    DEST="/etc/systemd/system/$SERVICE_FILE"

    if [ ! -f "$SRC" ]; then
        echo "  ⚠ Template not found: $SRC (skipping)"
        continue
    fi

    # Replace placeholders with actual paths
    sudo cp "$SRC" "$DEST"
    sudo sed -i "s|__USER__|$CURRENT_USER|g" "$DEST"
    sudo sed -i "s|__WORKDIR__|$PROJECT_ROOT|g" "$DEST"
    sudo sed -i "s|__CONDA_ENV__|$CONDA_ENV_PATH|g" "$DEST"

    echo "  ✓ Installed $SERVICE_FILE"
done

sudo systemctl daemon-reload
echo "  ✓ systemd reloaded."

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Copy your .env and secrets/ to $PROJECT_ROOT"
echo "  2. Copy competition.db to $PROJECT_ROOT"
echo "  3. Start services:"
echo "       sudo systemctl enable --now bomberland-web"
echo "       sudo systemctl enable --now bomberland-worker"
echo "  4. Check status:"
echo "       sudo systemctl status bomberland-web"
echo "       sudo systemctl status bomberland-worker"
echo "  5. Set up cron jobs (crontab -e):"
echo "       0 20 * * * cd $PROJECT_ROOT && $CONDA_ENV_PATH/bin/python3 -m scripts.post_daily_highlights"
echo "       30 20 * * * cd $PROJECT_ROOT && $CONDA_ENV_PATH/bin/python3 -m scripts.backup_db"
echo "============================================"
