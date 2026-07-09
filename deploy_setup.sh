#!/bin/bash
# =============================================================
# GeoAnomalyMapper Deployment Script for Vast.ai GPU Box
# =============================================================
# Usage: Run this on the rented GPU box after uploading the code.
#   bash deploy_setup.sh
# =============================================================

set -e

echo "=============================================="
echo "  GeoAnomalyMapper — GPU Box Setup"
echo "=============================================="

# --- System info ---
echo ""
echo "[1/5] System Info:"
nvidia-smi -L
python3 --version
echo "CPU cores: $(nproc)"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo ""

# --- Install Python deps ---
echo "[2/5] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    numpy scipy rasterio pyproj PyWavelets \
    scikit-image scikit-learn \
    torch torchvision \
    tqdm matplotlib python-dateutil \
    shapely requests geopandas pyogrio \
    beautifulsoup4 asf_search boto3 \
    folium PyYAML simplekml pyvista pandas

echo "  Dependencies installed."

# --- Verify GPU ---
echo ""
echo "[3/5] Verifying PyTorch + CUDA..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}:   {props.name} ({props.total_memory // 1024**3} GB)')
"

# --- Set up .env ---
echo ""
echo "[4/5] Setting up credentials..."
if [ ! -f .env ]; then
    {
        echo "# NASA Earthdata credentials for ASF/Sentinel-1 downloads"
        echo "# Fill these locally before real downloads, or export EARTHDATA_* before running this script."
        printf 'EARTHDATA_USERNAME=%s\n' "${EARTHDATA_USERNAME:-}"
        printf 'EARTHDATA_PASSWORD=%s\n' "${EARTHDATA_PASSWORD:-}"
    } > .env
    if [ -n "${EARTHDATA_USERNAME:-}" ] || [ -n "${EARTHDATA_PASSWORD:-}" ]; then
        echo "  Created .env from existing EARTHDATA_* environment variables (values not displayed)."
    else
        echo "  Created .env with empty Earthdata placeholders. Fill it locally before real downloads."
    fi
else
    echo "  .env already exists; leaving it unchanged and not displaying contents."
fi

# --- Quick validation ---
echo ""
echo "[5/5] Running quick validation..."
python3 -c "
import pinn_vibro_inversion
import sar_vibrometry
import visualize_3d_subsurface
import satellite_embeddings
print('  All modules import successfully!')
"

echo ""
echo "=============================================="
echo "  SETUP COMPLETE — Ready to scan!"
echo ""
echo "  Run the full scan:"
echo "    python3 run_full_scan.py standard california"
echo "    python3 run_full_scan.py standard dumbs"
echo "    python3 run_full_scan.py standard all"
echo ""
echo "  Or run 4 parallel GPU scans:"
echo "    CUDA_VISIBLE_DEVICES=0 python3 run_full_scan.py standard california &"
echo "    CUDA_VISIBLE_DEVICES=1 python3 run_full_scan.py standard dumbs &"
echo "    CUDA_VISIBLE_DEVICES=2 python3 run_full_scan.py standard caves &"
echo "    wait"
echo "=============================================="
