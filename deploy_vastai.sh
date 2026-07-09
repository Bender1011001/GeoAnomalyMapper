#!/bin/bash
# Deploy GeoAnomalyMapper to Vast.ai GPU instance
# Usage: Run this ON the vast.ai instance after uploading the repo

set -e

echo "=== Installing dependencies ==="
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --quiet numpy scipy scikit-image rasterio tqdm requests asf_search python-dotenv pyvista

echo "=== Setting up environment ==="
# Create .env without embedding credential values in this script.
if [ ! -f .env ]; then
    {
        echo "# NASA Earthdata credentials for ASF/Sentinel-1 downloads"
        echo "# Fill these locally before real downloads, or export EARTHDATA_* before running this script."
        printf 'EARTHDATA_USERNAME=%s\n' "${EARTHDATA_USERNAME:-}"
        printf 'EARTHDATA_PASSWORD=%s\n' "${EARTHDATA_PASSWORD:-}"
    } > .env
    if [ -n "${EARTHDATA_USERNAME:-}" ] || [ -n "${EARTHDATA_PASSWORD:-}" ]; then
        echo "Created .env from existing EARTHDATA_* environment variables (values not displayed)."
    else
        echo "Created .env with empty Earthdata placeholders. Fill it locally before real downloads."
    fi
else
    echo ".env already exists; leaving it unchanged and not displaying contents."
fi

echo "=== Verifying GPU ==="
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

echo "=== Running PINN training (full power settings for 4090) ==="
python run_biondi_exploration.py --phase 1 --resolution standard 2>&1 | tee training_log.txt

echo "=== Training complete! ==="
echo "Results saved in data/biondi_exploration/"
