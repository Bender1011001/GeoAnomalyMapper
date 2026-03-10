#!/bin/bash
# Deploy GeoAnomalyMapper to Vast.ai GPU instance
# Usage: Run this ON the vast.ai instance after uploading the repo

set -e

echo "=== Installing dependencies ==="
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --quiet numpy scipy scikit-image rasterio tqdm requests asf_search python-dotenv pyvista

echo "=== Setting up environment ==="
# Create .env with Earthdata credentials
cat > .env << 'EOF'
EARTHDATA_USERNAME=bender1011001
EARTHDATA_PASSWORD=/xSwtJnQ7c#^EQ4
EOF

echo "=== Verifying GPU ==="
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"

echo "=== Running PINN training (full power settings for 4090) ==="
python run_biondi_exploration.py --phase 1 --resolution standard 2>&1 | tee training_log.txt

echo "=== Training complete! ==="
echo "Results saved in data/biondi_exploration/"
