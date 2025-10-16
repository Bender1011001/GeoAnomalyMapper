.PHONY: stac download harmonize features fuse train infer vectorize serve

stac:
python -m gam.agents.stac_index init --out data/stac

download:
python -m gam.agents.gam_data_agent sync --config config/data_sources.yaml

harmonize:
python -m gam.io.reprojection run --tiling config/tiling_zones.yaml

fuse:
python -m gam.fusion.multi_resolution_fusion run --config config/fusion.yaml

features:
python -m gam.features.rolling_features run --tiling config/tiling_zones.yaml --schema data/feature_schema.json

train:
python -m gam.models.train run --dataset data/labels/training_points.csv --schema data/feature_schema.json --output artifacts

infer:
python -m gam.models.infer_tiles run --features data/features --model artifacts/selected_model.pkl --schema data/feature_schema.json --output data/products

vectorize:
python -m gam.models.postprocess run --probabilities data/products --output data/products/vectors --threshold from:mlflow

serve:
uvicorn gam.api.main:app --host 0.0.0.0 --port 8080
