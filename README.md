# GeoAnomalyMapper

GeoAnomalyMapper contains a small collection of data processing utilities for
combining publicly available geophysical datasets (gravity, magnetic, DEM and
optional InSAR products) into consistent analysis-ready rasters. The repository
keeps the focus on the processing and visualisation code that supports the
fusion workflow – large datasets, helper downloads and internal documentation
are deliberately excluded from version control.

## Key capabilities

- **Data preparation (`process_data.py`)** – clips and reprojects raw gravity,
  magnetic and elevation rasters into a common grid for a chosen area of
  interest.
- **InSAR pre-processing (`process_insar_data.py`)** – turns externally
  generated interferograms into deformation rate grids that match the rest of
  the pipeline.
- **Multi-resolution fusion (`multi_resolution_fusion.py`)** – resamples
  available layers to a shared resolution and combines them using
  uncertainty-aware weighting.
- **Void and anomaly assessment (`detect_voids.py`)** – calculates a simple
  probability score for potential subsurface voids based on the fused
  geophysical layers.
- **Visualisation utilities** – `create_visualization.py` and
  `create_enhanced_visualization.py` create publication-ready PNG, KMZ and
  summary graphics from GeoTIFF outputs.
- **Quality checks (`validate_against_known_features.py`)** – samples fused
  anomaly rasters around known features to provide an objective validation
  report.

All scripts are self-contained CLI tools with `--help` descriptions. None of the
command line utilities download data; obtaining raw datasets remains a manual
step carried out outside of the repository.

## Repository layout

```
GeoAnomalyMapper/
├── process_data.py
├── process_insar_data.py
├── multi_resolution_fusion.py
├── detect_voids.py
├── create_visualization.py
├── create_enhanced_visualization.py
├── validate_against_known_features.py
├── pyproject.toml
├── README.md
└── LICENSE
```

The `data/` directory expected by the scripts is not version controlled. Create
it locally with the following structure once you have gathered source rasters:

```
data/
├── raw/
│   ├── gravity/
│   ├── magnetic/
│   ├── dem/
│   └── insar/            # Optional
└── processed/
    ├── gravity/
    ├── magnetic/
    ├── dem/
    └── insar/
```

## Installation

GeoAnomalyMapper targets Python 3.9+. Install the dependencies into a virtual
environment of your choice:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras listed under `[project.optional-dependencies]` in
`pyproject.toml` enable COG creation and advanced InSAR processing. Install them
with, for example, `pip install .[insar]` if they are required for your
workflow.

## Preparing source data

1. **Gravity and magnetic grids** – download GeoTIFF products such as EGM2008
   free-air anomaly or EMAG2 magnetic anomaly files that cover your area of
   interest.
2. **DEM** – place Copernicus DEM, SRTM or other elevation tiles in
   `data/raw/dem/`.
3. **InSAR (optional)** – generate deformation products with ESA SNAP, ISCE or
   other toolchains outside the repository and export them as GeoTIFF rasters.
4. Keep the raw files outside of version control. The `.gitignore` shipped with
   the project already excludes the `data/` directory and related artefacts.

If Sentinel or other services require credentials, copy `.env.example` to `.env`
and populate it locally. Do not commit the resulting file.

## Typical workflow

1. **Process base layers**
   ```bash
   python process_data.py --region "-105.5,31.5,-103.5,33.5"
   ```
   The script clips available gravity, magnetic and DEM rasters to the supplied
   bounding box and stores the results under `data/processed/`.

2. **Process InSAR data (optional)**
   ```bash
   python process_insar_data.py --input data/raw/insar/my_stack.tif \
       --output data/processed/insar/insar_processed.tif
   ```

3. **Run multi-resolution fusion**
   ```bash
   python multi_resolution_fusion.py --region "-105.5,31.5,-103.5,33.5" \
       --output fused_anomaly.tif
   ```

4. **Create visual outputs**
   ```bash
   python create_visualization.py fused_anomaly.tif
   ```

5. **Validate against known features**
   ```bash
   python validate_against_known_features.py fused_anomaly.tif --buffer 3 \
       --output-dir reports/
   ```

6. **Optional void probability mapping**
   ```bash
   python detect_voids.py --region "-105.5,31.5,-103.5,33.5" --output caverns
   ```

Each command emits structured logging so processing steps can be audited.

## Publishing the interactive void explorer

The repository ships with a GitHub Pages-ready site under `docs/` that renders
processed void detections on an interactive Leaflet map.

1. Push the repository to GitHub and enable **GitHub Pages** in the repository
   settings, selecting the **docs/** folder as the publishing source.
2. Export your processed results as GeoJSON files and copy them into
   `docs/data/`.
3. Describe each dataset in `docs/data/datasets.json` by adding entries with an
   `id`, `name`, `description`, `file` path (relative to `docs/`) and `color`
   (HEX value). The viewer automatically loads every dataset listed in the
   configuration, draws it on the map and populates the dataset summary cards.
4. Commit and push the updated files. GitHub Pages republishes the site within a
   minute or two.

You can also drag-and-drop new GeoJSON files onto the "Preview a local file"
control in the published site to check their appearance before committing them.

## Validation philosophy

The validation utility samples the fused anomaly raster around a curated list of
known underground structures. It checks both the strength and the expected sign
of the anomaly before counting a location as correctly detected. Summary
statistics and a map are produced in the specified output directory to support
transparent reporting.

## Contributing

The project intentionally keeps scope narrow. When extending it, favour
self-contained processing or visualisation utilities over large download or
orchestration tooling. Please keep additional datasets, notebooks and generated
artefacts outside of git.

## License

GeoAnomalyMapper is released under the MIT License. See [LICENSE](LICENSE) for
full details.
