"""CLI interface for GeoAnomalyMapper using Click framework.

Provides the 'gam run' command for basic analysis, with support for --version and --help.
"""

import click
from pathlib import Path
from typing import List, Tuple

# Temporarily commented out for fetch isolation (missing modules)
# from .pipeline import GAMPipeline
# from .config import GAMConfig
# from .exceptions import PipelineError, ConfigurationError

import subprocess
import signal
import sys
import time
import webbrowser
import socket
from pathlib import Path
from typing import List
from dotenv import load_dotenv # For loading environment variables

# Load environment variables from .env file
load_dotenv()

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("1.0.0", "-v", "--version", message="GeoAnomalyMapper %(version)s")
def cli() -> None:
    """GeoAnomalyMapper CLI for geophysical anomaly detection analysis."""
    pass


# Temporarily commented out for fetch isolation (depends on missing pipeline/config modules)
# @cli.command()
# @click.option(
#     "--bbox",
#     required=True,
#     type=str,
#     help="Bounding box in format 'min_lon,min_lat,max_lon,max_lat' (e.g., '29.0,29.5,31.5,31.0')",
# )
# @click.option(
#     "--modalities",
#     "-m",
#     default="gravity,magnetic",
#     type=str,
#     help="Comma-separated modalities (default: gravity,magnetic). Supported: gravity,magnetic,seismic,insar",
# )
# @click.option(
#     "--output-dir",
#     "-o",
#     default="./results",
#     type=click.Path(exists=False, file_okay=False, writable=True),
#     help="Output directory for results (default: ./results)",
# )
# @click.option(
#     "--config",
#     "-c",
#     type=click.Path(exists=True, dir_okay=False),
#     help="Path to custom configuration YAML file (optional)",
# )
# @click.option(
#     "--verbose",
#     "-v",
#     is_flag=True,
#     help="Enable verbose logging",
# )
# def run(
#     bbox: str,
#     modalities: str,
#     output_dir: str,
#     config: str,
#     verbose: bool,
# ) -> None:
#     """Run GeoAnomalyMapper analysis for the specified bounding box.
#
#     Example: gam run --bbox "29.0,29.5,31.5,31.0" --modalities gravity --output-dir ./giza_analysis --verbose
#     """
#     try:
#         output_path = Path(output_dir)
#         modalities_list: List[str] = [m.strip() for m in modalities.split(",")]
#         config_path: Path | None = Path(config) if config else None
#
#         # Parse bbox
#         bbox_parts = [float(p.strip()) for p in bbox.split(",")]
#         if len(bbox_parts) != 4:
#             raise ValueError("Bounding box must be in format 'min_lon,min_lat,max_lon,max_lat'")
#         bbox_tuple: Tuple[float, float, float, float] = tuple(bbox_parts)
#
#         # Load config
#         config_obj = GAMConfig.from_yaml(config_path) if config_path else GAMConfig()
#
#         # Create pipeline instance
#         pipeline = GAMPipeline(
#             config=config_obj,
#             use_dask=config_obj.use_parallel,
#             n_workers=config_obj.n_workers,
#             cache_dir=Path(config_obj.cache_dir)
#         )
#
#         # Run the full pipeline
#         results = pipeline.run_analysis(
#             bbox=bbox_tuple,
#             modalities=modalities_list,
#             output_dir=str(output_path),
#             use_cache=True,
#             global_mode=False,
#             tiles=10
#         )
#
#         pipeline.close()
#
#         click.echo(f"Analysis completed successfully!")
#         click.echo(f"Results saved to: {output_path}")
#         click.echo(f"Detected {len(results.anomalies)} potential anomalies.")
#
#     except (PipelineError, ConfigurationError, ValueError) as e:
#         click.echo(click.style(f"Error: {str(e)}", fg="red"), err=True)
#         raise click.Abort()
#     except Exception as e:
#         click.echo(click.style(f"Unexpected error: {str(e)}", fg="red"), err=True)
#         raise click.Abort()

@cli.command()
@click.option('--no-browser', is_flag=True, help='Do not auto-open browser')
@click.option('--port-api', default=8000, type=int, help='Port for FastAPI backend')
@click.option('--port-dashboard', default=8501, type=int, help='Port for Streamlit dashboard')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def start(no_browser: bool, port_api: int, port_dashboard: int, debug: bool) -> None:
    """Start the GAM dashboard with backend API.

    This command starts both the FastAPI backend and Streamlit frontend concurrently.
    The browser will automatically open to the dashboard unless --no-browser is specified.
    Press Ctrl+C to stop both services gracefully.
    """
    # Check if ports are in use
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if is_port_in_use(port_api):
        click.echo(click.style(f"Error: Port {port_api} is already in use for FastAPI backend.", fg="red"), err=True)
        sys.exit(1)

    if is_port_in_use(port_dashboard):
        click.echo(click.style(f"Error: Port {port_dashboard} is already in use for Streamlit dashboard.", fg="red"), err=True)
        sys.exit(1)

    # Project root directory
    project_root = Path(__file__).parent.parent.parent

    # Start processes
    processes = []

    # Start FastAPI backend
    api_cmd = [
        "uvicorn",
        "gam.api.main:app",
        "--host", "0.0.0.0",
        "--port", str(port_api),
        "--log-level", "debug" if debug else "info"
    ]
    if debug:
        api_cmd.append("--reload")

    click.echo(click.style("🚀 Starting FastAPI backend on http://localhost:" + str(port_api) + "...", fg="green"))
    api_process = subprocess.Popen(api_cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    processes.append(api_process)

    # Start Streamlit dashboard
    dashboard_cmd = [
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.port", str(port_dashboard),
        "--server.address", "0.0.0.0"
    ]
    if debug:
        dashboard_cmd.extend(["--logger.level", "debug"])

    click.echo(click.style("📊 Starting Streamlit dashboard on http://localhost:" + str(port_dashboard) + "...", fg="green"))
    dashboard_process = subprocess.Popen(dashboard_cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    processes.append(dashboard_process)

    # Wait for services to start
    click.echo("⏳ Waiting for services to initialize...")
    time.sleep(3)

    # Auto-open browser
    if not no_browser:
        dashboard_url = f"http://localhost:{port_dashboard}"
        click.echo(click.style(f"🌐 Opening browser to {dashboard_url}...", fg="blue"))
        webbrowser.open(dashboard_url)

    click.echo(click.style("✅ GAM system started successfully!", fg="green"))
    click.echo(f"API: http://localhost:{port_api}")
    click.echo(f"Dashboard: http://localhost:{port_dashboard}")
    click.echo("Press Ctrl+C to stop both services gracefully.")

    # Graceful shutdown handler
    def signal_handler(sig: int, frame: object) -> None:
        click.echo(click.style("\n🛑 Shutting down services...", fg="yellow"))
        for proc in processes:
            if proc.poll() is None:  # Process is still running
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep the main process alive
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


import sys


def _get_download_dataset():
    """Lazy import to avoid importing ingestion stack at CLI import-time."""
    try:
        from gam.ingestion.fetchers import download_dataset as _download_dataset
    except Exception as e:
        raise ImportError(
            "download_dataset is required to execute 'gam fetch' commands; "
            "ensure ingestion dependencies are installed and 'gam.ingestion.fetchers' "
            "defines 'download_dataset'."
        ) from e
    return _download_dataset


@cli.group()
def fetch():
    """Fetch datasets from configured sources in data_sources.yaml."""
    pass


@fetch.command()
@click.option('--force', is_flag=True, help='Overwrite existing files')
@click.option('--timeout', default=60, type=int, help='Request timeout in seconds (default: 60)')
@click.option('--retries', default=3, type=int, help='Number of retry attempts (default: 3)')
@click.option('--skip-checksum', is_flag=True, help='Skip checksum verification if provided')
def emag2(force: bool, timeout: int, retries: int, skip_checksum: bool) -> None:
    """Fetch EMAG2 v3 magnetic anomaly dataset."""
    try:
        download_dataset = _get_download_dataset()
        final_path = download_dataset(
            dataset_id='emag2_v3',
            force=force,
            timeout=timeout,
            retries=retries,
            skip_checksum=skip_checksum
        )
        click.echo(f"Successfully fetched EMAG2 v3 to: {final_path}")
    except Exception as e:
        click.echo(click.style(f"Error fetching emag2: {str(e)}", fg="red"), err=True)
        sys.exit(1)


@fetch.command('dataset')
@click.argument('dataset_id', type=str)
@click.option('--force', is_flag=True, help='Overwrite existing files')
@click.option('--timeout', default=60, type=int, help='Request timeout in seconds (default: 60)')
@click.option('--retries', default=3, type=int, help='Number of retry attempts (default: 3)')
@click.option('--skip-checksum', is_flag=True, help='Skip checksum verification if provided')
def dataset_cmd(dataset_id: str, force: bool, timeout: int, retries: int, skip_checksum: bool) -> None:
    """Fetch an arbitrary dataset defined in data_sources.yaml by dataset_id.

    Usage:
      gam fetch dataset <dataset_id> [--force] [--timeout 60] [--retries 3] [--skip-checksum]

    Examples:
      python -m GeoAnomalyMapper.gam.cli fetch dataset gravity_egm2008_tif --timeout 600 --retries 5
      python -m GeoAnomalyMapper.gam.cli fetch gravity --timeout 600 --retries 5
      python -m GeoAnomalyMapper.gam.cli fetch all
    """
    try:
        download_dataset = _get_download_dataset()
        final_path = download_dataset(
            dataset_id=dataset_id,
            force=force,
            timeout=timeout,
            retries=retries,
            skip_checksum=skip_checksum
        )
        click.echo(f"Successfully fetched {dataset_id} to: {final_path}")
    except Exception as e:
        click.echo(click.style(f"Error fetching {dataset_id}: {str(e)}", fg="red"), err=True)
        sys.exit(1)

@fetch.command()
@click.option('--force', is_flag=True, help='Overwrite existing files')
@click.option('--timeout', default=60, type=int, help='Request timeout in seconds (default: 60)')
@click.option('--retries', default=3, type=int, help='Number of retry attempts (default: 3)')
@click.option('--skip-checksum', is_flag=True, help='Skip checksum verification if provided')
def gravity(force: bool, timeout: int, retries: int, skip_checksum: bool) -> None:
    """Fetch global gravity anomaly dataset."""
    try:
        download_dataset = _get_download_dataset()
        final_path = download_dataset(
            dataset_id='gravity_egm2008_tif',
            force=force,
            timeout=timeout,
            retries=retries,
            skip_checksum=skip_checksum
        )
        click.echo(f"Successfully fetched gravity to: {final_path}")
    except Exception as e:
        click.echo(click.style(f"Error fetching gravity: {str(e)}", fg="red"), err=True)
        sys.exit(1)

@fetch.command('all')
@click.option('--force', is_flag=True, help='Overwrite existing files')
@click.option('--timeout', default=60, type=int, help='Request timeout in seconds (default: 60)')
@click.option('--retries', default=3, type=int, help='Number of retry attempts (default: 3)')
@click.option('--skip-checksum', is_flag=True, help='Skip checksum verification if provided')
def all_cmd(force: bool, timeout: int, retries: int, skip_checksum: bool) -> None:
    """Fetch all configured datasets (emag2 and gravity)."""
    download_dataset = _get_download_dataset()
    datasets = ['emag2_v3', 'gravity_egm2008_tif']
    errors = []
    for dataset_id in datasets:
        try:
            final_path = download_dataset(
                dataset_id=dataset_id,
                force=force,
                timeout=timeout,
                retries=retries,
                skip_checksum=skip_checksum
            )
            ds_name = 'emag2' if dataset_id == 'emag2_v3' else 'gravity'
            click.echo(f"Successfully fetched {ds_name} to: {final_path}")
        except Exception as e:
            ds_name = 'emag2' if dataset_id == 'emag2_v3' else 'gravity'
            errors.append(f"{ds_name}: {str(e)}")
        
    if errors:
        click.echo(click.style("Errors fetching datasets:", fg="red"), err=True)
        for err in errors:
            click.echo(click.style(f"  - {err}", fg="red"), err=True)
        sys.exit(1)
    click.echo("All datasets fetched successfully.")


# =========================
# Preprocess and Fuse Groups
# =========================

# Constants and helpers local to this module to avoid broad import changes above.
_DEFAULT_INPUTS = {
    "mag": Path("./data/raw/emag2/EMAG2_V3_SeaLevel_DataTiff.tif"),
    "grav": Path("./data/raw/gravity/egm2008_anomaly_5min.tif"),
}
_DEFAULT_OUTDIRS = {
    "mag": Path("./data/outputs/cog/mag/z0p1d/tiles"),
    "grav": Path("./data/outputs/cog/grav/z0p1d/tiles"),
    "fused": Path("./data/outputs/cog/fused/z0p1d/tiles"),
}
_LOG_PATH = Path("./logs/global_open_fusion_mvp.log")


def _append_json_log(entry: dict) -> None:
    import json
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, allow_nan=False) + "\n")


def _stats_nan(a) -> dict | None:
    import numpy as np
    if a is None:
        return None
    finite = np.isfinite(a)
    if not np.any(finite):
        return None
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
    }


def _intersects(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> bool:
    minx1, miny1, maxx1, maxy1 = b1
    minx2, miny2, maxx2, maxy2 = b2
    return not (maxx1 <= minx2 or maxx2 <= minx1 or maxy1 <= miny2 or maxy2 <= miny1)


def _tiles_from_pilot(pilot: str) -> list[str]:
    """Parse --pilot like 'N20,E120,span=20' and return overlapping tile ids."""
    try:
        from gam.core.tiles import tiles_10x10_ids, tile_bounds_10x10
    except ImportError as e:
        raise click.ClickException("rasterio is required for tile utilities; install rasterio to use this feature.") from e
    pilot = (pilot or "").strip()
    parts = [p.strip() for p in pilot.split(",")]
    if len(parts) != 3 or not parts[2].startswith("span="):
        raise click.ClickException("Pilot format must be 'N20,E120,span=20'")
    lat_label, lon_label, span_part = parts
    # Lat parse
    if len(lat_label) != 3 or lat_label[0] not in ("N", "S") or not lat_label[1:].isdigit():
        raise click.ClickException("Pilot latitude must look like N20 or S10")
    lat_deg = int(lat_label[1:])
    if lat_deg % 10 != 0 or lat_deg > 80 and lat_label[0] == "N" or lat_deg > 90 and lat_label[0] == "S":
        raise click.ClickException("Pilot latitude out of 10° band range")
    lat_min_base = float(lat_deg if lat_label[0] == "N" else -lat_deg)
    # Lon parse
    if len(lon_label) != 4 or lon_label[0] not in ("E", "W") or not lon_label[1:].isdigit():
        raise click.ClickException("Pilot longitude must look like E120 or W010")
    lon_deg = int(lon_label[1:])
    if lon_deg % 10 != 0 or lon_deg > 170 and lon_label[0] == "E" or lon_deg > 180 and lon_label[0] == "W":
        raise click.ClickException("Pilot longitude out of 10° band range")
    lon_min = float(lon_deg if lon_label[0] == "E" else -lon_deg)
    # Span parse
    try:
        span = int(span_part.split("=", 1)[1])
    except Exception:
        raise click.ClickException("Pilot span must be an integer degrees multiple of 10")
    if span <= 0 or span % 10 != 0:
        raise click.ClickException("Pilot span must be a positive multiple of 10 degrees")
    n_bands = span // 10
    # Latitude span goes toward the equator from the given band, per example N20 -> [N10,N20]
    if lat_min_base >= 0:
        lat_min = lat_min_base - (n_bands - 1) * 10
        lat_max = lat_min_base + 10
    else:
        lat_min = lat_min_base
        lat_max = lat_min_base + n_bands * 10
    # Longitude span goes eastward from the given lon band, per example E120 -> [E120,E130]
    lon_max = lon_min + span
    bbox = (lon_min, lat_min, lon_max, lat_max)
    # Enumerate overlapping tiles
    ids = []
    for tid in tiles_10x10_ids():
        tb = tile_bounds_10x10(tid)
        if _intersects(tb, bbox):
            ids.append(tid)
    return ids


def _resolve_tiles(tiles_opt: str | None, pilot_opt: str | None, all_opt: bool) -> list[str]:
    """Resolve tiles selection according to options. If none provided, default to ['t_N00_E000']."""
    try:
        from gam.core.tiles import tiles_10x10_ids, tile_bounds_10x10
    except ImportError as e:
        raise click.ClickException("rasterio is required for tile utilities; install rasterio to use this feature.") from e
    if sum(bool(x) for x in [tiles_opt, pilot_opt, all_opt]) > 1:
        raise click.ClickException("Use only one of --tiles, --pilot, or --all")
    if all_opt:
        return tiles_10x10_ids()
    if pilot_opt:
        return _tiles_from_pilot(pilot_opt)
    if tiles_opt:
        # Validate and normalize
        tlist = [t.strip() for t in tiles_opt.split(",") if t.strip()]
        if not tlist:
            raise click.ClickException("No tiles parsed from --tiles")
        for t in tlist:
            # Validation by computing bounds
            tile_bounds_10x10(t)
        return tlist
    # Default single example tile
    return ["t_N00_E000"]


@cli.group()
def preprocess():
    """Preprocessing utilities for tiling and COG generation."""
    pass


@preprocess.command("list-tiles")
def preprocess_list_tiles() -> None:
    """List all 10°×10° tile IDs in order."""
    try:
        from gam.core.tiles import tiles_10x10_ids
    except ImportError as e:
        raise click.ClickException("rasterio is required for tile utilities; install rasterio to use this feature.") from e
    for tid in tiles_10x10_ids():
        click.echo(tid)


@preprocess.command("layer")
@click.option("--layer", type=click.Choice(["mag", "grav"], case_sensitive=False), required=True, help="Layer to preprocess")
@click.option("--tiles", type=str, default=None, help='Comma-separated tile ids like "t_N30_E120,t_N20_E120"')
@click.option("--pilot", type=str, default=None, help='Pilot selector like "N20,E120,span=20"')
@click.option("--all", "all_tiles", is_flag=True, help="Process all global 10°×10° tiles")
@click.option("--dst-res", type=float, default=0.1, show_default=True, help="Destination resolution in degrees")
@click.option("--outdir", type=click.Path(file_okay=False), default=None, help="Output tiles directory (defaults by layer)")
@click.option("--force", is_flag=True, help="Overwrite if output exists")
def preprocess_layer(layer: str, tiles: str | None, pilot: str | None, all_tiles: bool, dst_res: float, outdir: str | None, force: bool) -> None:
    """Warp+crop input layer into 10°×10° 0.1° COGs."""
    import numpy as np
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as e:
        raise click.ClickException("rasterio is required for preprocessing; install rasterio to use this feature.") from e
    try:
        from gam.core.tiles import tile_bounds_10x10
    except ImportError as e:
        raise click.ClickException("rasterio is required for tile utilities; install rasterio to use this feature.") from e
    try:
        from gam.preprocessing.cog_writer import warp_crop_to_tile, write_cog
    except ImportError as e:
        raise click.ClickException("rasterio (and rio-cogeo optional) is required for COG writing; install rasterio to use this feature.") from e

    layer = layer.lower()
    in_path = _DEFAULT_INPUTS[layer]
    if layer == "grav" and not in_path.exists():
        click.echo(click.style(f"Gravity source missing: {in_path}. Provide './data/raw/gravity/egm2008_anomaly_5min.tif'.", fg="red"), err=True)
        raise click.Abort()
    if layer == "mag" and not in_path.exists():
        click.echo(click.style(f"Magnetic source missing: {in_path}. You may run fetch or provide the file.", fg="red"), err=True)
        raise click.Abort()

    tiles_list = _resolve_tiles(tiles, pilot, all_tiles)
    out_dir = Path(outdir) if outdir else _DEFAULT_OUTDIRS[layer]
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    for tid in tiles_list:
        bounds = tile_bounds_10x10(tid)
        out_path = out_dir / f"{tid}.tif"
        if out_path.exists() and not force:
            skipped += 1
            _append_json_log({
                "stage": "preprocess",
                "step": "skip_existing",
                "tile_id": tid,
                "layer": layer,
                "path": str(out_path),
                "status": "skipped",
                "elapsed_ms": 0,
                "shape": None,
                "stats": None,
            })
            continue

        t0 = time.perf_counter()
        try:
            arr = warp_crop_to_tile(str(in_path), bounds, dst_res=float(dst_res), dst_crs="EPSG:4326", resampling="bilinear")
            shape = (int(arr.shape[0]), int(arr.shape[1]))
            # Self-check print: should be 100x100
            click.echo(f"{layer}:{tid} warped shape={shape}")
            write_cog(str(out_path), arr, bounds, dst_res=float(dst_res), tags={"layer": layer, "tile_id": tid})
            # Validate readability
            with rasterio.open(str(out_path)) as ds:
                _ = ds.read(1, window=Window(0, 0, 1, 1))
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _append_json_log({
                "stage": "preprocess",
                "step": "write_cog",
                "tile_id": tid,
                "layer": layer,
                "path": str(out_path),
                "status": "ok",
                "elapsed_ms": elapsed_ms,
                "shape": [shape[0], shape[1]],
                "stats": _stats_nan(arr),
            })
            processed += 1
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _append_json_log({
                "stage": "preprocess",
                "step": "error",
                "tile_id": tid,
                "layer": layer,
                "path": str(out_path),
                "status": "error",
                "elapsed_ms": elapsed_ms,
                "shape": None,
                "stats": None,
                "error": str(e),
            })
            failed += 1
            click.echo(click.style(f"Error processing {layer}:{tid} -> {e}", fg="red"), err=True)

    click.echo(f"Preprocess complete. processed={processed} skipped={skipped} failed={failed}")


@cli.group()
def fuse():
    """Fusion utilities for combining preprocessed tiles."""
    pass


@fuse.command("tiles")
@click.option("--tiles", type=str, default=None, help='Comma-separated tile ids like "t_N30_E120,t_N20_E120"')
@click.option("--pilot", type=str, default=None, help='Pilot selector like "N20,E120,span=20"')
@click.option("--all", "all_tiles", is_flag=True, help="Process all global 10°×10° tiles")
@click.option("--outdir", type=click.Path(file_okay=False), default=None, help="Output tiles directory for fused results (defaults)")
def fuse_tiles(tiles: str | None, pilot: str | None, all_tiles: bool, outdir: str | None) -> None:
    """Fuse available per-tile COGs (mag/grav) into a robust z-score mean."""
    import numpy as np
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as e:
        raise click.ClickException("rasterio is required for fusion; install rasterio to use this feature.") from e
    try:
        from gam.core.tiles import tile_bounds_10x10
    except ImportError as e:
        raise click.ClickException("rasterio is required for tile utilities; install rasterio to use this feature.") from e
    from gam.modeling.fuse_simple import robust_z, fuse_layers
    try:
        from gam.preprocessing.cog_writer import write_cog
    except ImportError as e:
        raise click.ClickException("rasterio (and rio-cogeo optional) is required for COG writing; install rasterio to use this feature.") from e

    tiles_list = _resolve_tiles(tiles, pilot, all_tiles)
    out_dir = Path(outdir) if outdir else _DEFAULT_OUTDIRS["fused"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def _read_array(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        with rasterio.open(str(path)) as ds:
            data = ds.read(1, masked=True)
        return np.asarray(data.filled(np.nan), dtype="float32")

    fused_ok = 0
    skipped = 0
    failed = 0

    for tid in tiles_list:
        bounds = tile_bounds_10x10(tid)
        mag_path = _DEFAULT_OUTDIRS["mag"] / f"{tid}.tif"
        grav_path = _DEFAULT_OUTDIRS["grav"] / f"{tid}.tif"
        out_path = out_dir / f"{tid}.tif"

        if out_path.exists():
            skipped += 1
            _append_json_log({
                "stage": "fuse",
                "step": "skip_existing",
                "tile_id": tid,
                "layer": "fused",
                "path": str(out_path),
                "status": "skipped",
                "elapsed_ms": 0,
                "shape": None,
                "stats": None,
            })
            continue

        t0 = time.perf_counter()
        try:
            mag_arr = _read_array(mag_path)
            grav_arr = _read_array(grav_path)
            if mag_arr is None and grav_arr is None:
                raise RuntimeError("No input layers found for tile")

            mag_z = robust_z(mag_arr) if mag_arr is not None else None
            grav_z = robust_z(grav_arr) if grav_arr is not None else None

            if mag_z is not None and grav_z is not None:
                fused_arr = fuse_layers(mag_z, grav_z)
                used = "mag,grav"
            elif mag_z is not None:
                fused_arr = mag_z
                used = "mag"
            else:
                fused_arr = grav_z  # type: ignore[assignment]
                used = "grav"

            write_cog(str(out_path), fused_arr, bounds, dst_res=0.1, tags={"layer": "fused", "tile_id": tid, "used": used})
            # Validate readability
            with rasterio.open(str(out_path)) as ds:
                _ = ds.read(1, window=Window(0, 0, 1, 1))
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _append_json_log({
                "stage": "fuse",
                "step": "write_cog",
                "tile_id": tid,
                "layer": "fused",
                "path": str(out_path),
                "status": "ok",
                "elapsed_ms": elapsed_ms,
                "shape": [int(fused_arr.shape[0]), int(fused_arr.shape[1])],
                "stats": _stats_nan(fused_arr),
            })
            click.echo(f"fused:{tid} used={used}")
            fused_ok += 1
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _append_json_log({
                "stage": "fuse",
                "step": "error",
                "tile_id": tid,
                "layer": "fused",
                "path": str(out_path),
                "status": "error",
                "elapsed_ms": elapsed_ms,
                "shape": None,
                "stats": None,
                "error": str(e),
            })
            failed += 1
            click.echo(click.style(f"Error fusing {tid} -> {e}", fg="red"), err=True)

    click.echo(f"Fuse complete. fused={fused_ok} skipped={skipped} failed={failed}")
if __name__ == "__main__":
    cli()