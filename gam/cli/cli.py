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


from gam.ingestion.fetchers import download_dataset
import sys


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


if __name__ == "__main__":
    cli()