import click
from pathlib import Path
from typing import List, Tuple
from gam.core.pipeline import GAMPipeline
from gam.core.config import GAMConfig
from gam.core.exceptions import PipelineError, ConfigurationError
import subprocess
import signal
import sys
import time
import webbrowser
import socket
from dotenv import load_dotenv

load_dotenv()

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("1.0.0", "-v", "--version", message="GeoAnomalyMapper %(version)s")
def cli() -> None:
    """GeoAnomalyMapper CLI for geophysical anomaly detection analysis."""
    pass

@cli.command()
@click.option(
    "--bbox",
    required=True,
    type=str,
    help="Bounding box in format 'min_lon,min_lat,max_lon,max_lat' (e.g., '29.0,29.5,31.5,31.0')",
)
@click.option(
    "--modalities",
    "-m",
    default="gravity,magnetic",
    type=str,
    help="Comma-separated modalities (default: gravity,magnetic). Supported: gravity,magnetic,seismic,insar",
)
@click.option(
    "--output-dir",
    "-o",
    default="./results",
    type=click.Path(exists=False, file_okay=False, writable=True),
    help="Output directory for results (default: ./results)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to custom configuration YAML file (optional)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def run(
    bbox: str,
    modalities: str,
    output_dir: str,
    config: str,
    verbose: bool,
) -> None:
    """Run GeoAnomalyMapper analysis for the specified bounding box.

    Example: gam run --bbox "29.0,29.5,31.5,31.0" --modalities gravity --output-dir ./giza_analysis --verbose
    """
    try:
        output_path = Path(output_dir)
        modalities_list: List[str] = [m.strip() for m in modalities.split(",")]
        config_path: Path | None = Path(config) if config else None

        # Parse bbox
        bbox_parts = [float(p.strip()) for p in bbox.split(",")]
        if len(bbox_parts) != 4:
            raise ValueError("Bounding box must be in format 'min_lon,min_lat,max_lon,max_lat'")
        bbox_tuple: Tuple[float, float, float, float] = tuple(bbox_parts)

        # Load config
        config_obj = GAMConfig.from_yaml(config_path) if config_path else GAMConfig()

        # Create pipeline instance
        pipeline = GAMPipeline(
            config=config_obj,
            use_dask=config_obj.parallel.backend == 'dask',
            n_workers=config_obj.parallel.n_workers,
            cache_dir=Path(config_obj.cache_dir)
        )

        # Run the full pipeline
        results = pipeline.run_analysis(
            bbox=bbox_tuple,
            modalities=modalities_list,
            output_dir=str(output_path),
            use_cache=True,
        )

        pipeline.close()

        click.echo(f"Analysis completed successfully!")
        click.echo(f"Results saved to: {output_path}")
        click.echo(f"Detected {len(results.anomalies)} potential anomalies.")

    except (PipelineError, ConfigurationError, ValueError) as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"), err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"Unexpected error: {str(e)}", fg="red"), err=True)
        raise click.Abort()

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
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if is_port_in_use(port_api):
        click.echo(click.style(f"Error: Port {port_api} is already in use for FastAPI backend.", fg="red"), err=True)
        sys.exit(1)

    if is_port_in_use(port_dashboard):
        click.echo(click.style(f"Error: Port {port_dashboard} is already in use for Streamlit dashboard.", fg="red"), err=True)
        sys.exit(1)

    project_root = Path(__file__).parent.parent.parent

    processes = []

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

    click.echo("⏳ Waiting for services to initialize...")
    time.sleep(3)

    if not no_browser:
        dashboard_url = f"http://localhost:{port_dashboard}"
        click.echo(click.style(f"🌐 Opening browser to {dashboard_url}...", fg="blue"))
        webbrowser.open(dashboard_url)

    click.echo(click.style("✅ GAM system started successfully!", fg="green"))
    click.echo(f"API: http://localhost:{port_api}")
    click.echo(f"Dashboard: http://localhost:{port_dashboard}")
    click.echo("Press Ctrl+C to stop both services gracefully.")

    def signal_handler(sig: int, frame: object) -> None:
        click.echo(click.style("\n🛑 Shutting down services...", fg="yellow"))
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    cli()