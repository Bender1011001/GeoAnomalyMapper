from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
import uuid
import asyncio
from datetime import datetime
from .job_store import job_store
from gam.core.pipeline import GAMPipeline
from gam.core.config import GAMConfig
from gam.core.exceptions import (
    PipelineError, IngestionError, PreprocessingError, 
    ModelingError, VisualizationError
)
from pathlib import Path

app = FastAPI(title="GeoAnomalyMapper API", version="1.0.0")

JOB_STATES = {"QUEUED", "RUNNING", "COMPLETED", "FAILED"}

class AnalysisRequest(BaseModel):
    bbox: List[float]
    modalities: List[str]
    output_dir: str
    config_path: Optional[str] = None
    verbose: bool = False

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    stage: str
    message: Optional[str] = None

class ResultsResponse(BaseModel):
    job_id: str
    results: Dict[str, Any]
    output_files: Dict[str, str]

async def run_analysis_job(job_id: str, request: AnalysisRequest):
    """Background task to run the analysis pipeline asynchronously."""
    def progress_callback(stage: str, progress: float):
        """Callback to update job store with progress."""
        job_store.update_job(job_id, {"stage": stage, "progress": progress})

    job_store.update_job(job_id, {
        "status": "RUNNING",
        "start_time": datetime.now(),
        "progress": 0.0,
        "stage": "Starting"
    })

    pipeline = None
    try:
        # Create unique output directory for this job
        base_output_dir = Path(request.output_dir)
        job_output_dir = base_output_dir / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        config = GAMConfig.from_yaml(request.config_path) if request.config_path else GAMConfig()

        # Create pipeline instance
        pipeline = GAMPipeline(
            config=config,
            use_dask=config.use_parallel,
            n_workers=config.n_workers,
            cache_dir=Path(config.cache_dir)
        )

        # Convert bbox to tuple
        bbox_tuple: Tuple[float, float, float, float] = tuple(request.bbox)

        # Run the full pipeline
        results = pipeline.run_analysis(
            bbox=bbox_tuple,
            modalities=request.modalities,
            output_dir=str(job_output_dir),
            use_cache=True,
            global_mode=False,
            tiles=10,
            progress_callback=progress_callback
        )

        # Store results
        job_store.update_job(job_id, {
            "status": "COMPLETED",
            "progress": 1.0,
            "stage": "Finished",
            "results": {
                "raw_data": results.raw_data,
                "processed_data": results.processed_data,
                "inversion_results": results.inversion_results,
                "anomalies": results.anomalies,
                "visualizations": {k: str(v) for k, v in results.visualizations.items()}
            },
            "output_files": {k: str(v) for k, v in results.visualizations.items()},
            "message": f"Analysis completed successfully. Found {len(results.anomalies)} anomalies."
        })

    except IngestionError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Ingestion failed: {str(e)}",
            "stage": "Ingestion"
        })
    except PreprocessingError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Preprocessing failed: {str(e)}",
            "stage": "Preprocessing"
        })
    except ModelingError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Modeling failed: {str(e)}",
            "stage": "Modeling"
        })
    except VisualizationError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Visualization failed: {str(e)}",
            "stage": "Visualization"
        })
    except PipelineError as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Pipeline error: {str(e)}",
            "stage": "Pipeline"
        })
    except Exception as e:
        job_store.update_job(job_id, {
            "status": "FAILED",
            "error_message": f"Unexpected error: {str(e)}",
            "stage": "Unknown"
        })
    finally:
        if pipeline:
            pipeline.close()

@app.post("/analysis", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = job_store.create_job()
    background_tasks.add_task(run_analysis_job, job_id, request)
    return AnalysisResponse(
        job_id=job_id,
        status="QUEUED",
        message="Analysis job queued successfully"
    )

@app.get("/analysis/{job_id}/status", response_model=StatusResponse)
async def get_status(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
        message=job.get("message") or job.get("error_message")
    )

@app.get("/analysis/{job_id}/results", response_model=ResultsResponse)
async def get_results(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=425, detail="Job not completed yet")
    if job["results"] is None:
        raise HTTPException(status_code=500, detail="Results not available")
    return ResultsResponse(
        job_id=job_id,
        results=job["results"],
        output_files=job["output_files"]
    )