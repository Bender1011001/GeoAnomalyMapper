from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
import uuid
import asyncio
from datetime import datetime
from gam.core.pipeline import GAMPipeline
from gam.core.config import GAMConfig
from gam.core.exceptions import (
    PipelineError, IngestionError, PreprocessingError, 
    ModelingError, VisualizationError
)
from pathlib import Path

app = FastAPI(title="GeoAnomalyMapper API", version="1.0.0")

# Global job storage
jobs: Dict[str, Dict[str, Any]] = {}
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
    job = jobs[job_id]
    job['status'] = "RUNNING"
    job['start_time'] = datetime.now()
    job['progress'] = 0.0
    job['stage'] = "Starting"

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

        # Stage 1: Ingestion
        job['stage'] = "Ingestion"
        job['progress'] = 20.0
        raw_data = pipeline.ingestion.fetch_multiple(
            modalities=request.modalities,
            bbox=bbox_tuple,
            use_cache=True,
            global_mode=False,
            tiles=10
        )

        # Stage 2: Preprocessing
        job['stage'] = "Preprocessing"
        job['progress'] = 40.0
        processed_data = pipeline.preprocessing.process(raw_data)

        # Stage 3: Modeling - Inversion
        job['stage'] = "Modeling (Inversion)"
        job['progress'] = 60.0
        inversion_results = pipeline.modeling.invert(processed_data)

        # Stage 4: Anomaly Detection
        job['stage'] = "Anomaly Detection"
        job['progress'] = 80.0
        anomalies = pipeline.modeling.detect_anomalies(inversion_results)

        # Stage 5: Visualization
        job['stage'] = "Visualization"
        job['progress'] = 100.0
        visualizations = pipeline.visualization.generate(
            processed_data, inversion_results, anomalies, output_dir=str(job_output_dir)
        )

        # Cleanup
        pipeline.close()

        # Store results
        job['status'] = "COMPLETED"
        job['results'] = {
            "raw_data": raw_data,
            "processed_data": processed_data,
            "inversion_results": inversion_results,
            "anomalies": anomalies,
            "visualizations": {k: str(v) for k, v in visualizations.items()}
        }
        job['output_files'] = {k: str(v) for k, v in visualizations.items()}
        job['message'] = f"Analysis completed successfully. Found {len(anomalies)} anomalies."

    except IngestionError as e:
        job['status'] = "FAILED"
        job['error_message'] = f"Ingestion failed: {str(e)}"
        job['stage'] = "Ingestion"
    except PreprocessingError as e:
        job['status'] = "FAILED"
        job['error_message'] = f"Preprocessing failed: {str(e)}"
        job['stage'] = "Preprocessing"
    except ModelingError as e:
        job['status'] = "FAILED"
        job['error_message'] = f"Modeling failed: {str(e)}"
        job['stage'] = "Modeling"
    except VisualizationError as e:
        job['status'] = "FAILED"
        job['error_message'] = f"Visualization failed: {str(e)}"
        job['stage'] = "Visualization"
    except PipelineError as e:
        job['status'] = "FAILED"
        job['error_message'] = f"Pipeline error: {str(e)}"
        job['stage'] = "Pipeline"
    except Exception as e:
        job['status'] = "FAILED"
        job['error_message'] = f"Unexpected error: {str(e)}"
        job['stage'] = "Unknown"
    finally:
        if 'pipeline' in locals():
            pipeline.close()

@app.post("/analysis", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "QUEUED",
        "progress": 0.0,
        "stage": "Queued",
        "start_time": None,
        "results": None,
        "error_message": None,
        "output_files": None
    }
    background_tasks.add_task(run_analysis_job, job_id, request)
    return AnalysisResponse(
        job_id=job_id,
        status="QUEUED",
        message="Analysis job queued successfully"
    )

@app.get("/analysis/{job_id}/status", response_model=StatusResponse)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
        message=job.get("message") or job.get("error_message")
    )

@app.get("/analysis/{job_id}/results", response_model=ResultsResponse)
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=425, detail="Job not completed yet")
    if job["results"] is None:
        raise HTTPException(status_code=500, detail="Results not available")
    return ResultsResponse(
        job_id=job_id,
        results=job["results"],
        output_files=job["output_files"]
    )