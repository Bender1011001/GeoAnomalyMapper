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
    job_store.update_job(job_id, {
        "status": "RUNNING",
        "start_time": datetime.now(),
        "progress": 0.0,
        "stage": "Starting"
    })

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
        job_store.update_job(job_id, {"stage": "Ingestion", "progress": 0.2})
        raw_data = pipeline.ingestion.fetch_multiple(
            modalities=request.modalities,
            bbox=bbox_tuple,
            use_cache=True,
            global_mode=False,
            tiles=10
        )

        # Stage 2: Preprocessing
        job_store.update_job(job_id, {"stage": "Preprocessing", "progress": 0.4})
        processed_data = pipeline.preprocessing.process(raw_data)

        # Stage 3: Modeling - Inversion
        job_store.update_job(job_id, {"stage": "Modeling (Inversion)", "progress": 0.6})
        inversion_results = pipeline.modeling.invert(processed_data)

        # Stage 4: Anomaly Detection
        job_store.update_job(job_id, {"stage": "Anomaly Detection", "progress": 0.8})
        anomalies = pipeline.modeling.detect_anomalies(inversion_results)

        # Stage 5: Visualization
        job_store.update_job(job_id, {"stage": "Visualization", "progress": 0.9})
        visualizations = pipeline.visualization.generate(
            processed_data, inversion_results, anomalies, output_dir=str(job_output_dir)
        )

        # Store results
        job_store.update_job(job_id, {
            "status": "COMPLETED",
            "progress": 1.0,
            "stage": "Finished",
            "results": {
                "raw_data": raw_data,
                "processed_data": processed_data,
                "inversion_results": inversion_results,
                "anomalies": anomalies,
                "visualizations": {k: str(v) for k, v in visualizations.items()}
            },
            "output_files": {k: str(v) for k, v in visualizations.items()},
            "message": f"Analysis completed successfully. Found {len(anomalies)} anomalies."
        })
    
        # Cleanup
        pipeline.close()

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
        if 'pipeline' in locals():
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