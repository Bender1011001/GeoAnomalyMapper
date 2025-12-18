import os
import shutil
from pathlib import Path

def archive_files():
    archive_dir = Path("legacy_archive")
    archive_dir.mkdir(exist_ok=True)
    
    # List of patterns or files match the cleanup plan
    files_to_move = [
        # Phase scripts
        "phase6_visualization.py", "phase7_extraction.py", 
        "phase8_spatial_analysis.py", "phase8b_land_mask.py", 
        "phase9_dossier.py", "phase9_visualization.py",
        
        # Batch scripts
        "batch_processor.py", "batch_workflow.py",
        
        # Obsolete Core
        "process_data.py", 
        "run_usa_supervised.py",
        "build_usa_mosaics.py",
        
        # Old Validation
        "validate_mining.py", "validate_california.py", 
        "validate_usa_supervised.py", "validate_against_known_features.py",
        "validation_pa.py", "monitor_validation.py", "create_validation_mask.py",
        
        # Old Classification
        "classify_supervised_optimized.py", "classify_supervised_backup.py", 
        "classify_anomalies.py",
        
        # Research / One-off
        "analyze_deposits.py", "analyze_variogram.py",
        "poisson_analysis.py", "insar_features.py", "detect_voids.py",
        "download_licsar.py", "download_lithology.py", 
        "download_usa_coherence.py", "download_usa_seasonal.py",
        "fetch_data.py", "fetch_lithology_density.py",
        "generate_residual.py", "merge_results.py",
        
        # Utils/Temporary
        "align_rasters.py", "convert_mag_to_float.py",
        "debug_country.py", "debug_probe.py", "test_gpu.py",
        
        # Root Logs/Output
        "extracted_targets.csv", "undiscovered_targets.csv",
        "full_log.txt", "full_log_final.txt", "full_log_restart.txt",
        "full_error.txt", "full_error_final.txt", "full_error_restart.txt",
        "validation_output.txt", "analysis_output.txt", "deposit_analysis.txt",
        "grav_map_info.txt", "mag_info.txt", "prob_map_info.txt", "pipeline.log",
        "full_processing.log", "full_usa_validation.log",
        "validation_report.txt", "validation_report_calibrated.txt", "validation_report_final.txt",
        "verification_results.txt", "verification_final.txt", "debug_error.md"
    ]
    
    print(f"Moving {len(files_to_move)} files to {archive_dir}...")
    
    moved_count = 0
    for fname in files_to_move:
        fpath = Path(fname)
        if fpath.exists():
            dest = archive_dir / fname
            try:
                shutil.move(str(fpath), str(dest))
                print(f"Moved: {fname}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {fname}: {e}")
        else:
            # print(f"Skipping missing: {fname}")
            pass
            
    print(f"\nSuccessfully archived {moved_count} files.")

if __name__ == "__main__":
    archive_files()
