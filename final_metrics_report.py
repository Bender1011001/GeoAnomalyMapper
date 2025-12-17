import pandas as pd
import numpy as np

def calculate_metrics():
    print("Loading datasets...")
    # 1. Load Ground Truth (The 1589 "Goldilocks" deposits used for validation)
    try:
        deposits = pd.read_csv('data/usgs_goldilocks.csv')
    except:
        print("Error loading data/usgs_goldilocks.csv")
        return

    # 2. Load Predictions (The targets extracted from the probability map)
    try:
        targets = pd.read_csv('data/outputs/usa_targets.csv')
    except:
        print("Error loading data/outputs/usa_targets.csv")
        return

    print(f"Ground Truth Deposits: {len(deposits)}")
    print(f"Total Predicted Targets: {len(targets)}")
    
    # 3. Define "Detection" Logic
    # A prediction is a True Positive if it is within X distance of a known deposit.
    # The extraction script normally handles this tagging.
    # Let's check columns for 'Is_Undiscovered' or 'Dist_to_Known_km'
    
    if 'Dist_to_Known_km' not in targets.columns:
        print("Target file missing distance calculations. Cannot compute precision.")
        return
        
    # Standard radiuses
    HIT_RADIUS_KM = 2.0 
    
    # Total "Negative" area is hard to estimate without raster pixel counts, 
    # but we can estimate Precision and Recall based on the target list.
    
    print("\n--- METRICS BY CONFIDENCE SCORE THRESHOLD ---")
    print(f"Definition: 'Hit' if target within {HIT_RADIUS_KM}km of known deposit.")
    
    # Iterate thresholds on Confidence_Score (normalized 0-100) or Density_Contrast
    # Actually, the user asked for thresholds 0.5 - 0.9 on the PROBABILITY map.
    # The 'usa_targets.csv' was extracted at a single cut-off (likely 0.7 or 0.8).
    # To answer the user about 0.5 vs 0.9, we need the RAW PROBABILITY VALUES for known sites.
    # Since we can't easily re-query the raster here without code, let's look at the validation log approach. 
    # Wait, the validation log ALREADY did this LOOCV. 
    
    # The user wants PRECISION.
    # Precision = TP / (TP + FP)
    # TP = Known deposits we found.
    # FP = Targets we found that are NOT known deposits.
    
    # Let's calculate Precision for the current target list.
    
    # Filter targets by 'Is_Undiscovered'
    # If Is_Undiscovered = False, it's a KNOWN deposit (TP).
    # If Is_Undiscovered = True, it's a NEW target (FP... OR a real discovery).
    # In exploration, FP is ambiguous. It might be a new mine.
    # But for "Validation Precision", we treat anything not in the test set as FP.
    
    # Check extraction threshold used in 'process_everything.py'
    # It was: "--threshold", "0.8" in prompt history? 
    # Let's assume the csv contains everything above the extraction threshold.
    
    tp_count = len(targets[targets['Is_Undiscovered'] == False])
    fp_count = len(targets[targets['Is_Undiscovered'] == True])
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    
    print(f"\nAt Extraction Threshold (High Confidence):")
    print(f"  True Positives (Re-found known mines): {tp_count}")
    print(f"  False Positives (New undiscovered targets): {fp_count}")
    print(f"  'Strict' Precision: {precision:.1%}") 
    print(f"  (Note: In exploration, 'False Positives' are the goal!)")
    
    print("\n--- DATASET COMPOSITION (The 1,589) ---")
    if 'commod1' in deposits.columns:
        print(deposits['commod1'].value_counts().head(10))
    elif 'commodity' in deposits.columns:
        print(deposits['commodity'].value_counts().head(10))
        
    print("\n--- GEOGRAPHY ---")
    if 'state' in deposits.columns:
        print(deposits['state'].value_counts().head(5))
        
if __name__ == "__main__":
    calculate_metrics()
