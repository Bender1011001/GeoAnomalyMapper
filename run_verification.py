#!/usr/bin/env python3
"""
run_verification.py - Master script to run all checks
"""

import subprocess
import pandas as pd
import sys
import os

def run_full_verification():
    """
    Run all verification scripts in sequence.
    """
    
    print("="*80)
    print("AUTOMATED TARGET VERIFICATION PIPELINE")
    print("="*80)
    
    python = sys.executable

    # Check input
    initial_input = 'data/outputs/undiscovered_targets.csv'
    if not os.path.exists(initial_input):
        print(f"WARNING: Initial input {initial_input} not found.")
        # Proceeding anyway as scripts might check other files? 
        # Actually verify_geography.py checks, so we let it fail or warn there.

    # Stage 1: Geography (1 minute)
    print("\n[1/6] Validating Geography...")
    subprocess.run([python, 'verify_geography.py'])
    
    # Stage 2: Geology (5 minutes)
    print("\n[2/6] Checking Geological Context...")
    subprocess.run([python, 'verify_geology.py'])
    
    # Stage 3: Mining Claims (4-5 hours due to API rate limits)
    print("\n[3/6] Checking Mining Claims (Large batch may take hours)...")
    subprocess.run([python, 'verify_claims.py'])
    
    # Stage 4: Land Status (30 minutes)
    print("\n[4/6] Checking Land Status...")
    subprocess.run([python, 'verify_land_status.py'])
    
    # Stage 5: Visual Indicators (2-3 hours)
    print("\n[5/6] Running Visual Analysis (Earth Engine)...")
    subprocess.run([python, 'verify_visual.py'])
    
    # Stage 6: Generate Final Report
    print("\n[6/6] Generating Final Report...")
    generate_final_report()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)

def generate_final_report():
    """
    Combine all verification results into final report.
    """
    try:
        # Load all verification outputs
        # We need to handle missing files if a stage failed or was skipped
        
        # Helper to load with ID index
        def load_if_exists(path):
            if os.path.exists(path):
                return pd.read_csv(path)
            return None

        geo = load_if_exists('data/outputs/targets_geography_verified.csv')
        geol = load_if_exists('data/outputs/targets_geology_verified.csv')
        claims = load_if_exists('data/outputs/targets_claims_verified.csv')
        land = load_if_exists('data/outputs/targets_mineable.csv')
        visual = load_if_exists('data/outputs/targets_visual_verified.csv')
        
        if geo is None:
            print("Geography results missing, cannot generate full report.")
            return

        # Start with max available targets (geo is usually the base filter)
        final = geo.copy()

        # Merge iteratively
        if geol is not None:
            final = final.merge(geol[['target_id', 'geology_favorable', 'lithology_code']], on='target_id', how='left')
        else:
            final['geology_favorable'] = False
            
        if claims is not None:
            # handle if target_id missing or mismatch
            final = final.merge(claims[['target_id', 'has_claims', 'claim_status']], on='target_id', how='left')
        else:
            final['has_claims'] = True # Assume worst case if unchecked? Or False?
            
        if land is not None:
            final = final.merge(land[['target_id', 'mineable', 'is_protected', 'protection_type']], on='target_id', how='left')
        else:
            final['mineable'] = False

        if visual is not None:
            final = final.merge(visual[['target_id', 'visual_score', 'ndvi', 'road_access']], on='target_id', how='left')
        else:
            final['visual_score'] = 0

        # Fill NaNs
        final.fillna({'geology_favorable': False, 'has_claims': True, 'mineable': False, 'visual_score': 0}, inplace=True)
        
        # Calculate composite score
        final['composite_score'] = (
            final['valid'].astype(int) * 10 +
            final['geology_favorable'].astype(int) * 20 +
            (~final['has_claims'].astype(bool)).astype(int) * 30 +
            final['mineable'].astype(int) * 20 +
            final['visual_score'] * 4
        )
        
        # Classify targets
        final['tier'] = pd.cut(final['composite_score'], 
                            bins=[-1, 40, 70, 200], 
                            labels=['Low', 'Medium', 'High'])
        
        # Summary stats
        print("\n" + "="*80)
        print("FINAL VERIFICATION SUMMARY")
        print("="*80)
        print(f"Total Targets Analyzed: {len(final)}")
        print(f"\nBy Tier:")
        print(final['tier'].value_counts())
        
        high_tier = final[final['tier'] == 'High']
        if len(high_tier) > 0:
            print(f"\nHigh-Tier Targets: {len(high_tier)}")
            print(f"  - Average Probability: {high_tier['probability'].mean():.1%}")
            print(f"  - Unclaimed: {(~high_tier['has_claims'].astype(bool)).sum()}")
        
        # Save tiered lists
        high_tier.to_csv('data/outputs/FINAL_high_confidence_targets.csv', index=False)
        
        medium_tier = final[final['tier'] == 'Medium']
        medium_tier.to_csv('data/outputs/FINAL_medium_confidence_targets.csv', index=False)
        
        final.to_csv('data/outputs/FINAL_all_targets_verified.csv', index=False)
        
        print(f"\nOutputs saved to data/outputs/FINAL_*.csv")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_verification()
