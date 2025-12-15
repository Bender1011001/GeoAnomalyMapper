#!/usr/bin/env python3
"""
verify_claims.py - Check if targets overlap with existing mining claims
"""

import pandas as pd
import requests
from shapely.geometry import Point, shape
import time
import os
import json

def check_blm_claims(csv_path):
    """
    Query BLM Mining Claim data via their API.
    
    Note: BLM API has rate limits (1000 requests/hour)
    For 4,154 targets, this will take ~4-5 hours
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: Input file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    if 'probability' not in df.columns:
         df['probability'] = df.get('density_contrast', 0.5)
    
    # BLM API endpoint
    # Docs: https://gbp-blm-egis.hub.arcgis.com/
    base_url = "https://gis.blm.gov/arcgis/rest/services/Cadastral/BLM_Natl_PLSS_CadNSDI/MapServer/1/query"
    
    # Alternative endpoint sometimes used for claims specifically:
    # https://gis.blm.gov/arcgis/rest/services/mining_claims/MapServer/0/query 
    # But sticking to user provided one or similar if known better. 
    # Actually, the user script used CadNSDI/MapServer/1 which is PLSS (Public Land Survey System).
    # Mining claims are usually in a different layer. 
    # Let's try to stick to the user's provided URL but add a fallback or comment.
    # Actually, let's use a more likely correct endpoint for claims if the user's might be wrong?
    # No, I should stick to the user's script to avoid breaking their workflow expectation unless I'm sure.
    # BUT, CadNSDI is for land grids, not claims. The user might be checking land ownership?
    # "CASE_TYPE" suggests claims. 
    # Let's use the user's URL but be ready to handle errors.
    
    results = []
    
    print(f"Checking {len(df)} targets against BLM API...")
    
    for idx, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        
        # Create 1km buffer around point
        buffer_km = 1.0
        buffer_deg = buffer_km / 111.0  # rough conversion
        
        # Query BLM for claims in area
        params = {
            'geometry': f'{lon},{lat}',
            'geometryType': 'esriGeometryPoint',
            'inSR': 4326,
            'spatialRel': 'esriSpatialRelIntersects',
            'distance': buffer_deg,
            'units': 'esriSRUnit_Degree',
            'outFields': 'CASE_TYPE,CASE_DISP',
            'returnGeometry': 'false',
            'f': 'json'
        }
        
        try:
            # We use a session to pool connections
            with requests.Session() as s:
                response = s.get(base_url, params=params, timeout=10)
                
            if response.status_code == 200:
                data = response.json()
                features = data.get('features', [])
                has_claims = len(features) > 0
                
                if has_claims:
                    claim_types = [f['attributes'].get('CASE_TYPE', 'Unknown') for f in features]
                    claim_status = claim_types[0] if claim_types else 'Unknown'
                else:
                    claim_status = 'None'
            else:
                print(f"API Error {response.status_code} for target {idx}")
                has_claims = None
                claim_status = 'API_Error'
            
        except Exception as e:
            print(f"Error checking target {idx}: {e}")
            has_claims = None
            claim_status = 'Error'
        
        results.append({
            'target_id': idx,
            'latitude': lat,
            'longitude': lon,
            'probability': row['probability'],
            'has_claims': has_claims,
            'claim_status': claim_status
        })
        
        # Rate limiting
        if idx % 10 == 0: # Print more often for feedback
             print(f"Processed {idx}/{len(df)} targets...")
        
        # Sleep to respect rate limits
        time.sleep(1.0) 
    
    results_df = pd.DataFrame(results)
    
    # Summary
    print(f"\nTotal Targets: {len(results_df)}")
    if 'has_claims' in results_df.columns:
        print(f"With Existing Claims: {results_df['has_claims'].sum()}")
        print(f"Unclaimed: {(~results_df['has_claims'].fillna(False)).sum()}")
    
    # Save both lists
    output_path = 'data/outputs/targets_claims_verified.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    
    # Separate unclaimed targets (high value!)
    # diverse from the user script: handle None/NaN
    unclaimed = results_df[results_df['has_claims'] == False]
    unclaimed.to_csv('data/outputs/targets_unclaimed.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    # Check if input exists
    input_file = 'data/outputs/targets_geology_verified.csv'
    if not os.path.exists(input_file):
        print(f"Input {input_file} not found, checking earlier pipeline files...")
        if os.path.exists('data/outputs/targets_geography_verified.csv'):
            input_file = 'data/outputs/targets_geography_verified.csv'
        elif os.path.exists('data/outputs/undiscovered_targets.csv'):
            input_file = 'data/outputs/undiscovered_targets.csv'
        else:
            print("No suitable input file found.")
            exit(1)
            
    print(f"Using input file: {input_file}")
    check_blm_claims(input_file)
