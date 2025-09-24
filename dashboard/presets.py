"""
Preset configurations for GeoAnomalyMapper analysis use cases.
Provides pre-configured settings for common archaeological and geological scenarios,
making the tool accessible to domain experts without deep technical knowledge.
"""

from typing import Dict, List, Optional

def get_all_presets() -> Dict[str, Dict]:
    """
    Get all available analysis presets.
    
    Returns:
        Dict of preset names to their configurations.
    """
    return {
        "Archaeological Survey": {
            "name": "Archaeological Survey",
            "description": "Detect buried archaeological structures, ancient settlements, and subsurface features using gravity and magnetic data for non-invasive surveys.",
            "default_modalities": ["gravity", "magnetic"],
            "recommended_resolution": 750.0,  # meters, average of 500-1000m range
            "typical_bbox_size": "Small site: 0.1-0.5 degrees (e.g., 29.95,30.05,31.10,31.20 for Giza-like areas)",
            "analysis_focus": "Shallow anomalies, cultural features",
            "typical_use_cases": [
                "Site prospection without excavation",
                "Heritage preservation mapping",
                "Integration with ground-penetrating radar"
            ]
        },
        "Regional Fault Mapping": {
            "name": "Regional Fault Mapping",
            "description": "Map large-scale geological structures and fault systems using gravity and seismic data for structural geology analysis.",
            "default_modalities": ["gravity", "seismic"],
            "recommended_resolution": 3500.0,  # meters, average of 2000-5000m
            "typical_bbox_size": "Regional: 1-5 degrees (e.g., 27.0,28.0,85.0,86.0 for Himalayan region)",
            "analysis_focus": "Deep structural features",
            "typical_use_cases": [
                "Tectonic plate boundary studies",
                "Earthquake risk assessment",
                "Global fault network mapping"
            ]
        },
        "Subsidence Monitoring": {
            "name": "Subsidence Monitoring",
            "description": "Monitor ground subsidence and stability issues using InSAR and gravity data for high-precision deformation detection.",
            "default_modalities": ["insar", "gravity"],
            "recommended_resolution": 300.0,  # meters, average of 100-500m
            "typical_bbox_size": "Local to regional: 0.5-2 degrees (focus on urban or vulnerable areas)",
            "analysis_focus": "Surface deformation",
            "typical_use_cases": [
                "Urban subsidence tracking",
                "Landslide risk monitoring",
                "Groundwater extraction impact"
            ]
        },
        "Resource Exploration": {
            "name": "Resource Exploration",
            "description": "Identify potential mineral or hydrocarbon resources using comprehensive gravity, magnetic, and seismic surveys.",
            "default_modalities": ["gravity", "magnetic", "seismic"],
            "recommended_resolution": 1500.0,  # meters, average of 1000-2000m
            "typical_bbox_size": "Basin scale: 2-7 degrees (e.g., 55.0,62.0,0.0,10.0 for North Sea basin)",
            "analysis_focus": "Deep subsurface anomalies",
            "typical_use_cases": [
                "Oil and gas prospecting",
                "Mineral deposit identification",
                "Basement fault mapping for reservoirs"
            ]
        },
        "Environmental Assessment": {
            "name": "Environmental Assessment",
            "description": "Assess environmental contamination and site stability using gravity and InSAR for shallow to medium-depth analysis.",
            "default_modalities": ["gravity", "insar"],
            "recommended_resolution": 600.0,  # meters, average of 200-1000m
            "typical_bbox_size": "Site-specific: 0.2-1 degree (e.g., industrial or contaminated zones)",
            "analysis_focus": "Shallow to medium depth anomalies",
            "typical_use_cases": [
                "Contamination plume mapping",
                "Site stability evaluation",
                "Environmental impact assessments"
            ]
        }
    }

def get_preset(name: str) -> Optional[Dict]:
    """
    Get a specific preset configuration by name.
    
    Args:
        name: The name of the preset.
        
    Returns:
        The preset dict if found, None otherwise.
    """
    presets = get_all_presets()
    return presets.get(name)