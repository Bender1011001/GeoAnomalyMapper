#!/usr/bin/env python3
"""
Dummy module for lithology processing compatibility.
Provides create_synthetic_lithology stub - not used in core workflow.
"""

def create_synthetic_lithology(
    output_path: str,
    region: tuple[float, float, float, float],
    resolution: float = 0.001,
    seed: int = 42
) -> None:
    """
    Stub for synthetic lithology generation (not called in pipeline test).
    """
    pass