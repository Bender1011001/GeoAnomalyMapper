# GeoAnomalyMapper Configuration Presets

This document details the available configuration presets for the GeoAnomalyMapper. These presets allow you to quickly switch between different operational modes, optimizing the system for speed, quality, or specific geological targets.

## Available Presets

| Preset | Config File | Goal | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **Default** | `config.yaml` | Balanced baseline | Standard parameters for general testing. |
| **Fast Mode** | `config_fast.yaml` | Speed (<2 min) | **Zero overlap** (stride=64), low epochs (3), high batch size. *Note: May show grid artifacts.* |
| **Production** | `config_production.yaml` | Quality | **High overlap** (stride=16), high epochs (50), conservative learning rate. Smoothest output. |
| **Mining** | `config_mining.yaml` | Discovery | **Large chip size** (128) for context, **Z-Score normalization** for outlier sensitivity. |

## Detailed Comparison

| Parameter | Default | Fast Mode | Production Mode | Mining Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Chip Size** | 64 | 64 | 64 | **128** |
| **Stride** | 32 | 64 | **16** | 32 |
| **Batch Size** | 128 | **512** | 64 | 64 |
| **Epochs** | 15 | 3 | **50** | 30 |
| **Learning Rate** | 0.001 | 0.005 | 0.0005 | 0.0002 |
| **Normalization** | MinMax | MinMax | MinMax | **Z-Score** |
| **Est. Runtime** | ~15 min | <2 min | ~60 min | ~30 min |

## Usage

To run the analysis with a specific preset, use the `--config` argument:

### 1. Fast Mode (Quick Check)
Use this to verify the pipeline runs end-to-end without waiting.
```bash
python run_analysis.py --config config_fast.yaml
```

### 2. Production Mode (Final Map)
Use this for the final deliverable. It produces the highest quality, seamless map.
```bash
python run_analysis.py --config config_production.yaml
```

### 3. Mining Mode (Deep Exploration)
Use this to find subtle or large-scale geological features (e.g., mineral deposits).
```bash
python run_analysis.py --config config_mining.yaml
```

## Decision Guide

```mermaid
graph TD
    A[Start Analysis] --> B{What is your goal?}
    B -->|Quick Test / Debug| C[Fast Mode]
    B -->|High Quality Map| D[Production Mode]
    B -->|Find Deposits / Voids| E[Mining Mode]
    
    C --> F[config_fast.yaml]
    D --> G[config_production.yaml]
    E --> H[config_mining.yaml]
    
    style F fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style G fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style H fill:#fff3e0,stroke:#ef6c00,stroke-width:2px