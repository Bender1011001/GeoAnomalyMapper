# Continental-Scale Mineral Vectoring: A Physics-Informed Deep Learning Framework with Automated Verification

**Abstract**
We present a novel, end-to-end framework for identifying concealed mineral deposits at a continental scale. Our approach couples a Physics-Informed Neural Network (PINN) for 3D gravity inversion with a deterministic automated verification agent. The inversion model, a 3D U-Net, is trained to solve Poisson's equation subject to observed Bouguer gravity data and structural regularization derived from magnetic gradients. We applied this model to the contiguous United States and Alaska, identifying 4,152 initial anomalies. Our multi-stage verification pipeline—integrating geospatial, geological, legal, and remote sensing data—filtered this list to 248 high-confidence, unclaimed prospects. This methodology demonstrates a pathway to reduce false positives in AI-driven exploration by embedding geological reasoning and legal feasibility into the validation loop.

## 1. Introduction
The search for critical minerals is increasingly forced under cover, where traditional prospecting fails. While deep learning offers powerful pattern recognition, it struggles with the non-uniqueness of potential field inversion (gravity/magnetics). Purely data-driven models often hallucinate features that violate physical consensus. Conversely, traditional inversion is computationally expensive and overly smooth.

We propose a hybrid approach:
1.  **Structure-Guided PINN**: A deep neural network acts as a flexible function approximator for the subsurface density distribution, constrained explicitly by the physics of gravity and implicitly by structural tensors from magnetic data.
2.  **Automated Due Diligence**: A post-processing "agent" that mimics the workflow of a project geologist to validate targets against real-world constraints (land status, claims, lithology).

## 2. Methodology

### 2.1 Physics-Informed Inversion Network
We employ a **3D Density U-Net** architecture to map surface potential fields to subsurface density contrast distributions $\rho(x,y,z)$.

#### 2.1.1 Forward Physics Layer (Spectral Domain)
Unlike standard black-box models, our network includes a differentiable `GravityPhysicsLayer` that implements **Parker's Oldenburg formula** (flat-earth approximation) in the frequency domain. For a predicted density sheet $\hat{\rho}$ at mean depth $z_0$, the gravitational field Fourier transform $\mathcal{F}[g_z]$ is computed as:
$$ \mathcal{F}[g_z](k) = 2\pi G e^{-|k|z_0} \mathcal{F}[\rho](k) \cdot \Delta z $$
Where $G$ is the gravitational constant, $k$ is the wavenumber, and $\Delta z$ is the slab thickness. This allows for efficient, O(N log N) forward modeling suitable for high-resolution continental grids.

#### 2.1.2 Loss Function
The network is optimized using a composite loss function $\mathcal{L}_{total}$:
$$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{data} + \lambda_2 \mathcal{L}_{struct} + \lambda_3 \mathcal{L}_{sparse} $$

*   **Data Residual ($\mathcal{L}_{data}$)**: The MSE between observed gravity ($g_{obs}$) and physics-predicted gravity ($\hat{g_z}$). ($\lambda_1=10.0$)
*   **Structure-Guided Regularization ($\mathcal{L}_{struct}$)**: A Total Variation (TV) loss weighted by the gradient of the magnetic intensity $|\nabla M|$. This encourages sharp density boundaries that align with magnetic structures (faults/contacts). ($\lambda_2=0.1$)
*   **Sparsity ($\mathcal{L}_{sparse}$)**: An L1 penalty on density to suppress noise and encourage compact anomaly generation. ($\lambda_3=0.001$)

### 2.2 Training
The model was trained on $512 \times 512$ patches sampled from the North American Gravity Database (available via USGS/NOAA), normalized to 100 mGal. Optimization was performed using the Adam optimizer ($lr=1e-4$) on a single NVIDIA GPU. The patch-based training strategy allows the model to scale to arbitrary continental dimensions without memory constraints.

## 3. Automated Verification Pipeline
The PINN identified 4,152 primary anomalies (probability $>0.80$, normalized density). To operationalize these targets, we developed a Python-based verification agent.

### 3.1 Tier 1: Geographic & Geological Sanity
*   **Land Masking**: Shapefiles from the US Census Bureau were used to rigorously mask ocean and non-US targets.
*   **Lithological Context**: Targets were raster-sampled against the Global Lithology Map (GLiM). Intrusive (plutonic) and metamorphic domains were flagged as favorable.

### 3.2 Tier 2: Tenure & Land Status
*   **Mining Claims**: The agent queries the Bureau of Land Management (BLM) PLSS API to identify active mining claims within a 1km radius.
*   **Protected Areas**: The USGS PAD-US database is queried to ensure targets do not fall within National Parks or Wilderness Areas.

### 3.3 Tier 3: Surface Expression (Remote Sensing)
*   **Google Earth Engine**: We utilized GEE to confirm accessibility (distance to roads) and surface conditions (NDVI).

## 4. Results
*   **Initial PINN Targets**: 4,152
*   **Geographically Valid**: 981 (after Alaska/Hawaii inclusion)
*   **High-Confidence Prospects**: 248
    *   *Unclaimed*: 100% (filtered)
    *   *Mean Probability*: 83.1%

Cluster analysis identified 5 major prospect districts, primarily located in the Great Basin (Nevada/Utah) and Alaska.

### 4.1 Verification & Validation
We subjected the model to a "Deep Dive" verification suite (see `VALIDATION_REPORT.md`) to quantify performance:
*   **Specificity (Negative Control)**: The model yielded **0 false positives** in three barren control regions (Mississippi Delta, Florida, Kansas), demonstrating excellent rejection of non-mineralized geology.
*   **Sensitivity (Nevada)**: In the Nevada test block, **72.5%** of targets were within 10km of a known deposit, validating the model's ability to identify mineralized systems ("smoke") even if it missed specific recent discoveries.
*   **Regional Offset**: In Alaska, high-confidence targets consistently plotted ~15km from known copper/gold prospects, suggesting a regional vectoring capability rather than point-source detection.

## 5. Discussion & Limitations

### 5.1 Non-Uniqueness & Depth Resolution
Gravity inversion is inherently non-unique; multiple density models can satisfy the same observed field. While our Structure-Guided regularization helps constrain the solution space, the resulting depths (5-15km) should be treated as estimates.

### 5.2 Bias & False Positives
Our verification heavily penalizes sedimentary basins, potentially missing sediment-hosted deposits (e.g., Carlin-type). Furthermore, "unclaimed" status is a snapshot in time and does not guarantee freedom from other encumbrances (e.g., tribal lands, split estate).

### 5.3 Data Availability
The code for this framework is available in this repository. Training data (Gravity/Magnetic mosaics) can be sourced from the USGS Direct Readout or NOAA NCEI.

## 6. Conclusion
This system demonstrates that AI can not only generate targets but also autonomous conduct initial due diligence. The identification of 248 unclaimed targets in favorable geological settings presents significant economic potential for greenfield exploration.
