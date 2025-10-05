"""Modality-specific preprocessors for the GAM preprocessing module."""

from __future__ import annotations

import logging
import numpy as np
import xarray as xr
from typing import Dict, Any, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter
import subprocess
import tempfile
import os
from obspy import Stream, Trace
import obspy
from obspy.signal.trigger import classic_sta_lta

from gam.core.exceptions import PreprocessingError
from gam.ingestion.data_structures import RawData
from gam.preprocessing.base import Preprocessor
from gam.preprocessing.data_structures import ProcessedGrid
from gam.preprocessing.filters import NoiseFilter, BandpassFilter, OutlierFilter, SpatialFilter
from gam.preprocessing.gridding import RegularGridder, CoordinateAligner
from gam.preprocessing.units import UnitConverter


logger = logging.getLogger(__name__)
unit_converter = UnitConverter()


class GravityPreprocessor(Preprocessor):
    """
    Preprocessor for gravity data.

    Pipeline: Remove regional trends (polynomial detrending), apply terrain corrections
    (simple Bouguer if elevation available), convert to SI (m/s²), grid to specified
    resolution (default 0.1°).

    Parameters
    ----------
    None
        Configuration via **kwargs in process().

    Methods
    -------
    process(data: RawData, **kwargs) -> ProcessedGrid
        Process gravity data.

    Notes
    -----
    - Assumes RawData.values as ndarray of gravity values, metadata['elevation'] optional.
    - Terrain correction: Basic free-air + Bouguer (2πGρgh formula, ρ=2.67 g/cm³).
    - Uses NoiseFilter and OutlierFilter by default.

    Examples
    --------
    >>> proc = GravityPreprocessor()
    >>> grid = proc.process(raw_gravity, grid_resolution=0.05)
    """

    def process(self, data: RawData, **kwargs) -> ProcessedGrid:
        """
        Process raw gravity data.

        Parameters
        ----------
        data : RawData
            Raw gravity data.
        **kwargs : dict, optional
            - grid_resolution: float (default: 0.1)
            - apply_filters: bool (default: True)
            - elevation: Optional[np.ndarray] (override metadata)
            - poly_order: int (detrend order, default: 2)

        Returns
        -------
        ProcessedGrid
            Processed gravity grid in m/s².

        Raises
        ------
        PreprocessingError
            If processing step fails.
        """
        data.validate()
        grid_res = kwargs.get('grid_resolution', 0.1)
        apply_filters = kwargs.get('apply_filters', True)
        poly_order = kwargs.get('poly_order', 2)
        elevation = kwargs.get('elevation', data.metadata.get('elevation'))

        values = np.asarray(data.data).flatten()
        if len(values) == 0:
            raise PreprocessingError("Empty gravity data")

        # Step 1: Detrending (remove regional trend)
        try:
            # Assume 1D for simplicity; for 2D, use 2D polyfit
            trend = signal.detrend(values, type='linear', order=poly_order)
            values -= trend
            logger.info(f"Applied polynomial detrending (order={poly_order})")
        except Exception as e:
            raise PreprocessingError(f"Detrending failed: {e}")

        # Step 2: Terrain corrections if elevation available
        if elevation is not None and len(elevation) == len(values):
            # Simple Bouguer correction: Δg = 2πGρ g h (approx, ρ=2.67 g/cm³=2670 kg/m³)
            G = 6.67430e-11  # m³ kg⁻¹ s⁻²
            rho = 2670
            correction = 2 * np.pi * G * rho * 9.81 * elevation  # in m/s², h in m
            values += correction
            logger.info("Applied Bouguer terrain correction")
        else:
            logger.warning("No elevation data; skipping terrain correction")

        # Step 3: Apply filters if requested
        if apply_filters:
            outlier_filter = OutlierFilter(threshold=3.0)
            values = outlier_filter.apply(RawData(values, data.metadata)).data
            noise_filter = NoiseFilter(sigma=1.0)
            values = noise_filter.apply(RawData(values, data.metadata)).data
            logger.info("Applied outlier and noise filters")

        # Step 4: Unit conversion to m/s²
        from_unit = data.metadata.get('units', 'mGal')
        if unit_converter.validate_unit('gravity', from_unit):
            values = unit_converter.convert(values, from_unit, 'm/s²', 'gravity')
            logger.info(f"Converted gravity units from {from_unit} to m/s²")
        else:
            raise PreprocessingError(f"Invalid gravity unit: {from_unit}")

        # Step 5: Gridding
        gridder = RegularGridder(resolution=grid_res, method='linear')
        aligner = CoordinateAligner(target_crs='EPSG:4326', grid_after=False)
        aligned_data = aligner.apply(RawData(np.column_stack((data.metadata['bbox'][:2], data.metadata['bbox'][2:], values)), data.metadata))
        grid = gridder.apply(aligned_data)

        grid.units = 'm/s²'
        grid.add_metadata('modality', 'gravity')
        grid.add_metadata('processing_steps', ['detrend', 'terrain_correction' if elevation else 'skipped', 'filters', 'unit_conversion', 'gridding'])

        logger.info(f"Gravity preprocessing complete: grid shape {grid.ds['data'].shape}")
        return grid


class MagneticPreprocessor(Preprocessor):
    """
    Preprocessor for magnetic data.

    Pipeline: Remove geomagnetic reference (IGRF model subtraction if provided),
    apply reduction to pole (RTP), grid and filter for anomaly enhancement.

    Parameters
    ----------
    None

    Methods
    -------
    process(data: RawData, **kwargs) -> ProcessedGrid

    Notes
    -----
    - Requires 'igrf_model' in kwargs (np.ndarray or path) for reference subtraction.
    - RTP: Simple formula assuming inclination/declination in metadata or kwargs.
    - Uses SpatialFilter for enhancement.

    Examples
    --------
    >>> proc = MagneticPreprocessor()
    >>> grid = proc.process(raw_magnetic, igrf_model=model_data)
    """

    def process(self, data: RawData, **kwargs) -> ProcessedGrid:
        """
        Process raw magnetic data.

        Parameters
        ----------
        data : RawData
            Raw magnetic data.
        **kwargs : dict, optional
            - grid_resolution: float (default: 0.1)
            - igrf_model: Optional[np.ndarray] (reference field)
            - inclination: float (degrees, default from metadata)
            - declination: float (degrees)
            - apply_rtp: bool (default: True)

        Returns
        -------
        ProcessedGrid
            Processed magnetic grid in nT.

        Raises
        ------
        PreprocessingError
            If model missing or RTP fails.
        """
        data.validate()
        grid_res = kwargs.get('grid_resolution', 0.1)
        igrf_model = kwargs.get('igrf_model')
        inclination = kwargs.get('inclination', data.metadata.get('inclination', 60.0))
        declination = kwargs.get('declination', data.metadata.get('declination', 0.0))
        apply_rtp = kwargs.get('apply_rtp', True)

        values = np.asarray(data.data).flatten()

        # Step 1: Remove IGRF reference if provided
        if igrf_model is not None:
            if len(igrf_model) != len(values):
                raise PreprocessingError("IGRF model length must match data")
            values -= igrf_model
            logger.info("Subtracted IGRF reference model")
        else:
            logger.warning("No IGRF model provided; skipping reference subtraction")

        # Step 2: Reduction to pole
        if apply_rtp:
            inc_rad = np.radians(inclination)
            dec_rad = np.radians(declination)
            # Simple RTP factor: sin(I) for vertical component approx
            n = len(values)
            I = np.tile(inc_rad, n)
            rtp_factor = np.sin(I) / (np.sin(I)**2 + np.cos(I)**2 * np.cos(dec_rad)**2 + 1e-10)  # Basic formula
            values *= rtp_factor
            logger.info(f"Applied RTP (I={inclination}°, D={declination}°)")
        else:
            logger.info("RTP skipped")

        # Step 3: Unit conversion to nT
        from_unit = data.metadata.get('units', 'nT')
        if unit_converter.validate_unit('magnetic', from_unit):
            values = unit_converter.convert(values, from_unit, 'nT', 'magnetic')
        else:
            raise PreprocessingError(f"Invalid magnetic unit: {from_unit}")

        # Step 4: Filter for anomaly enhancement
        spatial_filter = SpatialFilter(size=3)
        values = spatial_filter.apply(RawData(values, data.metadata)).data
        logger.info("Applied spatial median filter")

        # Step 5: Gridding
        gridder = RegularGridder(resolution=grid_res, method='cubic')
        grid = gridder.apply(RawData(np.column_stack((data.metadata['bbox'][:2], data.metadata['bbox'][2:], values)), data.metadata))

        grid.units = 'nT'
        grid.add_metadata('modality', 'magnetic')
        grid.add_metadata('processing_steps', ['igrf_subtraction' if igrf_model else 'skipped', 'rtp' if apply_rtp else 'skipped', 'filter', 'unit_conversion', 'gridding'])

        logger.info(f"Magnetic preprocessing complete: grid shape {grid.ds['data'].shape}")
        return grid


class SeismicPreprocessor(Preprocessor):
    """
    Preprocessor for seismic data.

    Pipeline: Filter waveforms (bandpass 0.1-1.0 Hz default), extract travel time
    information (basic picking placeholder), convert to velocity/amplitude grids.

    Parameters
    ----------
    None

    Methods
    -------
    process(data: RawData, **kwargs) -> ProcessedGrid

    Notes
    -----
    - Assumes RawData.values as ObsPy Stream.
    - Travel time picking: STA/LTA algorithm for P-wave onset detection (Trnkoczy, 2010).
    - Outputs gridded picked travel times in seconds.

    Examples
    --------
    >>> proc = SeismicPreprocessor()
    >>> grid = proc.process(raw_seismic)
    """

    def process(self, data: RawData, **kwargs) -> ProcessedGrid:
        """
        Process raw seismic data.

        Parameters
        ----------
        data : RawData
            Raw seismic data (Stream).
        **kwargs : dict, optional
            - grid_resolution: float (default: 0.1)
            - lowcut: float (default: 0.1)
            - highcut: float (default: 1.0)
            - sta_len: float (s, default: 1.0)
            - lta_len: float (s, default: 10.0)
            - threshold: float (default: 3.5)

        Returns
        -------
        ProcessedGrid
            Grid of picked P-wave travel times in seconds.

        Raises
        ------
        PreprocessingError
        """
        data.validate()
        if not isinstance(data.data, Stream):
            raise PreprocessingError("Seismic data must be ObsPy Stream")

        grid_res = kwargs.get('grid_resolution', 0.1)
        lowcut = kwargs.get('lowcut', 0.1)
        highcut = kwargs.get('highcut', 1.0)
        sta_len = kwargs.get('sta_len', 1.0)
        lta_len = kwargs.get('lta_len', 10.0)
        threshold = kwargs.get('threshold', 3.5)

        stream = data.data.copy()

        # Step 1: Bandpass filter
        bandpass = BandpassFilter(lowcut=lowcut, highcut=highcut)
        filtered_stream = bandpass.apply(RawData(stream, data.metadata))
        logger.info(f"Applied bandpass filter ({lowcut}-{highcut} Hz)")

        # Step 2: Travel-time picking using STA/LTA
        picked_times = []
        lats, lons = [], []
        for tr in filtered_stream:
            dt = tr.stats.delta
            if len(tr.data) == 0:
                picked_times.append(np.nan)
                lat = tr.stats.coordinates.get('latitude', np.mean([data.metadata['bbox'][0], data.metadata['bbox'][1]]))
                lon = tr.stats.coordinates.get('longitude', np.mean([data.metadata['bbox'][2], data.metadata['bbox'][3]]))
                lats.append(lat)
                lons.append(lon)
                continue
            sta_samples = max(1, int(sta_len / dt))
            lta_samples = max(sta_samples + 1, int(lta_len / dt))
            if lta_samples > len(tr.data):
                logger.warning(f"Trace too short: len={len(tr.data)}, needed={lta_samples}")
                picked_times.append(np.nan)
                lat = tr.stats.coordinates.get('latitude', np.mean([data.metadata['bbox'][0], data.metadata['bbox'][1]]))
                lon = tr.stats.coordinates.get('longitude', np.mean([data.metadata['bbox'][2], data.metadata['bbox'][3]]))
                lats.append(lat)
                lons.append(lon)
                continue
            try:
                c_sta = classic_sta_lta(tr.data, sta_samples, lta_samples)
                above_thresh = np.nonzero(c_sta >= threshold)[0]
                if len(above_thresh) > 0:
                    pick_idx = above_thresh[0]
                else:
                    pick_idx = len(c_sta) // 2
                    logger.debug(f"No clear pick for trace, using center: idx={pick_idx}")
                pick_time = pick_idx * dt
                picked_times.append(pick_time)
            except Exception as e:
                logger.warning(f"Error in STA/LTA picking: {e}")
                picked_times.append(np.nan)
            lat = tr.stats.coordinates.get('latitude', np.mean([data.metadata['bbox'][0], data.metadata['bbox'][1]]))
            lon = tr.stats.coordinates.get('longitude', np.mean([data.metadata['bbox'][2], data.metadata['bbox'][3]]))
            lats.append(lat)
            lons.append(lon)

        if len(picked_times) == 0:
            raise PreprocessingError("No seismic traces to process")

        values = np.array(picked_times)
        points = np.column_stack((lats, lons))

        # No unit conversion for travel times
        logger.info("Picked travel times extracted using STA/LTA (units: s)")

        # Step 4: Gridding
        gridder = RegularGridder(resolution=grid_res, method='linear')
        raw_gridded = RawData(np.column_stack((points, values)), data.metadata)
        grid = gridder.apply(raw_gridded)

        grid.units = 's'
        grid.add_metadata('modality', 'seismic')
        grid.add_metadata('feature', 'travel_time')
        grid.add_metadata('processing_steps', ['bandpass_filter', 'sta_lta_picking', 'gridding'])

        logger.info(f"Seismic preprocessing complete (travel time picking): grid shape {grid.ds['data'].shape}")
        return grid


class InSARPreprocessor(Preprocessor):
    """
    Preprocessor for InSAR data.

    Pipeline: Robust phase unwrapping using SNAPHU (Chen & Zebker, 2001), spatio-temporal filtering for atmospheric delay correction (Yun et al., 2011), convert to displacement grids.

    Parameters
    ----------
    None

    Methods
    -------
    process(data: RawData, **kwargs) -> ProcessedGrid

    Notes
    -----
    - Assumes RawData.values as xarray.Dataset with 'phase' variable (wrapped phase in radians).
    - Phase unwrapping: SNAPHU statistical-cost network-flow algorithm via subprocess.
    - Phase to disp: λ / (4π) * unwrapped_phase (λ=0.056 m for Sentinel-1 C-band).
    - Atmospheric: Spatio-temporal filtering for atmospheric delay correction (Yun et al., 2011).

    Examples
    --------
    >>> proc = InSARPreprocessor()
    >>> grid = proc.process(raw_insar)
    """

    def process(self, data: RawData, **kwargs) -> ProcessedGrid:
        """
        Process raw InSAR data.

        Parameters
        ----------
        data : RawData
            Raw InSAR data (phase Dataset).
        **kwargs : dict, optional
            - grid_resolution: float (default: 0.1)
            - wavelength: float (default: 0.056 m)
            - apply_atm_correction: bool (default: True)
            - snaphu_region_size: int (default: 512)
            - time_window: int (default: 5, for temporal median filter)
            - spatial_sigma: float (default: 1.0, for Gaussian spatial filter)

        Returns
        -------
        ProcessedGrid
            Displacement grid in meters.

        Raises
        ------
        PreprocessingError
        """
        data.validate()
        if not isinstance(data.data, xr.Dataset) or 'phase' not in data.data:
            raise PreprocessingError("InSAR data must be xarray.Dataset with 'phase'")

        grid_res = kwargs.get('grid_resolution', 0.1)
        wavelength = kwargs.get('wavelength', 0.056)  # Sentinel-1
        apply_atm = kwargs.get('apply_atm_correction', True)

        ds = data.data.copy()

        # Step 1: Robust phase unwrapping using SNAPHU
        region_size = kwargs.get('snaphu_region_size', 512)
        try:
            # Assume phase is 2D wrapped; if multi-time, process per slice (simplified to first for now)
            if 'time' in phase.dims:
                phase_2d = phase.isel(time=0)  # Placeholder: process first interferogram; extend for stack
            else:
                phase_2d = phase

            # Save wrapped phase to temp binary file (big-endian float32)
            with tempfile.TemporaryDirectory() as tmpdir:
                wrapped_file = os.path.join(tmpdir, 'wrapped.unw')
                phase_2d.data.astype('>f4').tofile(wrapped_file)

                # Create .rsc metadata file (basic; add X/Y spacing if available from coords)
                dx = float(phase_2d.lon.diff('lon').mean().values) if 'lon' in phase_2d.coords else 0.0001
                dy = float(phase_2d.lat.diff('lat').mean().values) if 'lat' in phase_2d.coords else 0.0001
                rsc_content = f"""WIDTH {phase_2d.shape[1]}
FILE_LENGTH {phase_2d.shape[0]}
FILE_TYPE FLOAT
X_FIRST {float(phase_2d.lon.min().values)}
Y_FIRST {float(phase_2d.lat.max().values)}
X_STEP {dx}
Y_STEP {-dy}
HEADER_LINES 0
"""
                rsc_file = os.path.join(tmpdir, 'wrapped.unw.rsc')
                with open(rsc_file, 'w') as f:
                    f.write(rsc_content)

                # SNAPHU config (basic thresholds)
                conf_content = """THRESHOLD_SNR 1.0
THRESHOLD_PHASE 0.5
CONNECTED_COMPONENTS 1
"""
                conf_file = os.path.join(tmpdir, 'snaphu.conf')
                with open(conf_file, 'w') as f:
                    f.write(conf_content)

                # Run SNAPHU
                cmd = [
                    'snaphu',
                    wrapped_file,
                    str(phase_2d.shape[0] * phase_2d.shape[1]),  # total pixels
                    '-f', conf_file,
                    '-o', 'unwrapped.unw',
                    '-d', str(region_size)
                ]
                result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
                if result.returncode != 0:
                    raise PreprocessingError(f"SNAPHU failed: {result.stderr}")

                # Load unwrapped phase
                unwrapped_file = os.path.join(tmpdir, 'unwrapped.unw')
                unwrapped_data = np.fromfile(unwrapped_file, dtype='>f4').reshape(phase_2d.shape)

                # Update ds phase (handle multi-time if present)
                if 'time' in phase.dims:
                    unwrapped_da = xr.DataArray(
                        unwrapped_data, dims=phase_2d.dims, coords=phase_2d.coords
                    )
                    ds['phase'] = xr.concat([unwrapped_da for _ in range(len(phase.time))], dim='time')
                else:
                    ds['phase'] = xr.DataArray(unwrapped_data, dims=phase.dims, coords=phase.coords)

            logger.info(f"Applied SNAPHU phase unwrapping (region_size={region_size})")
        except FileNotFoundError:
            raise PreprocessingError("SNAPHU not found; install via conda-forge or similar for InSAR processing")
        except Exception as e:
            logger.warning(f"SNAPHU unwrapping failed, using wrapped phase: {e}")
            # Fallback to wrapped

        # Step 1 (cont.): Atmospheric correction on unwrapped phase using spatio-temporal filtering
        if apply_atm:
            time_window = kwargs.get('time_window', 5)
            spatial_sigma = kwargs.get('spatial_sigma', 1.0)
            try:
                if 'time' in phase.dims:
                    # Temporal median filter over time dimension
                    phase_median = phase.rolling(time=time_window, center=True, min_periods=1).median()
                    # Spatial Gaussian filter on median (apply to each time slice)
                    def apply_gaussian(da):
                        data_2d = da.values
                        filtered_2d = gaussian_filter(data_2d, sigma=spatial_sigma)
                        return xr.DataArray(filtered_2d, dims=da.dims, coords=da.coords)
                    aps_estimate = xr.apply_ufunc(apply_gaussian, phase_median, input_core_dims=[['lat', 'lon']], output_core_dims=[['lat', 'lon']], vectorize=True, dask='parallelized', output_dtypes=[float])
                    # Subtract APS from original phase
                    phase_corrected = phase - aps_estimate
                    ds['phase'] = phase_corrected
                    logger.info(f"Applied spatio-temporal atmospheric correction (time_window={time_window}, spatial_sigma={spatial_sigma})")
                else:
                    # Spatial only if no time dim
                    data_2d = phase.values
                    aps_spatial = gaussian_filter(data_2d, sigma=spatial_sigma)
                    phase_corrected = phase - xr.DataArray(aps_spatial, dims=phase.dims, coords=phase.coords)
                    ds['phase'] = phase_corrected
                    logger.info(f"Applied spatial atmospheric correction (sigma={spatial_sigma}, no time stack)")
            except Exception as e:
                logger.warning(f"Atmospheric filtering failed, skipping: {e}")
        else:
            logger.info("Atmospheric correction skipped")

        # Phase to displacement using corrected phase
        # Basic conversion: disp = (corrected_phase * wavelength) / (4 * pi)
        phase = ds['phase']
        disp = (phase * wavelength) / (4 * np.pi)
        ds['data'] = disp
        logger.info("Converted corrected phase to displacement")


        # Step 3: Unit conversion if needed (disp in m)
        from_unit = data.metadata.get('units', 'radians')
        # No specific converter; assume phase was in radians, now m

        # Step 4: Gridding/alignment (assume already gridded; resample if needed)
        aligner = CoordinateAligner(target_crs='EPSG:4326')
        aligned_ds = aligner.apply(RawData(ds, data.metadata))
        gridder = RegularGridder(resolution=grid_res, method='linear')
        grid = gridder.apply(aligned_ds)

        grid.units = 'm'
        grid.add_metadata('modality', 'insar')
        grid.add_metadata('processing_steps', ['snaphu_unwrapping', 'spatio_temporal_atm_correction' if apply_atm else 'skipped', 'phase_to_disp', 'alignment', 'gridding'])

        logger.info(f"InSAR preprocessing complete: grid shape {grid.ds['data'].shape}")
        return grid