from dataclasses import dataclass
import xarray as xr

@dataclass(frozen=True)
class GridSchema:
    # dims: (lat, lon, depth)
    required_vars = ("lat","lon","depth","data")
    required_attrs = dict(
        # gravity/mag
        obs_elev_m=0.0,
        # mag
        B_T=5e-5, B_inc_deg=60.0, B_dec_deg=0.0,
        # insar
        incidence=0.35, heading=0.0,
    )

def validate(ds: xr.Dataset) -> xr.Dataset:
    for v in GridSchema.required_vars:
        assert v in ds, f"missing var {v}"
    for k, v in GridSchema.required_attrs.items():
        ds.attrs.setdefault(k, v)
    assert ds["lat"].ndim == ds["lon"].ndim == ds["depth"].ndim == 1
    return ds