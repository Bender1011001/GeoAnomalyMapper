import pytest
import yaml
from pathlib import Path

from GeoAnomalyMapper.gam.core.config_service import ConfigService
from GeoAnomalyMapper.gam.core.config_models import AppConfig


def test_load_minimal_config_success():
    # Resolve fixture path relative to this test file
    path = Path(__file__).parent.parent / "data" / "minimal_config.yaml"
    svc = ConfigService()
    cfg = svc.load(str(path))

    assert isinstance(cfg, AppConfig)
    assert cfg.app.output_dir == "./data/outputs"
    assert cfg.app.cache_dir == "./data/cache"
    assert cfg.app.default_modalities == ["gravity", "magnetic"]

    assert cfg.preprocessing.grid_resolution == 0.1
    assert cfg.preprocessing.filters == ["noise"]

    assert cfg.modeling.fusion_method == "joint_inversion"
    assert cfg.modeling.anomaly_threshold == 95.0
    assert cfg.modeling.max_iterations == 20

    assert cfg.feature_flags.enable_cache is True
    assert cfg.feature_flags.enable_parallel is False

    # DataSources is a root-mapped type; check the fixture key exists
    assert "usgs" in cfg.data_sources.__root__


def test_load_invalid_config_extra_key(tmp_path):
    # Create a temporary YAML with an unexpected top-level key (extra="forbid" should reject)
    bad = {"unexpected": 1}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))

    svc = ConfigService()
    with pytest.raises(ValueError):
        svc.load(str(p))