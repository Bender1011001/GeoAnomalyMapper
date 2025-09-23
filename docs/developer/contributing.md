# Contributing Guide

Thank you for considering contributing to GeoAnomalyMapper (GAM)! We welcome contributions that improve the codebase, documentation, or features. This guide outlines how to set up the development environment, code standards, testing procedures, and the pull request (PR) process.

## Development Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourorg/GeoAnomalyMapper.git
   cd GeoAnomalyMapper
   ```

2. **Create Virtual Environment** (Conda recommended for geospatial deps):
   ```bash
   conda create -n gam-dev python=3.12
   conda activate gam-dev
   # Install system deps if needed (Ubuntu):
   sudo apt install gdal-bin libgdal-dev libgeos-dev proj-bin
   ```

3. **Install Dependencies**:
   ```bash
   # Core + dev extras
   pip install -e .[dev,geophysics,visualization]
   ```
   This installs GAM in editable mode, plus testing (pytest, coverage), linting (Black, Flake8), and docs (Sphinx) tools. See [requirements.txt](../requirements.txt) for details.

4. **Verify Setup**:
   ```bash
   gam --version
   pytest tests/ -v  # Run tests
   black --check .   # Check code style
   ```

5. **Additional Tools**:
   - **IDE**: VS Code with Python extension; install pre-commit hooks: `pre-commit install`.
   - **Dask Dashboard**: For parallel debugging: `dask dashboard` during runs.
   - **Sphinx**: Build docs: `cd docs; make html`.

**Branching**: Use feature branches: `git checkout -b feature/new-modality`.

## Code Style and Standards

GAM follows Python best practices for readability and maintainability.

- **Style Guide**: PEP 8. Use Black for auto-formatting: `black .`.
- **Linting**: Flake8 for errors: `flake8 .`. Ignore E501 (line length) for docstrings.
- **Type Hints**: Required for all public functions/classes (mypy: `mypy .`).
- **Docstrings**: Google style with type annotations. Example:
  ```python
  def fetch_data(bbox: Tuple[float, float, float, float], modalities: List[str]) -> Dict[str, RawData]:
      """
      Fetch raw geophysical data for given bbox and modalities.

      Args:
          bbox: (min_lat, max_lat, min_lon, max_lon)
          modalities: List of 'gravity', 'seismic', etc.

      Returns:
          Dict of modality to RawData.

      Raises:
          DataFetchError: If API call fails.
      """
  ```
- **Commit Messages**: Conventional commits (e.g., "feat: add seismic preprocessor", "fix: resolve GDAL import"). Use 50-char summary, body for details.
- **Pre-commit Hooks**: Enforce style on commit: Includes Black, Flake8, trailing whitespace.

**File Organization**: Follow existing structure (gam/{module}/). Add new files to __init__.py exports.

## Testing Requirements and Procedures

GAM has 95%+ coverage with pytest. Tests in `tests/` mirror modules (e.g., test_ingestion.py).

### Writing Tests
- **Unit Tests**: Mock external APIs (unittest.mock, pytest-mock). Test functions/classes in isolation.
- **Integration Tests**: Use synthetic data in `tests/data/` (e.g., synthetic_gravity.json). Test full pipeline with small bbox.
- **Performance Benchmarks**: In `tests/test_performance.py`; use pytest-benchmark for timing.
- **CLI Tests**: `tests/test_cli_integration.py` with Click testing.
- **Coverage**: Run `pytest --cov=gam --cov-report=html`.

Example unit test:
```python
import pytest
from gam.ingestion.base import DataSource

def test_fetch_data(mock_api):
    source = DataSource()
    with pytest.raises(NotImplementedError):
        source.fetch((0, 10, 0, 10), {})
```

### Running Tests
- All: `pytest tests/ -v --cov=gam`
- Specific: `pytest tests/test_modeling.py`
- Parallel: `pytest -n auto` (pytest-xdist).
- CI: GitHub Actions runs on PRs (unit, integration, coverage >90%).

**Test Data**: Synthetic files in `tests/data/`; real samples via Git LFS if large.

## Pull Request Process

1. **Fork and Branch**: Fork on GitHub, create feature branch from `main`.
2. **Implement**: Code, test, document. Update CHANGELOG.md for new features.
3. **Pre-PR Checks**:
   - Run `pre-commit run --all-files`.
   - `pytest tests/ --cov-fail-under=95`.
   - Build docs: `cd docs; make html` (check for errors).
4. **Commit and Push**:
   ```bash
   git add .
   git commit -m "feat: add new feature\n\nDescription and rationale."
   git push origin feature/new-feature
   ```
5. **Open PR**:
   - Title: "feat: [brief description]" or "fix: [issue]".
   - Body: Link issues (e.g., "Closes #123"), describe changes, screenshots if UI.
   - Label: Use GitHub labels (e.g., enhancement, bug).
6. **Review Process**:
   - CI must pass (tests, linting).
   - At least one maintainer approval.
   - Address feedback in commits/PR comments.
   - For major changes, discuss in issue first.
7. **Merge**: Squash or rebase to main. Delete branch post-merge.

**Issue Reporting**: Use GitHub Issues template. Include repro steps, OS/Python version, logs.

## Code of Conduct

Follow our [Code of Conduct](CODE_OF_CONDUCT.md) for inclusive collaboration.

## Additional Guidelines

- **Documentation**: Update README, add examples in docs/examples/. Use Markdown with syntax highlighting.
- **Performance**: Profile changes with cProfile; avoid regressions.
- **Security**: No secrets in code; use env vars for auth.
- **Releases**: Tag vX.Y.Z on main; update setup.py, changelog.

Questions? Open an issue or ask in discussions. Your contributions make GAM better!

---

*Last Updated: 2025-09-23*