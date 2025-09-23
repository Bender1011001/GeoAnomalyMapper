# GeoAnomalyMapper Pre-Release Validation Checklist

This checklist must be completed before tagging a release. Run through all items and mark [x] when done. If any item fails, fix and re-verify. Use `./scripts/build_release.py` for automation where possible.

## 1. Code Quality and Review
- [ ] All code linted and formatted: `black . && flake8 .` (no errors)
- [ ] Type hints complete: `mypy .` (no errors)
- [ ] Code coverage >95%: `pytest --cov=gam --cov-report=html` (check htmlcov/index.html)
- [ ] No new warnings in static analysis: `bandit -r gam/`
- [ ] Code review completed: PR merged with 2+ approvals, no unresolved comments
- [ ] Commit history clean: Conventional commits, no WIP

## 2. Testing
- [ ] Unit tests pass: `pytest tests/unit/ -v` (100% pass)
- [ ] Integration tests pass: `pytest tests/integration/ -v` (100% pass)
- [ ] System tests pass: `pytest tests/system/ -m integration -v` (deployment, full pipeline)
- [ ] Performance benchmarks meet targets: `pytest tests/system/test_scalability.py -m scalability` (global <2h, memory <6GB)
- [ ] Compatibility tests: `pytest tests/test_compatibility.py` (Python 3.10-3.12)
- [ ] No regressions: Compare benchmarks with previous release

## 3. Documentation
- [ ] Docs build successfully: `cd docs && make html` (no errors, check _build/html/index.html)
- [ ] README.md complete: Installation, quickstart, examples
- [ ] API reference up-to-date: `sphinx-apidoc -o docs/developer/api_reference gam/`
- [ ] User guide complete: Tutorials, configuration reference
- [ ] Deployment guide updated: All envs (Docker, K8s, cloud) with screenshots/commands
- [ ] Changelog updated: `towncrier` or manual for vX.Y.Z changes
- [ ] License and contributors: LICENSE, AUTHORS.md

## 4. Performance Benchmarks
- [ ] Local benchmark: Small region analysis <30s
- [ ] Global benchmark: Full world gravity <2h on 8-core machine
- [ ] Memory profile: Peak <6GB for large dataset (valgrind or memory_profiler)
- [ ] Load test: 10 concurrent users, no errors, response <5s (locust or pytest)
- [ ] Compare with previous release: No degradation >10%

## 5. Security Audit
- [ ] Dependency scan: `python security/scan_dependencies.py --threshold critical` (0 vulns)
- [ ] Image scan: `trivy image gam:latest` (no critical/high)
- [ ] Secrets check: `git-secrets scan` or truffleHog (no committed secrets)
- [ ] Static code analysis: `bandit -r .` (no high severity)
- [ ] Config validation: All production configs load without errors
- [ ] TLS/HTTPS: All endpoints enforce TLS, certs valid

## 6. Packaging and Distribution
- [ ] Build package: `python setup.py sdist bdist_wheel` (no errors)
- [ ] Test install: `pip install dist/GeoAnomalyMapper-*.tar.gz` in clean venv
- [ ] PyPI test upload: `twine upload --repository testpypi dist/*`
- [ ] Docker build: `docker build -t gam:vX.Y.Z deployment/docker/` (push to registry)
- [ ] Version bump: setup.py version = "X.Y.Z"
- [ ] Changelog generated: Include all changes since last release

## 7. Deployment Testing
- [ ] Local deployment: `./scripts/deploy.sh --environment local` (health check pass)
- [ ] Docker deployment: `./scripts/deploy.sh --environment docker` (compose up, ps healthy)
- [ ] K8s deployment: `./scripts/deploy.sh --environment k8s` (kubectl get all, rollout complete)
- [ ] Cloud staging: `./scripts/deploy.sh --environment cloud --platform aws` (stack status CREATE_COMPLETE)
- [ ] Health checks: `./scripts/health_check.py --endpoint https://staging.gam.example.com` (all healthy)
- [ ] Rollback test: Deploy bad version, rollback, verify recovery

## 8. Final Gates
- [ ] All tests pass in CI: GitHub Actions green
- [ ] Documentation deployed: ReadTheDocs build successful
- [ ] Release notes ready: Detailed, user-friendly
- [ ] Announce prepared: Blog post, Twitter, mailing list
- [ ] Maintainer approval: 2+ sign-off

## Release Process
1. Complete checklist, commit fixes.
2. Tag release: `git tag vX.Y.Z && git push --tags`
3. Run `./scripts/build_release.py` to build, sign, upload to PyPI/Docker Hub.
4. Create GitHub release with notes, assets.
5. Deploy to production: `./scripts/deploy.sh --environment cloud --platform prod`
6. Monitor for 24h, hotfix if issues.

For questions, contact release@geoanomalymapper.org.

Last updated: $(date)