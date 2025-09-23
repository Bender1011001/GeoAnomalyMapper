#!/usr/bin/env python3
# build_release.py - Release build automation for GeoAnomalyMapper (GAM)
# Automates version update, changelog, packaging, signing, Docker build, GitHub release, and tagging.
# Prerequisites: twine, docker, gh CLI, GPG key for signing.
# Usage: python release/build_release.py --version 1.0.0 [--dry-run]

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

GAM_DIR = Path(__file__).parent.parent
SETUP_PY = GAM_DIR / "setup.py"
CHANGELOG_MD = GAM_DIR / "CHANGELOG.md"
DIST_DIR = GAM_DIR / "dist"
DOCKERFILE = GAM_DIR / "deployment" / "docker" / "Dockerfile"
VERSION = None

def check_prerequisites(dry_run):
    """Check required tools."""
    tools = {
        'twine': ['twine', '--version'],
        'docker': ['docker', '--version'],
        'gh': ['gh', '--version'],
        'gpg': ['gpg', '--version']
    }
    for tool, cmd in tools.items():
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ {tool} available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            if dry_run:
                print(f"DRY-RUN: {tool} check skipped")
            else:
                print(f"✗ {tool} not found or not working. Install it.")
                sys.exit(1)

def run_cmd(cmd, cwd=None, dry_run=False, check=True):
    """Run shell command with dry-run support."""
    full_cmd = f"cd {cwd or '.'} && {cmd}" if cwd else cmd
    if dry_run:
        print(f"DRY-RUN: {full_cmd}")
        return 0
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error executing '{cmd}': {result.stderr}")
        sys.exit(result.returncode)
    if result.stdout:
        print(result.stdout)
    return result.returncode

def update_setup_version(version):
    """Update version in setup.py."""
    with open(SETUP_PY, 'r') as f:
        content = f.read()
    old_version = 'version="0.1.0"'  # Update if changed
    new_line = f'version="{version}"'
    content = content.replace(old_version, new_line)
    with open(SETUP_PY, 'w') as f:
        f.write(content)
    print(f"Updated setup.py version to {version}")

def generate_changelog_entry(version):
    """Append changelog entry."""
    entry = f"""
## [{version}] - {datetime.now().strftime('%Y-%m-%d')}

### Added
- Final integration testing and deployment preparation completed.

### Changed
- Enhanced CI/CD with security scanning and performance tests.

### Fixed
- All known issues from development phase.

### Security
- Dependency scanning integrated.
- Hardening guides documented.

[Unreleased]: https://github.com/GeoAnomalyMapper/GAM/compare/v{version}...develop
"""
    with open(CHANGELOG_MD, 'a') as f:
        f.write(entry)
    print(f"Appended changelog entry for v{version}")

def build_python_packages(version, dry_run):
    """Build sdist and wheel."""
    run_cmd("python setup.py clean --all", dry_run=dry_run)
    run_cmd(f"python setup.py sdist bdist_wheel -d {DIST_DIR}", dry_run=dry_run)
    packages = list(DIST_DIR.glob("GeoAnomalyMapper-*.tar.gz")) + list(DIST_DIR.glob("GeoAnomalyMapper-*.whl"))
    print(f"Built packages: {packages}")
    return packages

def sign_packages(packages, dry_run):
    """Sign packages with GPG."""
    signed = []
    for pkg in packages:
        sig_file = pkg.with_suffix('.asc')
        run_cmd(f"gpg --detach-sign --armor -o {sig_file} {pkg}", dry_run=dry_run)
        signed.append(sig_file)
    print(f"Signed packages: {signed}")
    return signed

def upload_to_pypi(packages, signed, dry_run):
    """Upload to PyPI using twine."""
    for pkg in packages + signed:
        run_cmd(f"twine upload {pkg}", dry_run=dry_run)

def build_docker_image(version, dry_run):
    """Build and push Docker image."""
    image_name = "geoanomalymapper/gam"
    tag = f"{image_name}:{version}"
    latest_tag = f"{image_name}:latest"
    
    run_cmd(f"docker build -t {tag} -f {DOCKERFILE} {GAM_DIR}", dry_run=dry_run)
    run_cmd(f"docker tag {tag} {latest_tag}", dry_run=dry_run)
    
    # Push (assume logged in)
    run_cmd(f"docker push {tag}", dry_run=dry_run)
    run_cmd(f"docker push {latest_tag}", dry_run=dry_run)
    print(f"Docker image pushed: {tag}, {latest_tag}")

def create_github_release(version, packages, signed, dry_run):
    """Create GitHub release using gh CLI."""
    notes_file = GAM_DIR / "release-notes.md"
    with open(notes_file, 'w') as f:
        f.write(f"# GAM v{version}\n\nSee CHANGELOG.md for details.\n")
    
    assets = [str(p) for p in packages + signed]
    assets_str = " ".join(assets)
    
    run_cmd(f"gh release create v{version} release-notes.md {assets_str}", cwd=GAM_DIR, dry_run=dry_run)
    print(f"GitHub release created: v{version}")

def tag_and_push(version, dry_run):
    """Create git tag and push."""
    run_cmd(f"git add setup.py CHANGELOG.md", dry_run=dry_run)
    run_cmd(f'git commit -m "Release v{version}"', dry_run=dry_run)
    run_cmd(f"git tag v{version}", dry_run=dry_run)
    run_cmd("git push origin main", dry_run=dry_run)
    run_cmd(f"git push origin v{version}", dry_run=dry_run)
    print(f"Tagged and pushed v{version}")

def main(version, dry_run):
    check_prerequisites(dry_run)
    
    update_setup_version(version)
    generate_changelog_entry(version)
    
    packages = build_python_packages(version, dry_run)
    signed = sign_packages(packages, dry_run)
    
    upload_to_pypi(packages, signed, dry_run)
    build_docker_image(version, dry_run)
    create_github_release(version, packages, signed, dry_run)
    tag_and_push(version, dry_run)
    
    print(f"Release v{version} built and published successfully!")
    print("Next: Update docs, announce, monitor production.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build and release GAM")
    parser.add_argument('--version', required=True, help='Release version (e.g., 1.0.0)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without executing commands')
    args = parser.parse_args()
    
    main(args.version, args.dry_run)