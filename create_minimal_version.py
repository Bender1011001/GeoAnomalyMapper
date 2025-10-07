#!/usr/bin/env python3
"""
Create Ultra-Minimal GeoAnomalyMapper Version

This script removes all framework infrastructure and keeps only the 3 core processing scripts.
Reduces ~150+ files to just 6 essential files.

BEFORE: ~150+ files, complex framework, hundreds of MB
AFTER: 6 files, standalone scripts, minimal dependencies

What gets REMOVED:
- gam/ package (~50+ files) - entire framework
- tests/ directory - all unit tests
- docs/ directory - Sphinx documentation
- deployment/ directory - Docker/K8s/Cloud configs
- dashboard/ - Streamlit web interface
- monitoring/ - Grafana/Prometheus configs
- security/ - scanning scripts
- release/ - release automation
- scripts/ - helper scripts
- notebooks/ - Jupyter notebooks
- configs/ - complex configuration files
- .github/ - GitHub Actions workflows
- All old/unused files

What gets KEPT:
✓ process_global_map_standalone.py - Main pipeline
✓ analyze_results.py - Statistics and visualization
✓ create_globe_overlay.py - Interactive globe generators
✓ pyproject_minimal.toml - Minimal dependencies
✓ README_minimal.md - Instructions
✓ LICENSE - Legal requirement
✓ ../data/ directory - All input/output data

Usage:
    python create_minimal_version.py          # Dry run (shows what will be removed)
    python create_minimal_version.py --apply  # Actually perform the cleanup
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Set

# Console colors for Windows-safe output
class Colors:
    HEADER = ''
    BLUE = ''
    GREEN = ''
    YELLOW = ''
    RED = ''
    ENDC = ''
    BOLD = ''

def get_size_mb(path: Path) -> float:
    """Calculate total size of path in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except (PermissionError, FileNotFoundError):
                        pass
        except (PermissionError, FileNotFoundError):
            pass
        return total / (1024 * 1024)
    return 0.0


def find_items_to_remove(base_dir: Path) -> tuple[List[Path], List[Path]]:
    """Identify files and directories to remove."""
    
    # Directories to remove entirely
    dirs_to_remove = [
        'gam',           # Entire framework package
        'tests',         # Unit tests
        'docs',          # Documentation
        'deployment',    # Docker/K8s configs
        'dashboard',     # Streamlit UI
        'monitoring',    # Grafana/Prometheus
        'security',      # Security scanning
        'release',       # Release automation
        'scripts',       # Helper scripts
        'notebooks',     # Jupyter notebooks
        'config',        # Old config directory
        'configs',       # Old configs directory
        '.github',       # GitHub Actions
        '.ruff_cache',   # Ruff cache
        '.pytest_cache', # Pytest cache
        '__pycache__',   # Python cache
    ]
    
    # Individual files to remove
    files_to_remove = [
        'process_global_map.py',  # Old version (replaced by standalone)
        'integrate_additional_data.py',  # Advanced feature (optional)
        'cleanup_old_files.py',   # Already used, now obsolete
        'data_sources.yaml',      # Not used by standalone
        'env-gam.yml',            # Complex conda environment
        'pyproject.toml',         # Complex version (replaced by minimal)
        'README.md',              # Complex version (replaced by minimal)
        '.env',                   # Environment file
        '.env.example',           # Example env file
        '.gitignore',             # Git-specific
        '.coverage',              # Coverage file
        'dashboard.log',          # Old log file
        'DATA_ACQUISITION_GUIDE.md',  # Can be merged into README
    ]
    
    dir_paths = []
    file_paths = []
    
    # Find directories
    for dir_name in dirs_to_remove:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            dir_paths.append(dir_path)
    
    # Find files
    for file_name in files_to_remove:
        file_path = base_dir / file_name
        if file_path.exists() and file_path.is_file():
            file_paths.append(file_path)
    
    return dir_paths, file_paths


def print_summary(dirs: List[Path], files: List[Path], base_dir: Path):
    """Print summary of what will be removed."""
    print("\n" + "="*70)
    print("ULTRA-MINIMAL VERSION CREATION")
    print("="*70)
    
    total_size = 0.0
    
    print("\nDIRECTORIES TO REMOVE:")
    print("-" * 70)
    if dirs:
        for d in sorted(dirs):
            size = get_size_mb(d)
            total_size += size
            rel_path = str(d.relative_to(base_dir))
            print(f"  [-] {rel_path:<30} ({size:>8.2f} MB)")
    else:
        print("  (none)")
    
    print("\nFILES TO REMOVE:")
    print("-" * 70)
    if files:
        for f in sorted(files):
            size = get_size_mb(f)
            total_size += size
            rel_path = str(f.relative_to(base_dir))
            print(f"  [-] {rel_path:<30} ({size:>8.2f} MB)")
    else:
        print("  (none)")
    
    print("\n" + "="*70)
    print(f"TOTAL TO REMOVE: {len(dirs)} directories + {len(files)} files = {total_size:.2f} MB")
    print("="*70)
    
    print("\nWHAT WILL BE KEPT:")
    print("-" * 70)
    kept_files = [
        ('process_global_map_standalone.py', 'Main processing pipeline (standalone)'),
        ('analyze_results.py', 'Statistics and visualization'),
        ('create_globe_overlay.py', 'Interactive globe generators'),
        ('pyproject_minimal.toml', 'Minimal Python dependencies'),
        ('README_minimal.md', 'Simplified instructions'),
        ('LICENSE', 'Legal requirement'),
        ('../data/', 'All input/output data directories'),
    ]
    
    for filename, description in kept_files:
        print(f"  [+] {filename:<35} - {description}")
    
    print("="*70)


def perform_cleanup(dirs: List[Path], files: List[Path], base_dir: Path):
    """Actually remove the identified files and directories."""
    print("\nPerforming cleanup...")
    
    removed_dirs = 0
    removed_files = 0
    total_freed = 0.0
    
    # Remove directories
    for d in dirs:
        try:
            size = get_size_mb(d)
            shutil.rmtree(d)
            removed_dirs += 1
            total_freed += size
            print(f"  Removed directory: {str(d.relative_to(base_dir))}")
        except Exception as e:
            print(f"  ERROR removing {d}: {e}")
    
    # Remove files
    for f in files:
        try:
            size = get_size_mb(f)
            f.unlink()
            removed_files += 1
            total_freed += size
            print(f"  Removed file: {str(f.relative_to(base_dir))}")
        except Exception as e:
            print(f"  ERROR removing {f}: {e}")
    
    print("\n" + "="*70)
    print("CLEANUP COMPLETE!")
    print("="*70)
    print(f"\nRemoved: {removed_dirs} directories + {removed_files} files")
    print(f"Space freed: {total_freed:.2f} MB")
    print("\n" + "="*70)


def rename_minimal_files(base_dir: Path):
    """Rename _minimal files to replace the originals."""
    print("\nRenaming minimal versions to active names...")
    
    renames = [
        ('pyproject_minimal.toml', 'pyproject.toml'),
        ('README_minimal.md', 'README.md'),
        ('process_global_map_standalone.py', 'process_global_map.py'),
    ]
    
    for old_name, new_name in renames:
        old_path = base_dir / old_name
        new_path = base_dir / new_name
        
        if old_path.exists():
            if new_path.exists():
                print(f"  Replacing: {new_name}")
                new_path.unlink()
            old_path.rename(new_path)
            print(f"  Renamed: {old_name} -> {new_name}")
    
    print("Renaming complete!")


def print_final_structure(base_dir: Path):
    """Print the final minimal directory structure."""
    print("\n" + "="*70)
    print("FINAL MINIMAL STRUCTURE")
    print("="*70)
    print("""
GeoAnomalyMapper/
├── process_global_map.py     # Main pipeline (standalone, no dependencies)
├── analyze_results.py         # Statistics and visualization
├── create_globe_overlay.py    # Interactive globe generators
├── pyproject.toml             # Minimal Python dependencies only
├── README.md                  # Simplified instructions
└── LICENSE                    # MIT license

../data/                       # Data directory (preserved)
├── raw/                       # Source datasets
│   ├── emag2/
│   └── gravity/
└── outputs/                   # All processing outputs
    ├── cog/fused/             # 648 processed tiles
    └── final/                 # Final products (KMZ, HTML, etc.)

TOTAL: 6 essential files + data directories
""")
    print("="*70)
    print("\nREADY TO USE!")
    print("\nQuick Start:")
    print("  1. pip install numpy rasterio affine tqdm matplotlib simplekml")
    print("  2. python process_global_map.py")
    print("  3. python analyze_results.py")
    print("  4. python create_globe_overlay.py")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create ultra-minimal GeoAnomalyMapper version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_minimal_version.py           # Dry run (preview only)
  python create_minimal_version.py --apply   # Actually perform cleanup
        """
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually perform the cleanup (default is dry run)'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.resolve()
    
    print(f"\nScanning directory: {base_dir}")
    
    dirs_to_remove, files_to_remove = find_items_to_remove(base_dir)
    
    print_summary(dirs_to_remove, files_to_remove, base_dir)
    
    if not args.apply:
        print("\n" + "="*70)
        print("DRY RUN MODE - No files were removed")
        print("="*70)
        print("\nTo actually perform the cleanup, run:")
        print(f"  python {Path(__file__).name} --apply")
        print("="*70 + "\n")
        return
    
    # Confirm before proceeding
    print("\n" + "="*70)
    print("WARNING: This will permanently delete the listed files!")
    print("="*70)
    response = input("\nType 'yes' to proceed with cleanup: ").strip().lower()
    
    if response != 'yes':
        print("\nCleanup cancelled.")
        return
    
    # Perform the cleanup
    perform_cleanup(dirs_to_remove, files_to_remove, base_dir)
    
    # Rename minimal files to active names
    rename_minimal_files(base_dir)
    
    # Show final structure
    print_final_structure(base_dir)


if __name__ == '__main__':
    main()