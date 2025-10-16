# Build Instructions for Continental-Scale Underground Anomaly Detection Paper

## Requirements

### LaTeX Distribution
- **Recommended**: TeX Live 2022 or later
- **Alternative**: MiKTeX 2022 or later
- **Required packages**: agujournal2019, graphicx, amsmath, amssymb, booktabs, hyperref, url

### Bibliography
- BibTeX support required
- Bibliography style: agufull08 (included with AGU template)

## Compilation

### Recommended: Using latexmk
```bash
# Navigate to papers/agu directory
cd papers/agu

# Compile with latexmk (handles multiple passes automatically)
latexmk -pdf -interaction=nonstopmode main.tex

# Clean auxiliary files
latexmk -c

# Clean all generated files including PDF
latexmk -C
```

### Manual Compilation
```bash
# Navigate to papers/agu directory
cd papers/agu

# First pass
pdflatex main.tex

# Process bibliography
bibtex main

# Second pass (resolve citations)
pdflatex main.tex

# Third pass (resolve cross-references)
pdflatex main.tex
```

### Alternative: Using Make
```bash
# If Makefile is present
make pdf

# Clean build artifacts
make clean
```

## Troubleshooting

### Common Issues

**Missing AGU document class:**
```bash
# Download agujournal2019.cls from AGU LaTeX template
# Place in same directory as main.tex
```

**Bibliography not appearing:**
```bash
# Ensure BibTeX run completed successfully
bibtex main
# Check main.blg for errors
```

**Figure not found errors:**
```bash
# Ensure figure files exist in figures/ directory
# Check file extensions (.png, .pdf, .eps)
# Verify case-sensitive filenames on Unix systems
```

**Cross-reference warnings:**
```bash
# Run pdflatex multiple times (2-3 passes)
# Check for duplicate \label{} commands
```

### Dependencies

**Required data sources** (for figure regeneration):
- XGM2019e gravity model: http://icgem.gfz-potsdam.de/tom_longtime
- EMAG2v3 magnetic data: https://www.ngdc.noaa.gov/geomag/emag2/
- NASADEM elevation: https://lpdaac.usgs.gov/products/nasadem_hgtv001/

**Python environment** (for reproducing results):
```bash
# Use the project metadata as the single source of truth
pyenv install 3.10.14  # or rely on system Python >=3.10
pyenv local 3.10.14    # optional

# Install dependencies declared in pyproject.toml
pip install -e .[all]
```

## Output

Successful compilation produces:
- `main.pdf`: Final manuscript
- Supporting files: `main.aux`, `main.bbl`, `main.blg`, `main.log`

## Submission Preparation

For journal submission:
1. Compile final PDF: `latexmk -pdf main.tex`
2. Verify all references resolve correctly
3. Check figure quality (≥300 DPI for raster images)
4. Ensure all citations appear in references list
5. Validate equation numbering and cross-references

## Version Information

- Document class: agujournal2019
- Journal: Geophysical Research Letters
- LaTeX format: PDF
- Bibliography style: agufull08
- Figures: PNG format (≥300 DPI recommended)

## Support

For AGU-specific LaTeX issues:
- AGU Author Guidelines: https://www.agu.org/Publish-with-AGU/Publish/Author-Resources
- LaTeX template: https://www.agu.org/Publish-with-AGU/Publish/Author-Resources/LaTeX

For manuscript content questions:
- Reproducibility details: See Appendix Section A.1
- Data sources: See Data Availability Statement
- Code repository: [repository URL]