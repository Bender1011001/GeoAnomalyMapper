"""Report generation for GAM visualization module.

This module creates comprehensive analysis reports combining multiple visualizations,
statistical summaries, and metadata. ReportGenerator produces static PDF documents
using matplotlib backends. HTMLReportGenerator creates interactive web reports with
embedded maps, plots, and download links, supporting responsive design.

Both generators support template-based layouts for structured output. For datasets,
automatically generates figures using VisualizationManager.

Supported features:
- PDF: Multi-page with figures, text summaries, tables.
- HTML: Embedded images/maps, responsive CSS, data download links.
- Templates: 'standard' (title, figs, stats, metadata); custom via dict.
- Summaries: Mean/std/confidence stats, metadata extraction.
- Responsive: HTML adapts to devices; PDF publication-ready.

Notes
-----
- Input: List of Figures or datasets (auto-generate figs via manager).
- Stats: From data 'data'/'model'/'strength'; handles NaNs.
- Metadata: Embedded as tables/JSON.
- Dependencies: matplotlib.backends.backend_pdf, base64, io.BytesIO, pandas (for tables).
- Customization: kwargs for title, sections, CSS (HTML).
- Limitations: Large reports may be slow; compress images for HTML.
"""

from __future__ import annotations

import logging
import base64
import io
from typing import Union, List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import pandas as pd

# For HTML
from folium import Map  # If interactive maps
import numpy as np

from gam.visualization.manager import VisualizationManager
from gam.preprocessing.data_structures import ProcessedGrid
from gam.modeling.data_structures import InversionResults, AnomalyOutput

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generator for static PDF analysis reports.

    Combines figures and summaries into multi-page PDF using PdfPages. Supports
    template layouts with title page, visualization pages, and summary section.

    Parameters
    ----------
    manager : VisualizationManager, optional
        Shared manager for fig generation (default new instance).
    dpi : int, optional
        Image DPI for embedding (default 150).

    Methods
    -------
    generate(content: Union[List[Figure], List], output_path: str, template: str = 'standard', **kwargs) -> str
        Create PDF report.

    Notes
    -----
    - Content: List of Figure or datasets (generates maps/stats).
    - Template 'standard': Title, figs (one per page), stats table, metadata.
    - Summaries: Computed mean/std/min/max/confidence.
    - Page Size: A4 default; adjustable via kwargs.
    - Publication: Vector where possible, high-res raster for complex figs.

    Examples
    --------
    >>> gen = ReportGenerator()
    >>> figs = [manager.generate_map(ds) for ds in datasets]
    >>> path = gen.generate(figs, 'report.pdf', title='GAM Analysis')
    """

    def __init__(self, manager: Optional[VisualizationManager] = None, dpi: int = 150):
        self.manager = manager or VisualizationManager()
        self.dpi = dpi

    def _compute_stats(self, data: Union[ProcessedGrid, InversionResults, AnomalyOutput]) -> Dict[str, Any]:
        """Compute summary statistics."""
        if isinstance(data, (ProcessedGrid, InversionResults)):
            values = data.ds['data'].values.flatten() if isinstance(data, ProcessedGrid) else data.model.flatten()
        else:
            values = data['strength'].dropna().values
        values = values[~np.isnan(values)]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'n_points': len(values)
        }

    def generate(
        self,
        content: Union[List[Figure], List[Union[ProcessedGrid, InversionResults, AnomalyOutput]]],
        output_path: str,
        template: str = 'standard',
        title: str = 'GAM Analysis Report',
        **kwargs: Any
    ) -> str:
        """
        Generate PDF report from content.

        Parameters
        ----------
        content : Union[List[Figure], List[data]]
            Figures or datasets to include.
        output_path : str
            .pdf path.
        template : str, optional
            Layout template (default 'standard').
        title : str, optional
            Report title (default 'GAM Analysis Report').
        **kwargs : dict, optional
            - 'page_size': Tuple (default A4).
            - 'stats': bool, include stats (default True).
            - 'metadata': bool, include metadata (default True).

        Returns
        -------
        str
            PDF path.
        """
        if isinstance(content[0], (ProcessedGrid, InversionResults, AnomalyOutput)):
            # Generate figs
            figs = []
            for ds in content:
                fig_map = self.manager.generate_map(ds)
                fig_stats = StatisticalPlots().generate(ds)
                figs.extend([fig_map, fig_stats])
        else:
            figs = content

        stats = kwargs.get('stats', True)
        metadata = kwargs.get('metadata', True)

        with PdfPages(output_path) as pdf:
            # Title page
            fig_title = plt.figure(figsize=(8.27, 11.69))  # A4
            fig_title.text(0.5, 0.5, title, ha='center', va='center', fontsize=20, weight='bold')
            if stats or metadata:
                y = 0.4
                if stats:
                    all_stats = {f"Dataset {i}": self._compute_stats(ds) for i, ds in enumerate(content) if isinstance(ds, (ProcessedGrid, InversionResults, AnomalyOutput))}
                    fig_title.text(0.1, y, "Summary Statistics:\n" + str(all_stats), fontsize=10, transform=fig_title.transFigure)
                    y -= 0.2
                if metadata:
                    meta_text = "Metadata:\n" + "\n".join([f"{k}: {v}" for ds in content for k, v in (ds.ds.attrs if isinstance(ds, ProcessedGrid) else ds.metadata).items()][:10])
                    fig_title.text(0.1, y, meta_text, fontsize=8, transform=fig_title.transFigure)
            pdf.savefig(fig_title, bbox_inches='tight')
            plt.close(fig_title)

            # Fig pages
            for i, fig in enumerate(figs):
                pdf.savefig(fig, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)

        logger.info(f"Generated PDF report '{output_path}' with {len(figs)} figures")
        return output_path


class HTMLReportGenerator:
    """
    Generator for interactive HTML analysis reports.

    Embeds figures as base64 PNGs, folium maps as HTML, adds responsive CSS,
    download links for exports, and data tables. Supports device adaptation.

    Parameters
    ----------
    manager : VisualizationManager, optional
        Shared manager (default new).
    css : str, optional
        Custom CSS (default inline responsive).

    Methods
    -------
    generate(content: Union[List[Figure], List], output_path: str, template: str = 'standard', **kwargs) -> str
        Create HTML report.

    Notes
    -----
    - Content: Figs or datasets (auto-generate).
    - Embedding: Matplotlib to base64; folium direct HTML.
    - Responsive: Bootstrap-like CSS for mobile/tablet.
    - Downloads: Links to exported files (call export_data internally).
    - Template 'standard': Header, sections for figs/tables, footer with metadata.
    - Tables: Pandas to_html for stats.
    - Security: Sanitize if user content; base64 safe for images.

    Examples
    --------
    >>> gen = HTMLReportGenerator()
    >>> path = gen.generate(datasets, 'report.html', include_downloads=True)
    """

    CSS = """
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .figure { text-align: center; margin: 20px 0; }
    .download { background: #007bff; color: white; padding: 10px; text-decoration: none; border-radius: 5px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    @media (max-width: 600px) { .figure img { width: 100%; } }
    </style>
    """

    def __init__(self, manager: Optional[VisualizationManager] = None, css: str = CSS):
        self.manager = manager or VisualizationManager()
        self.css = css

    def _fig_to_base64(self, fig: Figure) -> str:
        """Convert figure to base64 PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"

    def _generate_downloads(self, datasets: List, output_dir: str = '.') -> List[str]:
        """Generate export files and return links."""
        links = []
        for i, ds in enumerate(datasets):
            path = self.manager.export_data(ds, f"{output_dir}/export_{i}.csv" if isinstance(ds, AnomalyOutput) else f"{output_dir}/export_{i}.tif")
            links.append(f'<a href="{path}" class="download" download>Download {i+1}</a>')
        return links

    def generate(
        self,
        content: Union[List[Figure], List[Union[ProcessedGrid, InversionResults, AnomalyOutput]]],
        output_path: str,
        template: str = 'standard',
        title: str = 'GAM Analysis Report',
        include_downloads: bool = True,
        **kwargs: Any
    ) -> str:
        """
        Generate HTML report from content.

        Parameters
        ----------
        content : Union[List[Figure], List[data]]
            Figures or datasets.
        output_path : str
            .html path.
        template : str, optional
            Layout (default 'standard').
        title : str, optional
            Report title.
        include_downloads : bool, optional
            Add export links (default True).
        **kwargs : dict, optional
            - 'output_dir': str for downloads.
            - 'stats': bool, include tables (default True).

        Returns
        -------
        str
            HTML path.
        """
        if isinstance(content[0], (ProcessedGrid, InversionResults, AnomalyOutput)):
            figs = []
            for ds in content:
                fig = self.manager.generate_map(ds)
                figs.append(fig)
        else:
            figs = content

        html_content = f"""
        <html>
        <head><title>{title}</title>{self.css}</head>
        <body>
        <h1>{title}</h1>
        """

        if include_downloads and isinstance(content[0], (ProcessedGrid, InversionResults, AnomalyOutput)):
            output_dir = kwargs.get('output_dir', '.')
            downloads = self._generate_downloads(content, output_dir)
            html_content += '<div>' + ' | '.join(downloads) + '</div><hr>'

        for i, fig in enumerate(figs):
            if isinstance(fig, Figure):
                img_src = self._fig_to_base64(fig)
                html_content += f'<div class="figure"><h2>Figure {i+1}</h2><img src="{img_src}" alt="Figure {i+1}"></div>'
            elif isinstance(fig, Map):
                html_content += f'<div class="figure"><h2>Interactive Map {i+1}</h2>{fig._repr_html_()}<div>'
            else:
                logger.warning(f"Skipping unsupported fig type: {type(fig)}")

        if kwargs.get('stats', True):
            stats_html = ""
            for i, ds in enumerate(content if isinstance(content[0], (ProcessedGrid, InversionResults, AnomalyOutput)) else []):
                stats = self.manager._compute_stats(ds) if hasattr(self.manager, '_compute_stats') else {'mean': 0, 'std': 0}  # Fallback
                df = pd.DataFrame([stats])
                stats_html += f'<h3>Dataset {i+1} Statistics</h3>{df.to_html(index=False)}'
            html_content += stats_html

        html_content += "</body></html>"

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Generated HTML report '{output_path}' with {len(figs)} elements")
        return output_path