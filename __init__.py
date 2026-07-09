# warp_templates/__init__.py
from .sources import WarpedTimeSeriesSource
from .models import get_warpedTimeSeriesModel
from .loaders import WarpfitTemplateLoader
from .lightcurves import (
    first_available_source,
    make_identity_warpdata,
    make_lightcurve_table,
    measure_peak_color,
    plot_lightcurve_table,
)
from .corrections import get_template_correction
from .types import TemplateCorrectionResult, FitLCResult
from .taxonomy import add_warpclasses, TEMPLATE_CLOSE_TYPES

__all__ = [
    "WarpedTimeSeriesSource",
    "get_warpedTimeSeriesModel",
    "WarpfitTemplateLoader",
    "first_available_source",
    "make_identity_warpdata",
    "make_lightcurve_table",
    "measure_peak_color",
    "plot_lightcurve_table",
    "get_template_correction",
    "TemplateCorrectionResult",
    "FitLCResult",
    "add_warpclasses",
]
