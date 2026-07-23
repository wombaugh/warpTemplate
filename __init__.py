# warp_templates/__init__.py
from .sources import DynamicColorWarpSource, WarpedTimeSeriesSource
from .models import get_warpedTimeSeriesModel
from .loaders import WarpfitTemplateLoader, WarpTemplateDescriptor
from .batch_simulation import (
    WarpSampleSpec,
    WarpSimulationRunner,
    allocate_group_balanced_counts,
)
from .source_cache import WarpSourceCache
from .survey_factory import SurveyConfig, SurveyFactory, SurveyRealizationSpec
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
from .population import (
    DEFAULT_ACTIVE_FITCLASSES_BROAD,
    DEFAULT_ACTIVE_FITCLASSES_IA_SUBTYPE,
    MissingMagnitudePriorError,
    MissingRateError,
    OverlapRateError,
    RateConfigError,
    discover_warp_fitclasses,
    get_magabs_priors,
    load_warp_rate_config,
    rate_config_to_dataframe,
    resolve_rate,
    validate_active_fitclasses,
    validate_magabs_config,
    validate_rate_config,
)
from .observer_population import draw_observer_redshift, observer_expected_count

__all__ = [
    "WarpedTimeSeriesSource",
    "DynamicColorWarpSource",
    "get_warpedTimeSeriesModel",
    "WarpfitTemplateLoader",
    "WarpTemplateDescriptor",
    "WarpSampleSpec",
    "WarpSimulationRunner",
    "allocate_group_balanced_counts",
    "WarpSourceCache",
    "SurveyConfig",
    "SurveyFactory",
    "SurveyRealizationSpec",
    "first_available_source",
    "make_identity_warpdata",
    "make_lightcurve_table",
    "measure_peak_color",
    "plot_lightcurve_table",
    "get_template_correction",
    "TemplateCorrectionResult",
    "FitLCResult",
    "add_warpclasses",
    "TEMPLATE_CLOSE_TYPES",
    "DEFAULT_ACTIVE_FITCLASSES_BROAD",
    "DEFAULT_ACTIVE_FITCLASSES_IA_SUBTYPE",
    "MissingMagnitudePriorError",
    "MissingRateError",
    "OverlapRateError",
    "RateConfigError",
    "draw_observer_redshift",
    "observer_expected_count",
    "discover_warp_fitclasses",
    "get_magabs_priors",
    "load_warp_rate_config",
    "rate_config_to_dataframe",
    "resolve_rate",
    "validate_active_fitclasses",
    "validate_magabs_config",
    "validate_rate_config",
]
