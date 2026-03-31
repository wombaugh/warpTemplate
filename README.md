
`warpTemplate` is a Python package designed to provide **warped supernova (SN) time series templates** and **correction models** using the [sncosmo](https://sncosmo.readthedocs.io/) package. It includes tools for **loading existing templates**, **warping time series data**, and **calculating correction coefficients** to match observed SN data. The package uses advanced techniques such as spline-based warping and template fitting to generate accurate, customized models for SN research.

## Features

- **Warped SN Time Series**: Load existing SN templates and apply warping based on user-supplied data (phase, wavelength, flux).
- **Template Correction**: Correct templates to match observed SN data by calculating correction coefficients and interpolation.
- **Easy-to-use Models**: Create `sncosmo.Model` objects with the corrected templates.
- **Data Handling**: The package works directly with SN data tables (e.g., from ZTF, DES, or other surveys).
- **Reproducible Results**: Optional reproducibility with random seed for template selection.

## Installation

### From PyPI (if released)
```bash
pip install warp_templates
````

### From GitHub (development version)

```bash
git clone https://github.com/your-username/warp_templates.git
cd warp_templates
pip install .
```

### Requirements

* Python >= 3.10
* `numpy`
* `scipy`
* `astropy`
* `sncosmo`
* `matplotlib`

## Usage

### 1. **Load and Warp Time Series Model**

```python
from warp_templates import get_warpedTimeSeriesModel

warpdata = {
    'corrmodel': {
        'phase': [...],
        'wave': [...],
        'flux': [...],
    }
}

# Example of generating a warped time series model
model = get_warpedTimeSeriesModel(
    name="warped_Ia",
    original_template_name="template_name",
    warpdata=warpdata,
    z=0.05,
    hostr_v=3.1
)

# Use the model for further analysis or plotting
```

### 2. **Load Templates and Generate Correction Coefficients**

```python
from warp_templates import WarpfitTemplateLoader, get_template_correction

# Load templates
loader = WarpfitTemplateLoader("warpcoeffs/")
templates = loader.get_templates("Ia")

# Compute correction for a template based on data (astropy Table)
tab = ...  # Your SN data table
correction = get_template_correction(tab, "Ia_template", z=0.05)

# Use the correction model for further work
```

### 3. **Apply Template Correction**

```python
# The correction model is a 2D spline object that you can apply to your data
corrmodel = correction['corrmodel']
flux_corrected = corrmodel(some_phase_values, some_wave_values)
```

## Functions

* `get_warpedTimeSeriesModel(name: str, original_template_name: str, warpdata: dict, ...)`: Creates an `sncosmo.Model` using the warped template data.
* `WarpfitTemplateLoader`: A class for loading and filtering warp coefficients for different supernova templates.
* `get_template_correction(tab: Table, templatename: str, z: float, ...)`: Computes the correction coefficients to match a template to observed SN data, including outlier rejection and interpolation.

## Example Workflow

1. **Load warp coefficients** using `WarpfitTemplateLoader`.
2. **Select a SN template** and compute the correction to match observed SN data using `get_template_correction`.
3. **Generate the warped model** using `get_warpedTimeSeriesModel`.
4. **Apply the correction model** to the observed SN data for analysis or fitting.

## Plotting

The package supports plotting of both the model and the corrected data. For example, you can save correction plots to disk:

```python
import matplotlib.pyplot as plt

# Plot model data
fig = sncosmo.plot_lc(tab, model=model, errors=result.errors)
fig.savefig("model_plot.png")
```

## Tests

This package includes a set of tests to ensure correct functionality. To run the tests:

```bash
pytest
```

## License

MIT License. See `LICENSE` for more details.

## Acknowledgments

This package leverages the [sncosmo](https://sncosmo.readthedocs.io/) library for SN model fitting and flux computation.

