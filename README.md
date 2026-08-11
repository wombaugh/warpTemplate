# warptemplate

`warptemplate` provides warped supernova time-series templates for
[sncosmo](https://sncosmo.readthedocs.io/) and tools for turning those
templates into reproducible survey samples and classifier inputs. It supports
three connected workflows:

1. load, inspect, and colour-warp existing template-coefficient libraries;
2. simulate batched ZTF, LSST, or combined training samples with SkySurvey;
3. train and evaluate ParSNIP or SuperNNova classifiers on grouped splits.

## Highlights

- Load coefficient libraries and sample templates by fit class.
- Build reusable, dynamically coloured `sncosmo` sources and models.
- Resolve volumetric rates, magnitude priors, redshift draws, and class counts.
- Construct normalized ZTF, LSST, or synthetic combined survey cadences.
- Cache source grids losslessly in HDF5 and load only the sources needed by a
  simulation batch.
- Stream truth and observation data to schema-6 Parquet datasets with
  deterministic resume support.
- Build leakage-free classifier splits and run reproducible ParSNIP or
  SuperNNova workflows.

## Repository layout

The importable package is located in `warptemplate/`:

- `warptemplate/loaders.py`, `models.py`, and `sources.py`: template loading and warped
  `sncosmo` models;
- `warptemplate/population.py` and `observer_population.py`: rates, class allocation,
  redshifts, and magnitude priors;
- `warptemplate/survey_factory.py`: survey cadence and observed-area construction;
- `warptemplate/source_cache.py` and `batch_simulation.py`: cached, batched sample
  generation;
- `warptemplate/classification.py`, `supernnova_backend.py`, and
  `sequence_rnn.py`: grouped data splits,
  classifier orchestration, evaluation, and recurrent-model support;
- `warptemplate/lightcurves.py`: light-curve tables, measurements, and plotting helpers;
- `warptemplate/corrections.py`: the correction algorithm used by the historical
  template-building workflow.

The reproducible v4 construction stages are command-line programs in `scripts/`.

The notebooks are grouped by purpose:

- [`notebooks/template_creation/`](notebooks/template_creation/) documents how
  coefficient libraries were created and validated;
- [`notebooks/template_usage/`](notebooks/template_usage/) contains the
  practical template and SkySurvey sample workflows;
- [`notebooks/classification/`](notebooks/classification/) contains the
  ParSNIP and SuperNNova training and evaluation workflows.

## Installation

`warptemplate` supports Python 3.11 and 3.12. Clone the canonical repository
and install it from the repository root:

```bash
git clone https://github.com/wombaugh/warpTemplate.git
cd warpTemplate
python -m pip install -e .
```

The base installation includes NumPy, SciPy, Astropy, sncosmo, pandas,
SkySurvey, PyArrow, and HDF5 support. Survey samples additionally require the
local cadence and coefficient files referenced by their configuration.

Install the optional classifier dependencies from the repository root:

```bash
python -m pip install -e '.[classification]'
python -m pip install --upgrade astro-parsnip
```

The `astro-parsnip` distribution provides the astronomical
[LSSTDESC ParSNIP](https://github.com/LSSTDESC/parsnip) module imported as
`parsnip`. The unrelated PyPI distribution named `parsnip` is a Python parser,
does not provide the required APIs, and must not be installed alongside it.
The recurrent workflow uses the repository's local `WarpSequenceRNN`. The
external SuperNNova package is only needed for optional HDF5 compatibility
checks and is deliberately not a runtime dependency.

## Coefficient libraries

Coefficient files are data products and are not included in this Git repository.
Place the v4 files in one directory, conventionally `data/warpcoeff_v4/`. Their
default names are:

```text
warpcoeffs_v4_<fitclass>_col.pkl
```

`WarpfitTemplateLoader` defaults to version `4` and suffix `_col`. For a gradual
migration it also recognizes legacy flat v3 files named
`warpcoeffs_v3_<fitclass>.pkl` when they are present in the configured directory.
This fallback is for compatibility and does not turn v3 data into the new compact
v4 representation.

## Start here

### Use an existing warped template

[`create_usable_template_lightcurves.ipynb`](notebooks/template_usage/create_usable_template_lightcurves.ipynb)
is the main interactive example. It loads a coefficient library, samples
templates, compares colour modes, and converts a warped model into a
light-curve table.

The loader can also be used directly:

```python
from warptemplate import WarpfitTemplateLoader

loader = WarpfitTemplateLoader("data/warpcoeff_v4")
templates = loader.get_templates("SN Ia")
```

The template-creation notebooks are not required for normal package use. Open
them only when the scientific construction and calibration of a coefficient
library needs to be inspected or repeated.

### Generate a SkySurvey training sample

[`skysurvey_warp_sample.ipynb`](notebooks/template_usage/skysurvey_warp_sample.ipynb)
shows the complete configuration, diagnostic, and production workflow. The
same entry point can be used from Python:

```python
from warptemplate import WarpSampleSpec, WarpSimulationRunner

spec = WarpSampleSpec(
    run_name="ztf_balanced_cc_schema6",
    active_fitclasses=["SN IIP", "SN IIb", "SN Ib", "SN Ic"],
    size=100_000,
    class_sampling="balanced",
    redshift_sampling="binned",
    redshift_bins=[0.0, 0.02, 0.04, 0.06, 0.08],
    zmax=0.08,
    survey_name="ztf",
    survey_options={
        "time_mode": "relative",
        "ztf_filters": ["g", "r"],
        "ztf_clean_only": True,
        "nside": 64,
    },
    batch_size=10_000,
    max_sources_per_batch=512,
    seed=20260709,
)

runner = WarpSimulationRunner(
    "data/warpcoeff_v4",
    source_cache_dir="data/warp_source_cache_v1",
)
manifest = runner.run(spec, "training_samples", resume=True)
```

`survey_name` accepts `ztf`, `lsst`, and `combined`. Combined mode is a
synthetic cadence using the intersection of the selected ZTF and LSST
footprints; it is not a coordinated-observing forecast.

The runner creates source-cache partitions as needed, draws one fit class at a
time, evaluates bounded batches through SkySurvey, and writes separate
`truth/` and `observations/` Parquet trees joined by `object_id`. Truth includes
drawn targets without retained visits. Schema 6 records the sample
configuration, source-cache fingerprint, population settings, and survey
provenance in its manifest.

Resuming a run validates that provenance and skips only batches whose manifest,
truth partition, and observation partition are all complete. Seeds are derived
per object, so changing batch boundaries does not change an object's simulated
realization.

For large runs, `batch_size` bounds target rows while
`max_sources_per_batch` bounds the distinct selected Warp sources. The cache
keeps neutral source grids reusable; colour, redshift, peak time, and amplitude
remain event-local.

### Train a classifier

Use
[`train_parsnip_classifier.ipynb`](notebooks/classification/train_parsnip_classifier.ipynb)
for the exact-redshift ParSNIP baseline or
[`train_supernnova_classifier.ipynb`](notebooks/classification/train_supernnova_classifier.ipynb)
for the parallel recurrent-network workflow. Both use deterministic grouped
splits, disposable smoke runs, and gated full-test evaluation.

The optional classifier dependencies must be installed in the active notebook
kernel. ParSNIP additionally requires the astronomical `astro-parsnip`
distribution shown above; the recurrent workflow uses the local
`WarpSequenceRNN` and does not require an external SuperNNova installation. The
notebooks validate their kernel packages before adding the workspace path for the
local `warptemplate` checkout, so neighbouring source copies cannot silently
replace them. Generated splits, checkpoints, predictions, plots, and run products
are excluded from Git.

## Migrating an older checkout

Install this branch as a new editable checkout, as recommended for the restructured
coefficient library. Imports now consistently use the lowercase package name:

```python
from warptemplate import WarpfitTemplateLoader
```

Root-level imports such as `from loaders import ...` and the old camel-case
`warpTemplate` package name are no longer supported. Existing v3 coefficient files
remain usable through the loader fallback, but new simulations should point at the
v4 directory explicitly so their provenance records the intended data product.

## Tests and benchmarks

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The coefficient files are optional for unit tests. To include all integration tests,
point the suite at a downloaded coefficient library:

```bash
WARPTEMPLATE_TEST_COEFFICIENT_DIR=data/warpcoeff_v4 python -m pytest -q
```

The source-reuse and end-to-end memory checks live in
[`benchmarks/`](benchmarks/):

```bash
python benchmarks/benchmark_source_reuse.py --events 1000000
python benchmarks/benchmark_batch_memory.py
```

Benchmark results depend on the selected cadence, local data, and environment;
they are not runtime guarantees.

## License

MIT License. See [`LICENSE`](LICENSE).

## Acknowledgments

This package builds on
[sncosmo](https://sncosmo.readthedocs.io/) for supernova models and
[SkySurvey](https://github.com/MickaelRigault/skysurvey) for survey
simulation.
