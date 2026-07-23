# Classifier notebooks

Start with `train_parsnip_classifier.ipynb`. It is the executable exact-redshift
ParSNIP baseline and contains explicit switches for the smoke run, full training,
representation building, classifier fitting, and the one-time test evaluation.

`train_supernnova_classifier.ipynb` is the parallel recurrent-network workflow.
Its default `EXECUTION_MODE = "smoke"` trains both `photometry_only` and
`photometry_plus_truth_z` for two epochs on CPU and displays preliminary
validation and disposable smoke-test confusion matrices. Change the single mode
to `full_cpu` for one full seed or `full_gpu_repeats` for seeds 20260721 through
20260725. The GPU mode refuses to run when CUDA is unavailable instead of
silently falling back to CPU.

The notebooks expect this workspace layout for WarpTemplate and generated data:

```text
warp_templates/
├── training_samples/              generated Parquet samples
└── warpTemplate/                  this package and notebooks
```

Use a kernel that provides the astronomical
[LSSTDESC ParSNIP](https://github.com/LSSTDESC/parsnip), SuperNNova, PyTorch,
LightGBM, `lcdata`, and the optional classifier dependencies. The notebooks
intentionally import ParSNIP and SuperNNova from that kernel before adding the
workspace path for the local WarpTemplate checkout. Reference source folders
beside this repository are therefore never runtime dependencies. The unrelated
PyPI parser that also uses the name `parsnip` does not provide the required
`ParsnipModel` and `Classifier` APIs and is rejected with an actionable error.

Persistent group manifests are written below `classification_splits/<sample>/`.
Run products are written below `classifier_runs/<evaluation_sample>/<strategy>/...`
and are excluded from Git. The observation Parquet tree is never duplicated.

Completed smoke and full runs are caches as well as result folders. Completion-marker
files are written only after all required artifacts exist. Rerunning an enabled cell
loads those artifacts instead of training again. Set the relevant switch only when a
deliberate recomputation is needed:

- `FORCE_RETRAIN_SMOKE`
- `FORCE_RETRAIN_FULL`
- `FORCE_REBUILD_REPRESENTATIONS`
- `FORCE_REFIT_CLASSIFIER`

Changing the scientific configuration creates a different hashed run directory, so
incompatible cached models are not silently reused.

The SuperNNova notebook follows the same cache and release rules. A shared,
configuration-hashed HDF5 database stores grouped nine-band sequences for both
redshift variants. Each run then writes resumable and best checkpoints, history,
predictions, metrics, figures, and a completion marker below its standard run
directory. During a new or resumed run, the training cell reports the active
variant, epoch, batch percentage, current weighted loss, validation loss, and
learning rate. A completed cached run instead prints its cache location and returns
immediately. The main controls are:

- `EXECUTION_MODE`: `smoke`, `full_cpu`, or `full_gpu_repeats`.
- `FORCE_REBUILD_DATABASE`: deliberately replace the compatible sequence cache.
- `FORCE_RETRAIN`: deliberately replace model and smoke prediction products.
- `EVALUATE_TEST`: open the official fold-0 test only for a full run.
- `ALLOW_TEST_OVERWRITE`: explicitly replace already frozen full-test artifacts.

Smoke mode mirrors the ParSNIP disposable folds: folds 4–9 train, fold 3
validates, and fold 2 supplies preliminary smoke-test matrices. Official folds 0
and 1 are not used. Full modes restore folds 2–9 for training, fold 1 for
validation, and retain fold 0 behind the evaluation gate. Full evaluation also
supports peak-relative partial sequences, observing-condition breakdowns,
calibration, grouped bootstrap intervals, and paired comparison of the two
redshift variants.
