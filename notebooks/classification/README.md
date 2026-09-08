# Classifier notebooks

Use the configured `skysurvey_env` kernel and install WarpTemplate in editable
mode.  Both classifier notebooks import their required packages directly and use
one external data directory:

```text
parent/
├── warpTemplate/                  Git repository and notebooks
└── data/
    ├── training_samples/
    ├── classification_splits/
    └── classifier_runs/
```

Every notebook anchors `DATA_ROOT` to the location of the editable `warptemplate`
package.  This is independent of the kernel working directory and contains no
directory search or fallback.  A missing sample, artifact, or package therefore
raises the normal error at its configured location.

## Explicit stage actions

`train_parsnip_classifier.ipynb` is the exact-redshift ParSNIP baseline;
`train_supernnova_classifier.ipynb` is the recurrent comparison.  Each begins with
one user-settings block containing a readable `RUN_NAME` and an action for every
expensive stage:

- `run`: create a new artifact and refuse an existing target;
- `load`: read the fixed artifact path directly;
- `skip`: do not execute that stage;
- `resume`: exactly continue an incomplete SuperNNova training checkpoint.

ParSNIP deliberately has no `resume` action.  Its native checkpoint contains model
weights and settings, but not optimizer, scheduler, epoch, or RNG state.  Reloading it
for further fitting would be a warm start, not a reproducible continuation.

Run products use predictable paths:

```text
../data/classifier_runs/<sample>/<split>/<backend>/<run_name>/
```

The stored experiment metadata still records a configuration hash.  Loading a
readable run name with changed scientific settings is rejected.

## Scientific safeguards

Smoke mode uses disposable training-side folds and never opens official fold 0.
Full evaluation remains a separate `TEST_ACTION`.  Test predictions, metrics, and
figures use write-once helpers and are never overwritten; choose a new `RUN_NAME` for
a new experiment.

Leakage-safe groups, train-only normalization, checkpoint/database identity,
probability validation, and grouped uncertainty estimates remain enforced in the
Python backends.  Only notebook-level path discovery and multi-file cache state
machines were removed.
