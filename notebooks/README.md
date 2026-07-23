# Notebook Layout

This directory separates the exploratory notebooks by intent.

- `template_creation/`: historical and scientific provenance for how warp coefficient files were built, validated, and colour-calibrated.
- `template_usage/`: current entry points for loading existing coefficient files, visualising templates, and producing light-curve tables.
- `classification/`: grouped-split classifier training, evaluation, and backend handoff notebooks.

For normal template work, start in `template_usage/`. For the first classifier
baseline, start with `classification/train_parsnip_classifier.ipynb`. Use
`template_creation/` when you need to understand or rebuild the coefficient
libraries.
