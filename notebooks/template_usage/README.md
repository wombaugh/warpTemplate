# Template Usage And Visualisation

This folder contains the current practical notebooks for using already-built warp coefficient files.

Start here:

1. `create_usable_template_lightcurves.ipynb`
2. `skysurvey_warp_sample.ipynb`

The first notebook loads `warpcoeff_v3`, samples templates with
`WarpfitTemplateLoader`, compares colour modes, measures peak colour, and can
turn one warped `sncosmo.Model` into a light-curve table.

The second notebook configures and diagnoses batched ZTF, LSST, or combined
SkySurvey samples, builds the required source cache, and writes resumable
schema-6 truth and observation datasets.
