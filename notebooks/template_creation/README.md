# Template Creation Notebooks

This folder contains the historical notebooks and helper files used to create, validate, and colour-calibrate the warp coefficient libraries.

Useful reading order:

1. `v2_0_define_warpclasses.ipynb`: class definitions and taxonomy context.
2. `v2_I_btssncosmo.ipynb`: BTS light-curve fitting and peak information.
3. `v2_II_class2warpset.ipynb`: grouping fitted objects into warp classes.
4. `warp_splines_construct.ipynb`: construction of phase-wavelength correction surfaces.
5. `v2_III_colors.ipynb` and `warptemplate_colors.ipynb`: peak-colour distributions.
6. `warpcoeff_distcolcorr.ipynb` and `warpcoeff_colorcorrect*.ipynb`: colour-correction calibration and checks.

These notebooks are provenance, not the preferred runtime API. New code should generally use `WarpfitTemplateLoader` and the helpers exported by the `warptemplate` package.
