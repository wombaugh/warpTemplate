#!/usr/bin/env python
# coding: utf-8

"""
Load and register OpenUniverse2024 templates into sncosmo for subsequent use.
Presumably, these will need to be added when simulating as well if so wanted.
"""

import os
import pandas as pd
import sncosmo
import matplotlib.pyplot as plt


def load_and_register_templates(bdir, modinfo, threshold=0.01, plot=True):
    """
    Load OpenUniverse SED templates, optionally plot diagnostics,
    and register them with sncosmo.
    """
    for name, tempdata in modinfo.items():
        print(name)
        path = os.path.join(bdir, tempdata['dir'], tempdata['fname'])
        print(path)

        df = pd.read_csv(
            path,
            sep=r'\s+',
            skiprows=2,
            names=['phase', 'wavelength', 'flux']
        )

        # Base restructure
        pivoted = df.pivot(index='phase', columns='wavelength', values='flux')
        print(pivoted.shape)

        # Restrict to significant fluxes
        phase_max = pivoted.abs().max(axis=1)
        wave_max = pivoted.abs().max(axis=0)

        phase_mask = phase_max > phase_max.max() * threshold
        wave_mask = wave_max > wave_max.max() * threshold
        pivoted_cut = pivoted.loc[phase_mask, wave_mask]
        print(pivoted_cut.shape)

        # Final template arrays
        phase = pivoted_cut.index.values
        wave = pivoted_cut.columns.values
        flux = pivoted_cut.values

        if plot:
            # Plot base template cut diagnostics
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            axes[0].plot(pivoted.index, phase_max, '.', ms=1)
            axes[0].set_xlabel('phase')
            axes[0].set_ylabel('max |flux|')
            axes[0].axhline(y=phase_max.max() * threshold, color='r', ls='--')

            axes[1].plot(pivoted.columns, wave_max, '.', ms=1)
            axes[1].set_xlabel('wavelength')
            axes[1].set_ylabel('max |flux|')
            axes[1].axhline(y=wave_max.max() * threshold, color='r', ls='--')
            plt.tight_layout()
            plt.show()

        # Create and register sncosmo source
        source = sncosmo.TimeSeriesSource(phase, wave, flux, name=name, version='1.0')

        if plot:
            # Plot phase evolution in ZTF bands
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            axes[0].plot(pivoted_cut.index, source.bandflux('ztfg', pivoted_cut.index))
            axes[0].set_xlabel('phase')
            axes[0].set_ylabel('ztfg flux')

            axes[1].plot(pivoted_cut.index, source.bandflux('ztfi', pivoted_cut.index))
            axes[1].set_xlabel('phase')
            axes[1].set_ylabel('ztfi flux')
            plt.tight_layout()
            plt.show()

            # Plot spectra at selected phases
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            axes[0].plot(pivoted_cut.columns, source.flux(-5, pivoted_cut.columns))
            axes[0].set_xlabel('wavelength')
            axes[0].set_ylabel('flux @ phase=-5')

            axes[1].plot(pivoted_cut.columns, source.flux(10, pivoted_cut.columns))
            axes[1].set_xlabel('wavelength')
            axes[1].set_ylabel('flux @ phase=10')
            plt.tight_layout()
            plt.show()

        sncosmo.registry.register(source, name=name, force=True)
        print(f"Registered: {name}\n")

    return


def main():
    # Configuration
    threshold = 0.01  # Extract phases/wavelengths where flux < threshold * peak

    # Base directory for extracted data
    bdir = '/Users/jnordin/data/openUniverse24/MODELS-1_TRANSIENT_SED'

    # Model definitions
    modinfo = {
        # TDE AT2019qiz: Hung et al. 2021, Nicholl et al. 2020
        # arXiv:2501.05632; Kessler 2025 (Zenodo, CC-BY-4.0)
        'tde-at2019qiz-ou': {
            'dir': 'NON1ASED.TDE-BBFIT',
            'fname': '2019qiz.sed.gz',
        },
        # SLSN-I 2016apd: Yan et al. 2017; Kangas et al. 2017; Guillochon et al. 2017
        # Grey BB fit by K. Das (Caltech)
        'slsn-i2016apd-ou': {
            'dir': 'NON1ASED.SLSN-I-BBFIT',
            'fname': '2016apd.sed.gz',
        },
        # Note: SNIax and PISN templates, POSSIS kilonova models available
        # but omitted here; can be added for training if needed.
    }

    load_and_register_templates(bdir, modinfo, threshold=threshold, plot=True)



if __name__ == '__main__':
    main()