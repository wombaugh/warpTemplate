#!/usr/bin/env python
# coding: utf-8

"""
Module: openuniverse_registry.py

Lädt OpenUniverse2024-Templates und registriert sie in sncosmo.
Import des Moduls genügt zur Registrierung; alternativ `register_all()`
explizit aufrufen.
"""

import os
import pickle
import hashlib
import warnings

import numpy as np
import pandas as pd
import sncosmo


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

BASE_DIR = os.environ.get(
    "WARPTEMPLATE_OPENUNIVERSE_DIR",
    "/Users/jnordin/data/openUniverse24/MODELS-1_TRANSIENT_SED",
)
THRESHOLD = 0.01

# Cache-Verzeichnis für serialisierte Source-Objekte
_CACHE_DIR = os.environ.get(
    "WARPTEMPLATE_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "sncosmo_openuniverse"),
)

MODEL_REGISTRY = {
    'tde-at2019qiz-ou': {
        'dir': 'NON1ASED.TDE-BBFIT',
        'fname': '2019qiz.sed.gz',
        'meta': {
            'type': 'TDE',
            'ref': 'Hung et al. 2021; Nicholl et al. 2020; arXiv:2501.05632',
        }
    },
    'slsn-i2016apd-ou': {
        'dir': 'NON1ASED.SLSN-I-BBFIT',
        'fname': '2016apd.sed.gz',
        'meta': {
            'type': 'SLSN-I',
            'ref': 'Yan et al. 2017; Kangas et al. 2017; Guillochon et al. 2017',
        }
    },
}


# -----------------------------------------------------------------------------
# Kernfunktionalität
# -----------------------------------------------------------------------------

def _compute_file_hash(path):
    """MD5-Hash für Cache-Invalidierung."""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def _load_sed_data(path, threshold=THRESHOLD):
    """
    Lädt SED-Datei, pivotiert und schneidet auf signifikante Flux-Werte.
    Gibt (phase, wave, flux, metadata_dict) zurück.
    """
    df = pd.read_csv(
        path,
        sep=r'\s+',
        skiprows=2,
        names=['phase', 'wavelength', 'flux']
    )

    pivoted = df.pivot(index='phase', columns='wavelength', values='flux')

    phase_max = pivoted.abs().max(axis=1)
    wave_max = pivoted.abs().max(axis=0)

    phase_mask = phase_max > phase_max.max() * threshold
    wave_mask = wave_max > wave_max.max() * threshold
    pivoted_cut = pivoted.loc[phase_mask, wave_mask]

    meta = {
        'original_shape': pivoted.shape,
        'cut_shape': pivoted_cut.shape,
        'phase_range': (float(pivoted_cut.index.min()), float(pivoted_cut.index.max())),
        'wave_range': (float(pivoted_cut.columns.min()), float(pivoted_cut.columns.max())),
    }

    return (
        pivoted_cut.index.values,
        pivoted_cut.columns.values,
        pivoted_cut.values,
        meta
    )


def _build_source(name, path, threshold=THRESHOLD, use_cache=True):
    """
    Erzeugt TimeSeriesSource, mit optionalem Pickle-Cache.
    """
    file_hash = _compute_file_hash(path)
    cache_path = os.path.join(_CACHE_DIR, f"{name}_{file_hash}.pkl")

    # Cache creation is deliberately lazy so importing ``warptemplate`` never
    # writes to the user's home directory. An unwritable cache is non-fatal.
    if use_cache:
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
        except OSError as error:
            warnings.warn(f"OpenUniverse cache disabled: {error}")
            use_cache = False

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            source = pickle.load(f)
        # Name kann bei Deserialisierung verloren gehen
        if not hasattr(source, '_name') or source._name != name:
            source._name = name
        return source

    phase, wave, flux, meta = _load_sed_data(path, threshold)
    source = sncosmo.TimeSeriesSource(phase, wave, flux, name=name, version='1.0')
    source._ou_meta = meta  # Anreicherung für Debugging

    if use_cache:
        with open(cache_path, 'wb') as f:
            pickle.dump(source, f)

    return source


def register_model(name, model_config, base_dir=BASE_DIR, force=True, use_cache=True):
    """
    Einzelnes Modell laden und in sncosmo registrieren.
    """
    path = os.path.join(base_dir, model_config['dir'], model_config['fname'])

    if not os.path.exists(path):
        warnings.warn(f"Datei nicht gefunden: {path}")
        return False

    source = _build_source(name, path, use_cache=use_cache)

    # Prüfen ob bereits registriert
    try:
        sncosmo.registry.retrieve(sncosmo.Source, name)
        already_registered = True
    except Exception:
        already_registered = False

    if already_registered and not force:
        return True

    sncosmo.registry.register(source, name=name, force=force)
    return True


def register_all(base_dir=BASE_DIR, force=True, use_cache=True):
    """
    Alle konfigurierten Modelle registrieren.
    """
    success = {}
    for name, config in MODEL_REGISTRY.items():
        success[name] = register_model(
            name, config, base_dir=base_dir, force=force, use_cache=use_cache
        )
    return success


def get_registered_names():
    """Liste aller aktuell registrierten OU-Modelle."""
    # sncosmo hat keine direkte Auflistung; wir filtern über retrieve-Versuche
    registered = []
    for name in MODEL_REGISTRY.keys():
        try:
            sncosmo.registry.retrieve(sncosmo.Source, name)
            registered.append(name)
        except Exception:
            pass
    return registered


def plot_diagnostics(name, base_dir=BASE_DIR):
    """Diagnostische Plots für ein Modell (vor Registrierung)."""
    import matplotlib.pyplot as plt

    config = MODEL_REGISTRY.get(name)
    if not config:
        raise ValueError(f"Unbekanntes Modell: {name}")

    path = os.path.join(base_dir, config['dir'], config['fname'])
    phase, wave, flux, meta = _load_sed_data(path)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Phase coverage
    phase_max = pd.Series(np.max(np.abs(flux), axis=1), index=phase)
    axes[0, 0].plot(phase, phase_max, '.', ms=1)
    axes[0, 0].axhline(y=phase_max.max() * THRESHOLD, color='r', ls='--')
    axes[0, 0].set_xlabel('phase')
    axes[0, 0].set_ylabel('max |flux|')

    # Wavelength coverage
    wave_max = pd.Series(np.max(np.abs(flux), axis=0), index=wave)
    axes[0, 1].plot(wave, wave_max, '.', ms=1)
    axes[0, 1].axhline(y=wave_max.max() * THRESHOLD, color='r', ls='--')
    axes[0, 1].set_xlabel('wavelength')
    axes[0, 1].set_ylabel('max |flux|')

    # ZTF light curves
    source = sncosmo.TimeSeriesSource(phase, wave, flux, name=name)
    axes[1, 0].plot(phase, source.bandflux('ztfg', phase))
    axes[1, 0].set_xlabel('phase')
    axes[1, 0].set_ylabel('ztfg flux')

    axes[1, 1].plot(phase, source.bandflux('ztfi', phase))
    axes[1, 1].set_xlabel('phase')
    axes[1, 1].set_ylabel('ztfi flux')

    plt.tight_layout()
    plt.suptitle(f"{name}: {meta['cut_shape']}", y=1.02)
    return fig


# -----------------------------------------------------------------------------
# Auto-Registrierung bei Import (optional, kommentierbar)
# -----------------------------------------------------------------------------

# Entkommentieren für automatische Registrierung:
# register_all()


# -----------------------------------------------------------------------------
# CLI / Standalone-Ausführung
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='OpenUniverse sncosmo template registry')
    parser.add_argument('--register', action='store_true', help='Register all models')
    parser.add_argument('--list', action='store_true', help='List registered models')
    parser.add_argument('--plot', type=str, help='Plot diagnostics for MODEL')
    parser.add_argument('--no-cache', action='store_true', help='Disable pickle cache')

    args = parser.parse_args()

    if args.plot:
        import matplotlib.pyplot as plt

        fig = plot_diagnostics(args.plot)
        plt.show()

    if args.register or not (args.list or args.plot):
        use_cache = not args.no_cache
        success = register_all(use_cache=use_cache)
        print("Registration status:")
        for name, ok in success.items():
            status = "✓" if ok else "✗"
            print(f"  {status} {name}")

    if args.list or args.register or not args.plot:
        registered = get_registered_names()
        print(f"\nCurrently registered: {registered}")
