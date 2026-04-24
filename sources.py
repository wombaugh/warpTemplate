# warpTemplate/sources.py
import numpy as np
import sncosmo
from scipy.interpolate import RectBivariateSpline as Spline2d

from sncosmo import TimeSeriesSource
import extinction

import numpy.typing as npt
from typing import List, Dict, Union, Optional



class WarpedTimeSeriesSource(TimeSeriesSource):
    """
    A TimeSeriesSource that applies a multiplicative warp to an existing
    template source.

    The warped flux is computed as:
        warped_flux = original_flux * warp(phase, wavelength)

    Parameters
    ----------
    phase : ndarray of shape (N,)
        Phase grid for the warp function.
    wave : ndarray of shape (M,)
        Wavelength grid for the warp function.
    flux : ndarray of shape (N, M)
        Warp values defined on (phase, wave). This acts as a multiplicative
        modifier to the original flux.
    original_template_name : str
        Name of the base template in sncosmo.
    original_template_version : str, optional
        Version of the base template.
    cut_negative_flux : bool, optional (default=True)
        If True, negative flux values after warping are set to zero.
    time_spline_degree : int, optional (default=3)
        Degree of the spline interpolation in the phase dimension.
    warp_reddening_ebv: float, optional
        If provided, applies a reddening correction to the warp function using the specified E(B-V)
    warp_reddening_rv: float (default=3.1)
        R_V value to use for the reddening correction if `warp_reddening_ebv` is provided.
    name : str, optional
        Name of this warped source.
    version : str, optional
        Version identifier for this warped source.
    """

    def __init__(
        self,
        phase: npt.ArrayLike,
        wave: npt.ArrayLike,
        flux: npt.ArrayLike,
        original_template_name: str,
        original_template_version: Optional[str] = None,
        cut_negative_flux: bool = True,
        time_spline_degree: int = 3,
        warp_reddening_ebv: Optional[float] = None,
        warp_reddening_rv: float = 3.1,
        name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        # Convert inputs to numpy arrays
        phase_arr: npt.NDArray[np.float64] = np.asarray(phase, dtype=float)
        wave_arr: npt.NDArray[np.float64] = np.asarray(wave, dtype=float)
        flux_arr: npt.NDArray[np.float64] = np.asarray(flux, dtype=float)

        # Metadata
        self.name: Optional[str] = name
        self.version: Optional[str] = version

        # Required parameter interface
        self._parameters: npt.NDArray[np.float64] = np.array([1.0])

        # Load original source
        self._original_source: TimeSeriesSource = sncosmo.get_source(
            original_template_name,
            original_template_version
        )

        # Create warp spline
        self._warp_spline: Spline2d = Spline2d(
            phase_arr,
            wave_arr,
            flux_arr,
            kx=time_spline_degree,
            ky=3
        )

        # Behavior flag
        self._zero_before: bool = True

        # Restrict phase grid to overlap region
        original_phase: npt.NDArray[np.float64] = self._original_source._phase
        self._phase: npt.NDArray[np.float64] = original_phase[
            (original_phase >= phase_arr.min()) &
            (original_phase <= phase_arr.max())
        ]

        # Restrict wavelength grid to overlap region
        original_wave: npt.NDArray[np.float64] = self._original_source._wave
        self._wave: npt.NDArray[np.float64] = original_wave[
            (original_wave >= wave_arr.min()) &
            (original_wave <= wave_arr.max())
        ]

        # Compute warped flux
        original_flux: npt.NDArray[np.float64] = self._original_source._flux(
            self._phase,
            self._wave
        )
        warp_factor: npt.NDArray[np.float64] = self._warp_spline(
            self._phase,
            self._wave
        )

        warped_flux: npt.NDArray[np.float64] = original_flux * warp_factor

        # Apply optional reddening correction to the warp factor
        if warp_reddening_ebv is not None:

            warped_flux = extinction.apply(
                extinction.ccm89(self._wave, warp_reddening_ebv * warp_reddening_rv, warp_reddening_rv), 
                warped_flux
                )

        # Clip negative values if requested
        if cut_negative_flux:
            warped_flux[warped_flux < 0] = 0.0

        # Final spline model
        self._model_flux: Spline2d = Spline2d(
            self._phase,
            self._wave,
            warped_flux,
            kx=time_spline_degree,
            ky=3
        )
        