"""Spectral time-series sources used by WarpTemplate."""

from __future__ import annotations

from typing import Optional

import extinction
import numpy as np
import numpy.typing as npt
import sncosmo
from scipy.interpolate import RectBivariateSpline as Spline2d
from sncosmo import TimeSeriesSource


DEFAULT_WAVE_COVERAGE = (2000.0, 30000.0)


def build_uncolored_warp_grid(
    phase: npt.ArrayLike,
    wave: npt.ArrayLike,
    flux: npt.ArrayLike,
    original_template_name: str,
    original_template_version: Optional[str] = None,
    *,
    cut_negative_flux: bool = True,
    time_spline_degree: int = 3,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Evaluate one base-template/warp combination without colour correction."""

    phase_arr = np.asarray(phase, dtype=float)
    wave_arr = np.asarray(wave, dtype=float)
    flux_arr = np.asarray(flux, dtype=float)
    if flux_arr.shape != (phase_arr.size, wave_arr.size):
        raise ValueError(
            "inconsistent warp grid shapes: "
            f"flux.shape={flux_arr.shape}, expected "
            f"({phase_arr.size}, {wave_arr.size})"
        )

    try:
        original_source = sncosmo.get_source(
            original_template_name, original_template_version
        )
    except Exception:
        # v4 introduces two external OpenUniverse bases. Register only a known
        # missing source on demand so ordinary imports remain side-effect free.
        from .openuniverse_registry import ensure_registered

        if not ensure_registered(original_template_name):
            raise
        original_source = sncosmo.get_source(
            original_template_name, original_template_version
        )
    warp_spline = Spline2d(
        phase_arr,
        wave_arr,
        flux_arr,
        kx=time_spline_degree,
        ky=3,
    )

    # The usable source is the exact phase/wavelength overlap of both inputs.
    original_phase = np.asarray(original_source._phase, dtype=float)
    original_wave = np.asarray(original_source._wave, dtype=float)
    output_phase = original_phase[
        (original_phase >= phase_arr.min()) & (original_phase <= phase_arr.max())
    ]
    output_wave = original_wave[
        (original_wave >= wave_arr.min()) & (original_wave <= wave_arr.max())
    ]
    if output_phase.size < time_spline_degree + 1 or output_wave.size < 4:
        raise ValueError(
            "base template and warp grid do not have enough overlapping samples"
        )

    original_flux = original_source._flux(output_phase, output_wave)
    warped_flux = np.asarray(
        original_flux * warp_spline(output_phase, output_wave), dtype=float
    )
    if cut_negative_flux:
        warped_flux[warped_flux < 0.0] = 0.0
    return output_phase, output_wave, warped_flux


class DynamicColorWarpSource(TimeSeriesSource):
    """Warped source whose CCM89 colour correction is an event parameter."""

    _param_names = ["amplitude", "samplecorr_ebv"]
    param_names_latex = ["A", "E(B-V)_{sample}"]

    def __init__(
        self,
        phase: npt.ArrayLike,
        wave: npt.ArrayLike,
        flux: npt.ArrayLike,
        *,
        samplecorr_rv: float = 3.1,
        zero_before: bool = True,
        time_spline_degree: int = 3,
        wave_coverage: Optional[tuple[float, float]] = DEFAULT_WAVE_COVERAGE,
        name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """Build a reusable spline with conservative zero-flux edge coverage."""

        phase_arr = np.asarray(phase, dtype=float)
        wave_arr = np.asarray(wave, dtype=float)
        flux_arr = np.asarray(flux, dtype=float)
        if flux_arr.shape != (phase_arr.size, wave_arr.size):
            raise ValueError(
                "inconsistent source grid shapes: "
                f"flux.shape={flux_arr.shape}, expected "
                f"({phase_arr.size}, {wave_arr.size})"
            )
        if not np.isfinite(samplecorr_rv) or samplecorr_rv <= 0:
            raise ValueError("samplecorr_rv must be finite and positive")

        # Keep interpolation strictly inside the measured native grid.  The
        # broader advertised range lets redshifted edge bands be integrated,
        # while _flux returns zero rather than extrapolating unknown spectra.
        native_wave_min = float(wave_arr[0])
        native_wave_max = float(wave_arr[-1])
        if wave_coverage is None:
            coverage_min, coverage_max = native_wave_min, native_wave_max
        else:
            requested_min, requested_max = map(float, wave_coverage)
            if not np.isfinite([requested_min, requested_max]).all():
                raise ValueError("wave_coverage bounds must be finite")
            if requested_min >= requested_max:
                raise ValueError("wave_coverage bounds must be strictly increasing")
            coverage_min = min(requested_min, native_wave_min)
            coverage_max = max(requested_max, native_wave_max)

        self.name = name
        self.version = version
        self._phase = phase_arr
        self._native_wave = wave_arr
        coverage_wave = wave_arr
        if coverage_min < native_wave_min:
            coverage_wave = np.concatenate(([coverage_min], coverage_wave))
        if coverage_max > native_wave_max:
            coverage_wave = np.concatenate((coverage_wave, [coverage_max]))
        self._wave = coverage_wave
        self._parameters = np.array([1.0, 0.0])
        self._model_flux = Spline2d(
            phase_arr,
            wave_arr,
            flux_arr,
            kx=time_spline_degree,
            ky=3,
        )
        self._zero_before = bool(zero_before)
        self._samplecorr_rv = float(samplecorr_rv)

    @classmethod
    def from_warp_grid(
        cls,
        phase: npt.ArrayLike,
        wave: npt.ArrayLike,
        flux: npt.ArrayLike,
        original_template_name: str,
        original_template_version: Optional[str] = None,
        **kwargs: object,
    ) -> "DynamicColorWarpSource":
        """Create a dynamic-colour source from a compact multiplicative warp."""

        output_phase, output_wave, output_flux = build_uncolored_warp_grid(
            phase,
            wave,
            flux,
            original_template_name,
            original_template_version,
        )
        return cls(output_phase, output_wave, output_flux, **kwargs)

    def _flux(
        self, phase: npt.ArrayLike, wave: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Evaluate native wavelengths and return zero outside their support."""

        phase_arr = np.atleast_1d(np.asarray(phase, dtype=float))
        wave_arr = np.atleast_1d(np.asarray(wave, dtype=float))
        evaluated = np.zeros((phase_arr.size, wave_arr.size), dtype=float)
        native_mask = (
            (wave_arr >= self._native_wave[0])
            & (wave_arr <= self._native_wave[-1])
        )
        if np.any(native_mask):
            native_wave = wave_arr[native_mask]
            native_flux = self._parameters[0] * self._model_flux(
                phase_arr, native_wave
            )
            samplecorr_ebv = float(self._parameters[1])
            if samplecorr_ebv != 0.0:
                attenuation = extinction.ccm89(
                    native_wave,
                    samplecorr_ebv * self._samplecorr_rv,
                    self._samplecorr_rv,
                )
                native_flux = extinction.apply(attenuation, native_flux)
            evaluated[:, native_mask] = native_flux
        if self._zero_before:
            evaluated[phase_arr < self.minphase(), :] = 0.0
        return np.asarray(evaluated, dtype=float)

    @property
    def native_wave(self) -> npt.NDArray[np.float64]:
        """Return the wavelength samples on which the spline is defined."""

        return self._native_wave

    def native_flux_grid(self) -> npt.NDArray[np.float64]:
        """Return the colour-neutral flux grid on native wavelength support."""

        return np.asarray(
            self._model_flux(self._phase, self._native_wave),
            dtype=float,
        )


class WarpedTimeSeriesSource(DynamicColorWarpSource):
    """Backward-compatible constructor for a dynamically coloured Warp source."""

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
        """Prepare the uncoloured grid and expose the historical API."""

        output_phase, output_wave, output_flux = build_uncolored_warp_grid(
            phase,
            wave,
            flux,
            original_template_name,
            original_template_version,
            cut_negative_flux=cut_negative_flux,
            time_spline_degree=time_spline_degree,
        )
        super().__init__(
            output_phase,
            output_wave,
            output_flux,
            samplecorr_rv=warp_reddening_rv,
            zero_before=True,
            time_spline_degree=time_spline_degree,
            name=name,
            version=version,
        )
        self._parameters[1] = (
            0.0 if warp_reddening_ebv is None else float(warp_reddening_ebv)
        )


__all__ = [
    "DEFAULT_WAVE_COVERAGE",
    "DynamicColorWarpSource",
    "WarpedTimeSeriesSource",
    "build_uncolored_warp_grid",
]
