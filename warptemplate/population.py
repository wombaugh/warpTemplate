"""Population configuration and scientific priors for WarpTemplate samples.

This module keeps the Warp-specific decisions explicit:

* which fitclasses are active,
* which literature rate is attached to each fitclass,
* which fitclasses overlap taxonomically,
* how absolute-magnitude priors and rate provenance are resolved.

The code intentionally imports SkySurvey and sncosmo only inside methods that
actually need them. This keeps rate validation and config tests usable in
minimal environments.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


RATE_CONFIG_FILENAME = "warp_rates_literature.json"

RATE_STATUSES = {
    "adopted",
    "derived",
    "aggregate",
    "missing_direct_rate",
}

RATE_KINDS = {
    "volumetric",
    "fraction_of_parent",
    "derived_from_children",
}

MAGABS_PRIOR_STATUSES = {
    "adopted",
    "proxy",
    "aggregate",
    "missing_direct_prior",
    "user_override",
}

MAGABS_DISTRIBUTIONS = {
    "normal",
    "split_normal",
    "uniform",
    "fixed",
    "children_mixture",
    "missing_direct_prior",
}


DEFAULT_ACTIVE_FITCLASSES_BROAD = [
    "SN Ia (w)",
    "SN IIP",
    "SN IIb",
    "SN IIn",
    "SN Ib",
    "SN Ic",
    "SN Ic-BL",
    "SLSN-I",
    "SLSN-II",
]

DEFAULT_ACTIVE_FITCLASSES_IA_SUBTYPE = [
    "SN Ia-91bg",
    "SN Ia-91T",
    "SN Ia-CSM",
    "SN Ia-SC",
    "SN Iax",
]

DEFAULT_OVERLAP_CHILDREN = {
    "SN Ia (a)": [
        "SN Ia (w)",
        "SN Ia-91bg",
        "SN Ia-91T",
        "SN Ia-CSM",
        "SN Ia-SC",
        "SN Ia-pec",
        "SN Ia-pec (w)",
        "SN Iax",
    ],
    "SN Ia (w)": [
        "SN Ia-91bg",
        "SN Ia-91T",
        "SN Ia-CSM",
        "SN Ia-SC",
        "SN Ia-pec",
        "SN Ia-pec (w)",
        "SN Iax",
    ],
    "SN Ia-pec (w)": ["SN Ia-pec", "SN Ia-CSM", "SN Ia-SC", "SN Iax"],
    "SN Ia-pec": ["SN Ia-CSM", "SN Ia-SC", "SN Iax"],
    "SN CC (a)": [
        "SN II",
        "SN II (w)",
        "SN IIP",
        "SN IIb",
        "SN IIn",
        "SN Ib",
        "SN Ibn",
        "SN Ic",
        "SN Ic-BL",
        "SN Ibc",
        "SN Ibc (e)",
        "SN Ibc (w)",
        "SLSN (e)",
        "SLSN (w)",
        "SLSN-I",
        "SLSN-II",
    ],
    "SN II (w)": ["SN II", "SN IIP", "SN IIb", "SN IIn"],
    "SN II": ["SN IIP", "SN IIb", "SN IIn"],
    "SN Ibc (w)": ["SN Ibc", "SN Ibc (e)", "SN IIb", "SN Ib", "SN Ibn", "SN Ic", "SN Ic-BL"],
    "SN Ibc (e)": ["SN Ibc", "SN IIb", "SN Ib", "SN Ibn", "SN Ic", "SN Ic-BL"],
    "SN Ibc": ["SN Ib", "SN Ibn", "SN Ic", "SN Ic-BL"],
    "SLSN (w)": ["SLSN (e)", "SLSN-I", "SLSN-II", "SN IIn"],
    "SLSN (e)": ["SLSN-I", "SLSN-II"],
}


class RateConfigError(ValueError):
    """Base error raised for invalid Warp rate configuration."""


class MissingRateError(RateConfigError):
    """Raised when an active fitclass has no usable rate."""


class OverlapRateError(RateConfigError):
    """Raised when active fitclasses double-count the same population."""


class MissingMagnitudePriorError(RateConfigError):
    """Raised when no usable absolute-magnitude prior is available."""


@dataclass(frozen=True)
class ResolvedRate:
    """Resolved rate and provenance for one fitclass."""

    fitclass: str
    rate_gpc3_yr: Optional[float]
    status: str
    rate_kind: str
    parent: Optional[str]
    fraction_of_parent: Optional[float]
    sources: tuple[Mapping[str, Any], ...]
    notes: str
    derived_from: tuple[str, ...]

    @property
    def source_bibkeys(self) -> tuple[str, ...]:
        """Return non-empty BibTeX keys from the configured rate sources."""

        return tuple(
            source.get("bibkey", "")
            for source in self.sources
            if source.get("bibkey")
        )


def default_rate_config_path() -> Path:
    """Return the package-local literature rate JSON path."""

    return Path(__file__).resolve().parent / "config" / RATE_CONFIG_FILENAME


def load_warp_rate_config(path: Optional[str | Path | Mapping[str, Any]] = None) -> dict[str, Any]:
    """Load a Warp rate configuration.

    Parameters
    ----------
    path:
        None for the package default, a path to a JSON file, or an already
        loaded mapping.
    """

    if path is None:
        path = default_rate_config_path()

    if isinstance(path, Mapping):
        return deepcopy(dict(path))

    with open(Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_rate_entries(rate_config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """Return the fitclass-to-entry mapping from a rate config."""

    if "fitclasses" in rate_config:
        entries = rate_config["fitclasses"]
    else:
        entries = rate_config

    if not isinstance(entries, Mapping):
        raise RateConfigError("rate_config must contain a mapping of fitclasses")

    return entries


def get_magabs_priors(rate_config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """Return the fitclass-to-absolute-magnitude-prior mapping."""

    priors = rate_config.get("magabs_priors", {})
    if not isinstance(priors, Mapping):
        raise RateConfigError("rate_config['magabs_priors'] must be a mapping")

    return priors


def discover_warp_fitclasses(warpcoeffs_dir: str | Path) -> list[str]:
    """Discover unique fitclasses from versioned Warp coefficient pickles."""

    directory = Path(warpcoeffs_dir)
    pattern = re.compile(r"^warpcoeffs_v[^_]+_(?P<fitclass>.+?)(?:_col)?\.pkl$")
    fitclasses = {
        match.group("fitclass")
        for path in directory.glob("warpcoeffs_v*_*.pkl")
        if (match := pattern.match(path.name)) is not None
    }
    return sorted(fitclasses)


def validate_magabs_config(
    rate_config: Mapping[str, Any],
    available_fitclasses: Optional[Iterable[str]] = None,
) -> None:
    """Validate absolute-magnitude prior structure and optional coverage."""

    priors = get_magabs_priors(rate_config)
    if available_fitclasses is not None:
        available = set(available_fitclasses)
        configured = set(priors)
        missing_entries = sorted(available - configured)
        if missing_entries:
            raise RateConfigError(
                "Magnitude-prior config has no entry for fitclasses: "
                + ", ".join(missing_entries)
            )

    required = {
        "status",
        "distribution",
        "sncosmo_band",
        "magsys",
        "source_band",
        "sources",
        "notes",
    }
    for fitclass, prior in priors.items():
        missing = required - set(prior)
        if missing:
            raise RateConfigError(
                f"{fitclass} is missing required magabs_prior keys: {sorted(missing)}"
            )

        status = prior["status"]
        if status not in MAGABS_PRIOR_STATUSES:
            raise RateConfigError(f"{fitclass} has invalid magabs prior status {status!r}")

        distribution = prior["distribution"]
        if distribution not in MAGABS_DISTRIBUTIONS:
            raise RateConfigError(
                f"{fitclass} has invalid magabs distribution {distribution!r}"
            )

        if not isinstance(prior.get("sources"), list):
            raise RateConfigError(f"{fitclass} magabs sources must be a list")

        if distribution == "normal":
            if prior.get("loc") is None or prior.get("scale") is None:
                raise RateConfigError(f"{fitclass} normal prior needs loc and scale")

        if distribution == "split_normal":
            needed = {"loc", "scale_bright", "scale_faint"}
            missing_split = needed - set(prior)
            if missing_split:
                raise RateConfigError(
                    f"{fitclass} split_normal prior needs {sorted(missing_split)}"
                )

        if distribution == "uniform":
            if prior.get("min") is None or prior.get("max") is None:
                raise RateConfigError(f"{fitclass} uniform prior needs min and max")

        if distribution == "fixed" and prior.get("loc") is None:
            raise RateConfigError(f"{fitclass} fixed prior needs loc")

        if distribution == "children_mixture" and not prior.get("children"):
            raise RateConfigError(f"{fitclass} children_mixture prior needs children")


def validate_rate_config(
    rate_config: Mapping[str, Any],
    available_fitclasses: Optional[Iterable[str]] = None,
) -> None:
    """Validate basic structure and optional coverage of a rate config."""

    entries = get_rate_entries(rate_config)
    required = {
        "rate_gpc3_yr",
        "status",
        "rate_kind",
        "parent",
        "fraction_of_parent",
        "uncertainty",
        "sources",
        "notes",
    }

    for fitclass, entry in entries.items():
        missing = required - set(entry)
        if missing:
            raise RateConfigError(
                f"{fitclass} is missing required rate keys: {sorted(missing)}"
            )

        status = entry["status"]
        if status not in RATE_STATUSES:
            raise RateConfigError(f"{fitclass} has invalid status {status!r}")

        rate_kind = entry["rate_kind"]
        if rate_kind not in RATE_KINDS:
            raise RateConfigError(f"{fitclass} has invalid rate_kind {rate_kind!r}")

        rate = entry.get("rate_gpc3_yr")
        if rate is not None and float(rate) < 0:
            raise RateConfigError(f"{fitclass} has a negative rate")

        fraction = entry.get("fraction_of_parent")
        if fraction is not None and not (0 <= float(fraction) <= 1):
            raise RateConfigError(f"{fitclass} has invalid fraction_of_parent")

        if not isinstance(entry.get("sources"), list):
            raise RateConfigError(f"{fitclass} sources must be a list")

    if available_fitclasses is not None:
        available = set(available_fitclasses)
        configured = set(entries)
        missing_entries = sorted(available - configured)
        if missing_entries:
            raise RateConfigError(
                "Rate config has no entry for fitclasses: "
                + ", ".join(missing_entries)
            )

        if "magabs_priors" in rate_config:
            validate_magabs_config(rate_config, available_fitclasses=available_fitclasses)


def _children_from_config(rate_config: Mapping[str, Any]) -> dict[str, list[str]]:
    """Merge built-in taxonomy descendants with configured child relations."""

    children = {key: list(value) for key, value in DEFAULT_OVERLAP_CHILDREN.items()}
    for fitclass, entry in get_rate_entries(rate_config).items():
        configured_children = entry.get("children")
        if configured_children:
            merged = set(children.get(fitclass, []))
            merged.update(configured_children)
            children[fitclass] = sorted(merged)

    return children


def _descendants(fitclass: str, children: Mapping[str, Sequence[str]]) -> set[str]:
    """Return every recursive descendant of one fitclass."""

    seen: set[str] = set()
    stack = list(children.get(fitclass, []))

    while stack:
        child = stack.pop()
        if child in seen:
            continue

        seen.add(child)
        stack.extend(children.get(child, []))

    return seen


def find_overlapping_fitclasses(
    fitclasses: Iterable[str],
    rate_config: Optional[Mapping[str, Any]] = None,
) -> list[tuple[str, str]]:
    """Return active fitclass pairs that would double-count populations."""

    active = list(dict.fromkeys(fitclasses))
    children = (
        _children_from_config(rate_config)
        if rate_config is not None
        else DEFAULT_OVERLAP_CHILDREN
    )

    descendants = {fitclass: _descendants(fitclass, children) for fitclass in active}
    overlaps = []

    for i, left in enumerate(active):
        left_family = descendants[left] | {left}
        for right in active[i + 1 :]:
            right_family = descendants[right] | {right}
            if right in descendants[left] or left in descendants[right]:
                overlaps.append((left, right))
            elif left_family & right_family:
                overlaps.append((left, right))

    return overlaps


def resolve_rate(
    fitclass: str,
    rate_config: Mapping[str, Any],
    _seen: Optional[set[str]] = None,
) -> ResolvedRate:
    """Resolve a fitclass rate, following parent/child derivations."""

    entries = get_rate_entries(rate_config)
    if fitclass not in entries:
        raise RateConfigError(f"No rate entry for {fitclass!r}")

    if _seen is None:
        _seen = set()
    if fitclass in _seen:
        raise RateConfigError(f"Cyclic rate derivation involving {fitclass!r}")
    _seen.add(fitclass)

    entry = entries[fitclass]
    status = entry["status"]
    rate_kind = entry["rate_kind"]
    parent = entry.get("parent")
    fraction = entry.get("fraction_of_parent")
    children = entry.get("children") or []
    sources = tuple(deepcopy(entry.get("sources", [])))
    notes = entry.get("notes", "")
    derived_from: list[str] = []

    rate = entry.get("rate_gpc3_yr")
    if rate is not None:
        resolved_rate = float(rate)
    elif rate_kind == "fraction_of_parent" and parent is not None and fraction is not None:
        parent_rate = resolve_rate(parent, rate_config, _seen=set(_seen))
        if parent_rate.rate_gpc3_yr is None:
            resolved_rate = None
        else:
            resolved_rate = parent_rate.rate_gpc3_yr * float(fraction)
            derived_from.append(parent)
    elif rate_kind == "derived_from_children" and children:
        child_rates = [
            resolve_rate(child, rate_config, _seen=set(_seen))
            for child in children
        ]
        if any(child.rate_gpc3_yr is None for child in child_rates):
            resolved_rate = None
        else:
            resolved_rate = float(sum(child.rate_gpc3_yr for child in child_rates))
            derived_from.extend(children)
    else:
        resolved_rate = None

    return ResolvedRate(
        fitclass=fitclass,
        rate_gpc3_yr=resolved_rate,
        status=status,
        rate_kind=rate_kind,
        parent=parent,
        fraction_of_parent=fraction,
        sources=sources,
        notes=notes,
        derived_from=tuple(derived_from),
    )


def validate_active_fitclasses(
    fitclasses: Iterable[str],
    rate_config: Mapping[str, Any],
    *,
    allow_missing_rates: bool = False,
    allow_overlaps: bool = False,
) -> list[str]:
    """Validate active fitclasses and return them as a stable unique list."""

    active = list(dict.fromkeys(fitclasses))
    validate_rate_config(rate_config)

    missing = []
    for fitclass in active:
        resolved = resolve_rate(fitclass, rate_config)
        if resolved.status == "missing_direct_rate" or resolved.rate_gpc3_yr is None:
            missing.append(fitclass)

    if missing and not allow_missing_rates:
        raise MissingRateError(
            "Active fitclasses have no usable literature rate: "
            + ", ".join(missing)
        )

    overlaps = find_overlapping_fitclasses(active, rate_config)
    if overlaps and not allow_overlaps:
        formatted = ", ".join(f"{left} + {right}" for left, right in overlaps)
        raise OverlapRateError(
            "Active fitclasses overlap and would double-count rates: " + formatted
        )

    return active


def rate_config_to_dataframe(rate_config: Mapping[str, Any]):
    """Return a pandas DataFrame with resolved rates and provenance."""

    import pandas as pd

    rows = []
    magabs_priors = get_magabs_priors(rate_config)
    for fitclass in sorted(get_rate_entries(rate_config)):
        entry = get_rate_entries(rate_config)[fitclass]
        resolved = resolve_rate(fitclass, rate_config)
        magabs_prior = magabs_priors.get(fitclass, {})
        rows.append(
            {
                "fitclass": fitclass,
                "rate_gpc3_yr": resolved.rate_gpc3_yr,
                "status": resolved.status,
                "rate_kind": resolved.rate_kind,
                "parent": resolved.parent,
                "fraction_of_parent": resolved.fraction_of_parent,
                "uncertainty_plus": (entry.get("uncertainty") or {}).get("plus"),
                "uncertainty_minus": (entry.get("uncertainty") or {}).get("minus"),
                "derived_from": ",".join(resolved.derived_from),
                "source_bibkeys": ",".join(resolved.source_bibkeys),
                "notes": resolved.notes,
                "magabs_status": magabs_prior.get("status"),
                "magabs_distribution": magabs_prior.get("distribution"),
                "magabs_loc": magabs_prior.get("loc"),
                "magabs_scale": magabs_prior.get("scale"),
                "magabs_min": magabs_prior.get("min"),
                "magabs_max": magabs_prior.get("max"),
                "magabs_source_band": magabs_prior.get("source_band"),
                "magabs_bibkeys": ",".join(_prior_source_bibkeys(magabs_prior)),
                "magabs_notes": magabs_prior.get("notes"),
            }
        )

    return pd.DataFrame(rows).set_index("fitclass")


def _prior_source_bibkeys(prior: Mapping[str, Any]) -> tuple[str, ...]:
    """Return non-empty BibTeX keys attached to a magnitude prior."""

    return tuple(
        source.get("bibkey", "")
        for source in prior.get("sources", [])
        if source.get("bibkey")
    )


def _tuple_magabs_prior(value: Sequence[float]) -> dict[str, Any]:
    """Convert a compact user magnitude tuple into a full prior mapping."""

    loc, *scale = value
    if not scale:
        return {
            "status": "user_override",
            "distribution": "fixed",
            "loc": float(loc),
            "sncosmo_band": "bessellb",
            "magsys": "ab",
            "source_band": "user",
            "sources": [],
            "notes": "User-provided fixed absolute-magnitude override.",
        }

    if len(scale) == 1:
        return {
            "status": "user_override",
            "distribution": "normal",
            "loc": float(loc),
            "scale": float(scale[0]),
            "sncosmo_band": "bessellb",
            "magsys": "ab",
            "source_band": "user",
            "sources": [],
            "notes": "User-provided normal absolute-magnitude override.",
        }

    return {
        "status": "user_override",
        "distribution": "split_normal",
        "loc": float(loc),
        "scale_bright": float(scale[0]),
        "scale_faint": float(scale[1]),
        "sncosmo_band": "bessellb",
        "magsys": "ab",
        "source_band": "user",
        "sources": [],
        "notes": "User-provided split-normal absolute-magnitude override.",
    }


def _get_magabs_prior(
    fitclass: str,
    rate_config: Mapping[str, Any],
    magabs_by_fitclass: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return a user override or configured magnitude prior for one fitclass."""

    if magabs_by_fitclass and fitclass in magabs_by_fitclass:
        override = magabs_by_fitclass[fitclass]
        if isinstance(override, Mapping):
            return override
        return _tuple_magabs_prior(override)

    priors = get_magabs_priors(rate_config)
    if fitclass not in priors:
        raise MissingMagnitudePriorError(
            f"No absolute-magnitude prior configured for {fitclass!r}"
        )

    return priors[fitclass]


def _sample_magabs_from_prior(
    prior: Mapping[str, Any],
    rng: np.random.Generator,
) -> float:
    """Draw one absolute magnitude from a validated direct prior."""

    distribution = prior["distribution"]

    if distribution in {"missing_direct_prior", "children_mixture"}:
        raise MissingMagnitudePriorError(
            f"Cannot directly sample magabs distribution {distribution!r}"
        )

    if distribution == "fixed":
        return float(prior["loc"])

    if distribution == "normal":
        return float(rng.normal(float(prior["loc"]), float(prior["scale"])))

    if distribution == "split_normal":
        loc = float(prior["loc"])
        if rng.random() < 0.5:
            return float(loc - abs(rng.normal(0.0, float(prior["scale_bright"]))))
        return float(loc + abs(rng.normal(0.0, float(prior["scale_faint"]))))

    if distribution == "uniform":
        return float(rng.uniform(float(prior["min"]), float(prior["max"])))

    raise RateConfigError(f"Unsupported magabs distribution {distribution!r}")


def _draw_magabs_with_provenance(
    fitclass: str,
    rng: np.random.Generator,
    rate_config: Mapping[str, Any],
    magabs_by_fitclass: Optional[Mapping[str, Any]],
) -> tuple[float, Mapping[str, Any], str]:
    """Draw one magnitude and return its resolved prior provenance."""

    prior = _get_magabs_prior(fitclass, rate_config, magabs_by_fitclass)

    if (
        prior.get("status") == "missing_direct_prior"
        or prior.get("distribution") == "missing_direct_prior"
    ):
        raise MissingMagnitudePriorError(
            f"No usable absolute-magnitude prior for active fitclass {fitclass!r}"
        )

    if prior.get("distribution") == "children_mixture":
        children = list(prior.get("children", []))
        if not children:
            raise MissingMagnitudePriorError(
                f"children_mixture prior for {fitclass!r} has no children"
            )

        weights = []
        for child in children:
            try:
                resolved = resolve_rate(child, rate_config)
                weights.append(resolved.rate_gpc3_yr or 0.0)
            except RateConfigError:
                weights.append(0.0)

        weights = np.asarray(weights, dtype=float)
        probabilities = weights / weights.sum() if weights.sum() > 0 else None
        child = str(rng.choice(children, p=probabilities))
        return _draw_magabs_with_provenance(
            child,
            rng,
            rate_config,
            magabs_by_fitclass,
        )

    return _sample_magabs_from_prior(prior, rng), prior, fitclass

__all__ = [
    "DEFAULT_ACTIVE_FITCLASSES_BROAD",
    "DEFAULT_ACTIVE_FITCLASSES_IA_SUBTYPE",
    "MissingRateError",
    "MissingMagnitudePriorError",
    "OverlapRateError",
    "RATE_CONFIG_FILENAME",
    "RateConfigError",
    "ResolvedRate",
    "default_rate_config_path",
    "discover_warp_fitclasses",
    "find_overlapping_fitclasses",
    "get_magabs_priors",
    "load_warp_rate_config",
    "rate_config_to_dataframe",
    "resolve_rate",
    "validate_active_fitclasses",
    "validate_magabs_config",
    "validate_rate_config",
]
