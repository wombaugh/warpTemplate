"""Locate optional coefficient data used by integration-style tests."""

from __future__ import annotations

import os
from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CONFIGURED_DIR = os.environ.get("WARPTEMPLATE_TEST_COEFFICIENT_DIR")
_CANDIDATE_DIRS = (
    Path(_CONFIGURED_DIR).expanduser() if _CONFIGURED_DIR else None,
    REPOSITORY_ROOT / "data" / "warpcoeff_v4",
    REPOSITORY_ROOT / "data" / "warpcoeff_v3",
)
COEFFICIENT_DIR = next(
    (path for path in _CANDIDATE_DIRS if path is not None and path.is_dir()),
    REPOSITORY_ROOT / "data" / "warpcoeff_v4",
)


def has_coefficients(*fitclasses: str) -> bool:
    """Return whether coefficient files exist for every requested fit class."""

    return all(
        any(COEFFICIENT_DIR.glob(f"warpcoeffs_v*_{re.sub(r'/', '', name)}*.pkl"))
        for name in fitclasses
    )


def requires_coefficients(*fitclasses: str):
    """Skip an integration test when its external coefficient data is absent."""

    return unittest.skipUnless(
        has_coefficients(*fitclasses),
        "external coefficient library is unavailable; set "
        "WARPTEMPLATE_TEST_COEFFICIENT_DIR",
    )
