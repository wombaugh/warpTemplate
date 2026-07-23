"""Persistent, lossless cache for colour-neutral Warp source grids."""

from __future__ import annotations

from hashlib import blake2b
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py
import numpy as np
import sncosmo

from .loaders import WarpfitTemplateLoader, WarpTemplateDescriptor
from .sources import DynamicColorWarpSource


CACHE_SCHEMA_VERSION = 1


def _package_version(name: str) -> Optional[str]:
    """Return an installed package version without requiring installation."""

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _file_digest(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a stable digest without loading a complete coefficient file."""

    digest = blake2b(digest_size=20)
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    """Hash source arrays including dtype and shape metadata."""

    digest = blake2b(digest_size=20)
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


class WarpSourceCache:
    """Build and read fitclass-partitioned HDF5 source-grid caches."""

    def __init__(
        self,
        warpcoeffs_dir: str | Path,
        cache_dir: str | Path,
        *,
        loader: Optional[WarpfitTemplateLoader] = None,
    ) -> None:
        """Bind the cache to one coefficient library and output directory."""

        self.warpcoeffs_dir = Path(warpcoeffs_dir)
        self.cache_dir = Path(cache_dir)
        self.loader = loader or WarpfitTemplateLoader(str(self.warpcoeffs_dir))
        self._validity: dict[str, tuple[int, int, bool]] = {}

    @staticmethod
    def _entry_id(template_key: str) -> str:
        """Map an arbitrary stable template key to an HDF5-safe identifier."""

        return blake2b(template_key.encode(), digest_size=12).hexdigest()

    @staticmethod
    def _safe_fitclass(fitclass: str) -> str:
        """Return the filename spelling used by coefficient files."""

        return fitclass.replace("/", "")

    def fitclass_path(self, fitclass: str) -> Path:
        """Return the HDF5 path for one fitclass cache partition."""

        return self.cache_dir / f"warp_sources_v1_{self._safe_fitclass(fitclass)}.h5"

    def _expected_metadata(self, fitclass: str) -> dict[str, Any]:
        """Return metadata required for accepting one cache partition."""

        coefficient_path = self.loader.coefficient_path(fitclass)
        if not coefficient_path.exists():
            raise FileNotFoundError(coefficient_path)
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fitclass": fitclass,
            "coefficient_digest": _file_digest(coefficient_path),
            "sncosmo_version": sncosmo.__version__,
            "warpTemplate_version": _package_version("warpTemplate") or "0.1.0",
            "complete": True,
        }

    @staticmethod
    def _content_metadata_is_valid(handle: h5py.File) -> bool:
        """Validate entry structure and the stored base-source fingerprints."""

        try:
            fingerprints_json = str(handle.attrs["source_fingerprints"])
            fingerprints = json.loads(fingerprints_json)
            expected_digest = blake2b(
                fingerprints_json.encode(), digest_size=20
            ).hexdigest()
            if handle.attrs.get("source_fingerprint") != expected_digest:
                return False
            groups = handle["entries"]
            if len(groups) != int(handle.attrs["entry_count"]):
                return False
            for group in groups.values():
                template_sn = str(group.attrs["template_sn"])
                if fingerprints.get(template_sn) != group.attrs.get("source_digest"):
                    return False
                if not {"phase", "wave", "flux"}.issubset(group.keys()):
                    return False
                if group["flux"].shape != (
                    group["phase"].shape[0],
                    group["wave"].shape[0],
                ):
                    return False
            return True
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def is_valid(self, fitclass: str) -> bool:
        """Return whether a complete partition matches the current inputs."""

        coefficient_path = self.loader.coefficient_path(fitclass)
        if not coefficient_path.exists():
            return False
        coefficient_stat = coefficient_path.stat()
        cached = self._validity.get(fitclass)
        if cached is not None and cached[:2] == (
            coefficient_stat.st_mtime_ns,
            coefficient_stat.st_size,
        ):
            return cached[2]
        path = self.fitclass_path(fitclass)
        if not path.exists():
            self._validity[fitclass] = (
                coefficient_stat.st_mtime_ns,
                coefficient_stat.st_size,
                False,
            )
            return False
        expected = self._expected_metadata(fitclass)
        try:
            with h5py.File(path, "r") as handle:
                valid = all(
                    handle.attrs.get(key) == value for key, value in expected.items()
                )
                valid = valid and self._content_metadata_is_valid(handle)
                self._validity[fitclass] = (
                    coefficient_stat.st_mtime_ns,
                    coefficient_stat.st_size,
                    bool(valid),
                )
                return bool(valid)
        except (OSError, ValueError):
            self._validity[fitclass] = (
                coefficient_stat.st_mtime_ns,
                coefficient_stat.st_size,
                False,
            )
            return False

    def _build_fitclass(self, fitclass: str) -> dict[str, Any]:
        """Write one complete fitclass partition and publish it atomically."""

        path = self.fitclass_path(fitclass)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        expected = self._expected_metadata(fitclass)
        entries = self.loader.get_entry_probabilities(fitclass)
        source_fingerprints: dict[str, str] = {}
        try:
            with h5py.File(temporary, "w") as handle:
                for key, value in expected.items():
                    handle.attrs[key] = value
                handle.attrs["complete"] = False
                handle.attrs["entry_count"] = len(entries)
                groups = handle.create_group("entries")
                for descriptor, _ in entries:
                    source = self.loader.build_uncolored_source(descriptor)
                    group = groups.create_group(self._entry_id(descriptor.template_key))
                    group.attrs["template_key"] = descriptor.template_key
                    group.attrs["basis_sn"] = descriptor.basis_sn
                    group.attrs["template_index"] = descriptor.template_index
                    group.attrs["template_sn"] = descriptor.template_sn
                    for name, values in (
                        ("phase", source._phase),
                        # Persist only native data; zero-flux edge coverage is
                        # a deterministic runtime property of the source.
                        ("wave", source.native_wave),
                        ("flux", source.native_flux_grid()),
                    ):
                        group.create_dataset(
                            name,
                            data=np.asarray(values, dtype=np.float64),
                            compression="lzf",
                            shuffle=True,
                            fletcher32=True,
                        )
                    original = sncosmo.get_source(descriptor.template_sn)
                    fingerprint = source_fingerprints.get(descriptor.template_sn)
                    if fingerprint is None:
                        fingerprint = _array_digest(
                            np.asarray(original._phase),
                            np.asarray(original._wave),
                            np.asarray(original._flux(original._phase, original._wave)),
                        )
                        source_fingerprints[descriptor.template_sn] = fingerprint
                    group.attrs["source_digest"] = fingerprint
                fingerprints_json = json.dumps(source_fingerprints, sort_keys=True)
                handle.attrs["source_fingerprints"] = fingerprints_json
                handle.attrs["source_fingerprint"] = blake2b(
                    fingerprints_json.encode(), digest_size=20
                ).hexdigest()
                handle.attrs["complete"] = True
                handle.flush()
            os.replace(temporary, path)
            coefficient_stat = self.loader.coefficient_path(fitclass).stat()
            self._validity[fitclass] = (
                coefficient_stat.st_mtime_ns,
                coefficient_stat.st_size,
                True,
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return self.describe_fitclass(fitclass)

    def build(
        self,
        fitclasses: Optional[Iterable[str]] = None,
        *,
        overwrite: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Build missing or stale partitions and resume at fitclass boundaries."""

        selected = (
            self.loader.available_fitclasses()
            if fitclasses is None
            else list(fitclasses)
        )
        result: dict[str, dict[str, Any]] = {}
        for fitclass in selected:
            if overwrite:
                self._validity.pop(fitclass, None)
            if not overwrite and self.is_valid(fitclass):
                result[fitclass] = self.describe_fitclass(fitclass)
                continue
            result[fitclass] = self._build_fitclass(fitclass)
            self.loader.clear_fitclass_cache(fitclass)
        return result

    def load_source(
        self, descriptor: WarpTemplateDescriptor | dict[str, Any]
    ) -> Optional[DynamicColorWarpSource]:
        """Load one source from a valid cache, or return ``None`` on a miss."""

        if not isinstance(descriptor, WarpTemplateDescriptor):
            fields = WarpTemplateDescriptor.__dataclass_fields__
            descriptor = WarpTemplateDescriptor(
                **{name: descriptor[name] for name in fields}
            )
        if not self.is_valid(descriptor.fitclass):
            return None
        path = self.fitclass_path(descriptor.fitclass)
        entry_id = self._entry_id(descriptor.template_key)
        try:
            with h5py.File(path, "r") as handle:
                group = handle["entries"].get(entry_id)
                if group is None or group.attrs.get("template_key") != descriptor.template_key:
                    return None
                return DynamicColorWarpSource(
                    group["phase"][:],
                    group["wave"][:],
                    group["flux"][:],
                    name=f"{descriptor.basis_sn}_{descriptor.template_sn}",
                )
        except (OSError, KeyError, ValueError):
            return None

    def describe_fitclass(self, fitclass: str) -> dict[str, Any]:
        """Return JSON-safe cache provenance for manifests and diagnostics."""

        path = self.fitclass_path(fitclass)
        if not self.is_valid(fitclass):
            return {"valid": False, "path": str(path)}
        with h5py.File(path, "r") as handle:
            return {
                "valid": True,
                "path": str(path),
                "schema_version": int(handle.attrs["schema_version"]),
                "coefficient_digest": str(handle.attrs["coefficient_digest"]),
                "source_fingerprint": str(handle.attrs["source_fingerprint"]),
                "sncosmo_version": str(handle.attrs["sncosmo_version"]),
                "warpTemplate_version": str(handle.attrs["warpTemplate_version"]),
                "entry_count": int(handle.attrs["entry_count"]),
            }

    def describe(self, fitclasses: Iterable[str]) -> dict[str, Any]:
        """Summarize cache availability for a simulation manifest."""

        partitions = {
            fitclass: self.describe_fitclass(fitclass) for fitclass in fitclasses
        }
        payload = json.dumps(partitions, sort_keys=True).encode()
        return {
            "enabled": True,
            "directory": str(self.cache_dir),
            "fingerprint": blake2b(payload, digest_size=16).hexdigest(),
            "partitions": partitions,
        }


__all__ = ["CACHE_SCHEMA_VERSION", "WarpSourceCache"]
