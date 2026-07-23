"""SuperNNova adaptation, training, prediction, and comparison utilities.

The installed :mod:`supernnova` package supplies the upstream training utilities.
This module translates Warp schema-6 photometry into its packed-sequence contract
while reusing the classifier project's frozen taxonomy, grouped splits, and artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from . import classification as workflow


SUPERNOVA_ZEROPOINT = 27.5
DATABASE_SCHEMA_VERSION = 2
FLUX_NORMALIZATION_FLOOR = -2000.0
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
REDSHIFT_MODES = ("photometry_only", "photometry_plus_truth_z")
FLUX_FEATURES = tuple(f"FLUXCAL_{band}" for band in workflow.EXPECTED_BANDS)
FLUXERR_FEATURES = tuple(f"FLUXCALERR_{band}" for band in workflow.EXPECTED_BANDS)
PRESENCE_FEATURES = tuple(f"PRESENT_{band}" for band in workflow.EXPECTED_BANDS)
PHOTOMETRY_FEATURES = (*FLUX_FEATURES, *FLUXERR_FEATURES, *PRESENCE_FEATURES, "delta_time")
REDSHIFT_FEATURES = ("HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR")
ALL_FEATURES = (*PHOTOMETRY_FEATURES, *REDSHIFT_FEATURES)
NORMALIZED_FEATURES = (*FLUX_FEATURES, *FLUXERR_FEATURES, "delta_time")


@dataclass(frozen=True)
class SuperNNovaTrainingConfig:
    """Serializable architecture and optimization settings for one RNN run."""

    redshift_mode: str = "photometry_only"
    seed: int = workflow.DEFAULT_SPLIT_SEED
    epochs: int = 90
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.05
    batch_size: int = 128
    learning_rate: float = 1e-3
    bidirectional: bool = True
    rnn_output_option: str = "mean"
    random_length: bool = True
    lr_patience: int = 5
    lr_factor: float = 0.5
    early_stopping_patience: int = 12
    device: str = "auto"
    threads: int = 14

    def __post_init__(self) -> None:
        """Reject configurations that violate the supported experiment contract."""
        if self.redshift_mode not in REDSHIFT_MODES:
            raise ValueError(f"Unsupported redshift mode: {self.redshift_mode!r}")
        if self.epochs < 1 or self.batch_size < 1 or self.hidden_dim < 1:
            raise ValueError("Epochs, batch size, and hidden dimension must be positive")
        if self.num_layers < 1 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("Invalid recurrent layer count or dropout")
        if self.lr_patience < 1 or self.early_stopping_patience < self.lr_patience:
            raise ValueError("Early-stopping patience must be at least LR patience")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError("lr_factor must lie strictly between zero and one")

    def normalized(self) -> dict[str, Any]:
        """Return a stable JSON-compatible configuration mapping."""
        return asdict(self)


@dataclass
class PredictionResult:
    """Hold model probabilities together with partial-curve eligibility counts."""

    classifications: Any
    eligible_objects: int
    ineligible_objects: int
    cutoff_days: float | None


def feature_names(redshift_mode: str) -> tuple[str, ...]:
    """Return the fixed feature order for a supported redshift mode."""
    if redshift_mode not in REDSHIFT_MODES:
        raise ValueError(f"Unsupported redshift mode: {redshift_mode!r}")
    selected = PHOTOMETRY_FEATURES
    if redshift_mode == "photometry_plus_truth_z":
        selected = (*selected, *REDSHIFT_FEATURES)
    if redshift_mode == "photometry_only" and any("Z" in name for name in selected):
        raise AssertionError("Photometry-only features unexpectedly contain redshift")
    return selected


def build_execution_role_manifest(
    split_manifest: pd.DataFrame,
    execution_mode: str,
    objects_per_class: int = 64,
    seed: int = workflow.DEFAULT_SPLIT_SEED,
) -> pd.DataFrame:
    """Map persistent folds to full roles or ParSNIP-matched smoke roles."""
    if execution_mode not in {"smoke", "full_cpu", "full_gpu_repeats"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode!r}")
    if execution_mode != "smoke":
        manifest = split_manifest.copy()
        manifest["role"] = manifest["split"]
        return manifest

    selected = {
        "train": workflow.sample_balanced_object_ids(
            split_manifest,
            per_class=objects_per_class,
            seed=seed,
            folds=list(range(4, 10)),
        ),
        "validation": workflow.sample_balanced_object_ids(
            split_manifest,
            per_class=objects_per_class,
            seed=seed + 1,
            folds=[3],
        ),
        "test": workflow.sample_balanced_object_ids(
            split_manifest,
            per_class=objects_per_class,
            seed=seed + 2,
            folds=[2],
        ),
    }
    id_to_role = {
        object_id: role for role, object_ids in selected.items() for object_id in object_ids
    }
    manifest = split_manifest[
        split_manifest["object_id"].isin(id_to_role)
    ].copy()
    manifest["role"] = manifest["object_id"].map(id_to_role)
    group_sets = {
        role: set(manifest.loc[manifest["role"] == role, "group_id"])
        for role in SPLIT_CODES
    }
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise ValueError("Smoke roles reuse a persistent provenance group")
    return manifest


def _sequence_from_grouped_rows(
    grouped_rows: pd.DataFrame,
    redshift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot grouped observations into the fixed all-feature sequence representation."""
    grouped_times = np.sort(grouped_rows["grouped_mjd"].unique().astype(float))
    sequence = np.zeros((len(grouped_times), len(ALL_FEATURES)), dtype=np.float32)
    time_to_index = {value: index for index, value in enumerate(grouped_times)}
    band_to_index = {band: index for index, band in enumerate(workflow.EXPECTED_BANDS)}

    for row in grouped_rows.itertuples(index=False):
        time_index = time_to_index[float(row.grouped_mjd)]
        band_index = band_to_index[str(row.band)]
        sequence[time_index, band_index] = float(row.flux)
        sequence[time_index, len(FLUX_FEATURES) + band_index] = float(row.fluxerr)
        sequence[
            time_index,
            len(FLUX_FEATURES) + len(FLUXERR_FEATURES) + band_index,
        ] = 1.0

    # The first delta is zero by definition; later values measure grouped-epoch gaps.
    sequence[:, PHOTOMETRY_FEATURES.index("delta_time")] = np.concatenate(
        ([0.0], np.diff(grouped_times))
    )
    sequence[:, ALL_FEATURES.index("HOSTGAL_SPECZ")] = float(redshift)
    # Exact simulated redshift has no measurement uncertainty in the schema-6 truth.
    sequence[:, ALL_FEATURES.index("HOSTGAL_SPECZ_ERR")] = 0.0
    if not np.isfinite(sequence).all():
        raise ValueError("SuperNNova sequence contains non-finite values")
    if (sequence[:, PHOTOMETRY_FEATURES.index("delta_time")] < 0).any():
        raise ValueError("SuperNNova sequence contains negative delta times")
    return sequence, grouped_times.astype(np.float64)


def build_supernnova_sequences(
    truth: pd.DataFrame,
    observations: pd.DataFrame,
    object_ids: Sequence[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert selected Warp observations into all-feature SuperNNova sequences."""
    required_truth = {"object_id", "z"}
    required_observations = {"object_id", "mjd", "band", "flux", "fluxerr", "zp"}
    if missing := required_truth - set(truth):
        raise ValueError(f"Missing truth columns: {sorted(missing)}")
    if missing := required_observations - set(observations):
        raise ValueError(f"Missing observation columns: {sorted(missing)}")

    selected_ids = set(map(str, object_ids)) if object_ids is not None else None
    selected_truth = truth.copy()
    selected_truth["object_id"] = selected_truth["object_id"].astype(str)
    if selected_ids is not None:
        selected_truth = selected_truth[selected_truth["object_id"].isin(selected_ids)]
    selected_observations = observations.copy()
    selected_observations["object_id"] = selected_observations["object_id"].astype(str)
    if selected_ids is not None:
        selected_observations = selected_observations[
            selected_observations["object_id"].isin(selected_ids)
        ]
    unknown_bands = sorted(set(selected_observations["band"]) - set(workflow.EXPECTED_BANDS))
    if unknown_bands:
        raise ValueError(f"Unexpected SuperNNova bands: {unknown_bands}")
    if (selected_observations["fluxerr"] <= 0).any():
        raise ValueError("SuperNNova requires strictly positive reported flux errors")

    converted = workflow.rescale_flux_to_zeropoint(
        selected_observations, target_zeropoint=SUPERNOVA_ZEROPOINT
    )
    grouped = workflow.group_observing_epochs(converted)
    redshifts = selected_truth.set_index("object_id")["z"].astype(float)
    sequences: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for object_id, rows in grouped.groupby("object_id", sort=False):
        object_id = str(object_id)
        if object_id not in redshifts.index:
            raise ValueError(f"Observations have no matching truth row: {object_id}")
        sequences[object_id] = _sequence_from_grouped_rows(rows, redshifts.loc[object_id])
    return sequences


def _create_database_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    """Create resizable datasets used by the local SuperNNova loader and backend."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    float_array_dtype = h5py.vlen_dtype(np.dtype("float32"))
    time_array_dtype = h5py.vlen_dtype(np.dtype("float64"))
    datasets = {
        "data": handle.create_dataset("data", shape=(0,), maxshape=(None,), dtype=float_array_dtype),
        "grouped_mjd": handle.create_dataset(
            "grouped_mjd", shape=(0,), maxshape=(None,), dtype=time_array_dtype
        ),
        "SNID": handle.create_dataset("SNID", shape=(0,), maxshape=(None,), dtype=string_dtype),
        "target": handle.create_dataset(
            "target_7classes", shape=(0,), maxshape=(None,), dtype=np.int8
        ),
        "split": handle.create_dataset(
            "dataset_photometry_7classes", shape=(0,), maxshape=(None,), dtype=np.int8
        ),
        "role": handle.create_dataset("role", shape=(0,), maxshape=(None,), dtype=string_dtype),
        "group_id": handle.create_dataset(
            "group_id", shape=(0,), maxshape=(None,), dtype=string_dtype
        ),
        "t0": handle.create_dataset("t0", shape=(0,), maxshape=(None,), dtype=np.float64),
        "z": handle.create_dataset("z", shape=(0,), maxshape=(None,), dtype=np.float64),
        "survey_realization_id": handle.create_dataset(
            "survey_realization_id", shape=(0,), maxshape=(None,), dtype=string_dtype
        ),
    }
    features = handle.create_dataset("features", shape=(len(ALL_FEATURES),), dtype=string_dtype)
    features[:] = list(ALL_FEATURES)
    datasets["data"].attrs["n_features"] = len(ALL_FEATURES)
    return datasets


def _append_database_record(
    datasets: Mapping[str, h5py.Dataset],
    object_id: str,
    sequence: np.ndarray,
    grouped_mjd: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    """Append one light curve and its split/provenance metadata to HDF5."""
    index = len(datasets["SNID"])
    for dataset in datasets.values():
        dataset.resize((index + 1,))
    datasets["data"][index] = sequence.reshape(-1)
    datasets["grouped_mjd"][index] = grouped_mjd
    datasets["SNID"][index] = object_id
    datasets["target"][index] = workflow.FINAL_CLASSES.index(str(metadata["final_label"]))
    datasets["split"][index] = SPLIT_CODES[str(metadata["role"])]
    datasets["role"][index] = str(metadata["role"])
    datasets["group_id"][index] = str(metadata["group_id"])
    datasets["t0"][index] = float(metadata["t0"])
    datasets["z"][index] = float(metadata["z"])
    datasets["survey_realization_id"][index] = str(
        metadata.get("survey_realization_id", "")
    )


def _write_training_normalization(handle: h5py.File) -> dict[str, tuple[float, float, float]]:
    """Derive and store every normalization using training sequences only."""
    split_codes = handle["dataset_photometry_7classes"][:]
    train_indices = np.flatnonzero(split_codes == SPLIT_CODES["train"])
    if not len(train_indices):
        raise ValueError("Cannot normalize a SuperNNova database without training objects")
    n_features = int(handle["data"].attrs["n_features"])
    delta_index = ALL_FEATURES.index("delta_time")

    # A first pass obtains the shift required by SuperNNova's logarithm without
    # materializing every training time step in memory.
    minima = {"FLUXCAL": np.inf, "FLUXCALERR": np.inf, "delta_time": np.inf}
    for index in train_indices:
        sequence = handle["data"][index].reshape(-1, n_features)
        minima["FLUXCAL"] = min(
            minima["FLUXCAL"], float(sequence[:, : len(FLUX_FEATURES)].min())
        )
        minima["FLUXCALERR"] = min(
            minima["FLUXCALERR"],
            float(
                sequence[
                    :,
                    len(FLUX_FEATURES) : len(FLUX_FEATURES)
                    + len(FLUXERR_FEATURES),
                ].min()
            ),
        )
        minima["delta_time"] = min(
            minima["delta_time"], float(sequence[:, delta_index].min())
        )
    # SuperNNova's reference preprocessing caps extreme negative FLUXCAL values
    # before normalization so a single noise outlier cannot collapse the useful scale.
    minima["FLUXCAL"] = max(minima["FLUXCAL"], FLUX_NORMALIZATION_FLOOR)

    # A second pass accumulates log-space moments using constant memory.
    sums = {key: 0.0 for key in minima}
    squared_sums = {key: 0.0 for key in minima}
    counts = {key: 0 for key in minima}
    for index in train_indices:
        sequence = handle["data"][index].reshape(-1, n_features)
        values_by_group = {
            "FLUXCAL": sequence[:, : len(FLUX_FEATURES)],
            "FLUXCALERR": sequence[
                :,
                len(FLUX_FEATURES) : len(FLUX_FEATURES)
                + len(FLUXERR_FEATURES),
            ],
            "delta_time": sequence[:, delta_index],
        }
        for key, values in values_by_group.items():
            clipped = np.clip(values.astype(np.float64), minima[key], np.inf)
            logged = np.log(clipped - minima[key] + 1e-5)
            sums[key] += float(logged.sum())
            squared_sums[key] += float(np.square(logged).sum())
            counts[key] += int(logged.size)

    normalization = {}
    for key in minima:
        mean = sums[key] / counts[key]
        variance = max(squared_sums[key] / counts[key] - mean**2, 0.0)
        normalization[key] = (float(minima[key]), float(mean), max(variance**0.5, 1e-6))
    per_feature = handle.create_group("normalizations")
    global_group = handle.create_group("normalizations_global")
    for prefix in ("FLUXCAL", "FLUXCALERR"):
        group = global_group.create_group(prefix)
        for key, value in zip(("min", "mean", "std"), normalization[prefix]):
            group.create_dataset(key, data=value)
    delta_group = per_feature.create_group("delta_time")
    for key, value in zip(("min", "mean", "std"), normalization["delta_time"]):
        delta_group.create_dataset(key, data=value)
    return normalization


def _database_identity(
    role_manifest: pd.DataFrame,
    minimum_training_epochs: int,
) -> str:
    """Hash ordered membership and preprocessing choices for cache validation."""
    columns = ["object_id", "role", "final_label", "group_id"]
    ordered = role_manifest[columns].astype(str).sort_values("object_id")
    digest = sha256()
    digest.update(ordered.to_csv(index=False).encode())
    digest.update(json.dumps(list(ALL_FEATURES)).encode())
    digest.update(str(DATABASE_SCHEMA_VERSION).encode())
    digest.update(str(SUPERNOVA_ZEROPOINT).encode())
    digest.update(str(minimum_training_epochs).encode())
    return digest.hexdigest()[:16]


def prepare_supernnova_database(
    sample_dir: str | Path,
    truth: pd.DataFrame,
    role_manifest: pd.DataFrame,
    output_path: str | Path,
    minimum_training_epochs: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stream selected schema-6 light curves into a normalized HDF5 database."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    required = {"object_id", "role", "final_label", "group_id"}
    if missing := required - set(role_manifest):
        raise ValueError(f"Role manifest is missing columns: {sorted(missing)}")
    if not set(role_manifest["role"]).issubset(SPLIT_CODES):
        raise ValueError("Role manifest must use train, validation, and test")
    if role_manifest["object_id"].duplicated().any():
        raise ValueError("Role manifest contains duplicate object IDs")

    output_path = Path(output_path)
    identity = _database_identity(role_manifest, minimum_training_epochs)
    if output_path.exists() and not overwrite:
        with h5py.File(output_path, "r") as handle:
            existing = str(handle.attrs.get("database_identity", ""))
            if existing != identity:
                raise FileExistsError(
                    f"Existing SuperNNova database has incompatible identity: {output_path}"
                )
            return json.loads(str(handle.attrs["summary_json"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_ids = set(role_manifest["object_id"].astype(str))
    metadata = role_manifest.copy()
    metadata["object_id"] = metadata["object_id"].astype(str)
    truth_copy = truth.copy()
    truth_copy["object_id"] = truth_copy["object_id"].astype(str)
    truth_index = truth_copy.set_index("object_id", drop=False)
    truth_columns = [
        column
        for column in ("z", "t0", "survey_realization_id")
        if column in truth_copy and column not in metadata
    ]
    if truth_columns:
        metadata = metadata.merge(
            truth_copy[["object_id", *truth_columns]], on="object_id", how="left"
        )
    if metadata[["z", "t0"]].isna().any().any():
        raise ValueError("Every selected object requires finite z and t0 truth")
    metadata_lookup = metadata.set_index("object_id").to_dict("index")

    written: set[str] = set()
    skipped_short: list[str] = []
    requested_value_set = pa.array(sorted(requested_ids))
    with h5py.File(output_path, "w") as handle:
        datasets = _create_database_datasets(handle)
        handle.attrs["requested_ids"] = len(requested_ids)
        for file in workflow.sample_table_files(sample_dir, "observations"):
            table = pq.ParquetFile(file).read(
                columns=["object_id", "mjd", "band", "flux", "fluxerr", "zp"]
            )
            mask = pc.is_in(table["object_id"], value_set=requested_value_set)
            table = table.filter(mask)
            if not table.num_rows:
                continue
            frame = table.to_pandas()
            file_ids = frame["object_id"].astype(str).unique()
            file_truth = truth_index.loc[file_ids].reset_index(drop=True)
            sequences = build_supernnova_sequences(file_truth, frame)
            for object_id, (sequence, grouped_mjd) in sequences.items():
                if object_id in written:
                    raise ValueError(f"Object crosses observation files: {object_id}")
                role = str(metadata_lookup[object_id]["role"])
                if role == "train" and len(sequence) < minimum_training_epochs:
                    skipped_short.append(object_id)
                    continue
                _append_database_record(
                    datasets,
                    object_id,
                    sequence,
                    grouped_mjd,
                    metadata_lookup[object_id],
                )
                written.add(object_id)

        missing_ids = requested_ids - written - set(skipped_short)
        if missing_ids:
            raise ValueError(f"Selected objects lack observations: {sorted(missing_ids)[:5]}")
        normalization = _write_training_normalization(handle)
        handle.attrs["database_identity"] = identity
        handle.attrs["database_schema_version"] = DATABASE_SCHEMA_VERSION
        handle.attrs["zeropoint"] = SUPERNOVA_ZEROPOINT
        handle.attrs["minimum_training_epochs"] = minimum_training_epochs
        handle.attrs["class_order_json"] = json.dumps(list(workflow.FINAL_CLASSES))
        summary = {
            "database_identity": identity,
            "database_schema_version": DATABASE_SCHEMA_VERSION,
            "objects": len(written),
            "skipped_short_training_objects": len(skipped_short),
            "role_counts": {
                role: int((handle["dataset_photometry_7classes"][:] == code).sum())
                for role, code in SPLIT_CODES.items()
            },
            "normalization": {
                key: dict(zip(("min", "mean", "std"), values))
                for key, values in normalization.items()
            },
            "feature_order": list(ALL_FEATURES),
            "zeropoint": SUPERNOVA_ZEROPOINT,
        }
        handle.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
    return summary


def build_supernnova_settings(
    database_path: str | Path,
    config: SuperNNovaTrainingConfig,
    use_cuda: bool = False,
) -> SimpleNamespace:
    """Construct the settings contract used by bundled SuperNNova utilities."""
    database_path = Path(database_path)
    selected_features = feature_names(config.redshift_mode)
    with h5py.File(database_path, "r") as handle:
        all_features = tuple(handle["features"][:].astype(str))
        normalization = {
            "FLUXCAL": tuple(
                float(handle[f"normalizations_global/FLUXCAL/{key}"][()])
                for key in ("min", "mean", "std")
            ),
            "FLUXCALERR": tuple(
                float(handle[f"normalizations_global/FLUXCALERR/{key}"][()])
                for key in ("min", "mean", "std")
            ),
            "delta_time": tuple(
                float(handle[f"normalizations/delta_time/{key}"][()])
                for key in ("min", "mean", "std")
            ),
        }
    if all_features != ALL_FEATURES:
        raise ValueError("Database feature order does not match this backend version")
    selected_indices = [all_features.index(name) for name in selected_features]
    normalized_indices = [all_features.index(name) for name in NORMALIZED_FEATURES]
    arr_norm = []
    for name in NORMALIZED_FEATURES:
        prefix = (
            "FLUXCALERR"
            if name.startswith("FLUXCALERR_")
            else "FLUXCAL"
            if name.startswith("FLUXCAL_")
            else name
        )
        arr_norm.append(normalization[prefix])
    return SimpleNamespace(
        processed_dir=str(database_path.parent),
        hdf5_file_name=str(database_path),
        source_data="photometry",
        nb_classes=len(workflow.FINAL_CLASSES),
        data_fraction=1.0,
        training_features=list(selected_features),
        training_features_to_normalize=list(NORMALIZED_FEATURES),
        all_features=list(all_features),
        idx_features=selected_indices,
        idx_features_to_normalize=normalized_indices,
        idx_specz=[
            index for index, name in enumerate(selected_features) if "HOSTGAL_SPECZ" in name
        ],
        idx_flux=[
            index for index, name in enumerate(selected_features) if name.startswith("FLUXCAL_")
        ],
        idx_fluxerr=[
            index
            for index, name in enumerate(selected_features)
            if name.startswith("FLUXCALERR_")
        ],
        idx_delta_time=[
            index for index, name in enumerate(selected_features) if name == "delta_time"
        ],
        arr_norm=np.asarray(arr_norm, dtype=float),
        norm="global",
        random_length=config.random_length,
        random_redshift=False,
        redshift="zspe" if config.redshift_mode == "photometry_plus_truth_z" else "none",
        use_cuda=use_cuda,
        layer_type="lstm",
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        rnn_output_option=config.rnn_output_option,
        model="vanilla",
        no_dump=True,
    )


class HDF5SequenceStore:
    """Lazily normalize and read selected sequences from a backend database."""

    def __init__(self, database_path: str | Path, config: SuperNNovaTrainingConfig):
        """Open database metadata without retaining a writable file handle."""
        self.database_path = Path(database_path)
        self.config = config
        self.settings = build_supernnova_settings(self.database_path, config)
        with h5py.File(self.database_path, "r") as handle:
            self.ids = handle["SNID"][:].astype(str)
            self.targets = handle["target_7classes"][:].astype(int)
            self.split_codes = handle["dataset_photometry_7classes"][:].astype(int)
            self.t0 = handle["t0"][:].astype(float)
            self.database_identity = str(handle.attrs["database_identity"])

    def indices(self, role: str) -> np.ndarray:
        """Return database indices for a named partition role."""
        if role not in SPLIT_CODES:
            raise ValueError(f"Unknown sequence-store role: {role!r}")
        return np.flatnonzero(self.split_codes == SPLIT_CODES[role])

    def _normalize(self, sequence: np.ndarray) -> np.ndarray:
        """Apply training-derived SuperNNova normalization and feature selection."""
        normalized = sequence.copy()
        selected = normalized[:, self.settings.idx_features_to_normalize]
        minimum = self.settings.arr_norm[:, 0]
        mean = self.settings.arr_norm[:, 1]
        std = self.settings.arr_norm[:, 2]
        selected = np.clip(selected, minimum, np.inf)
        selected = (np.log(selected - minimum + 1e-5) - mean) / std
        normalized[:, self.settings.idx_features_to_normalize] = selected
        normalized = normalized[:, self.settings.idx_features]
        if not np.isfinite(normalized).all():
            raise ValueError("Normalized SuperNNova sequence contains non-finite values")
        return normalized.astype(np.float32, copy=False)

    def fetch(
        self,
        indices: Sequence[int],
        cutoff_days: float | None = None,
    ) -> tuple[list[tuple[np.ndarray, int, str]], int]:
        """Load normalized records and count objects ineligible at a phase cutoff."""
        records = []
        ineligible = 0
        with h5py.File(self.database_path, "r") as handle:
            n_features = int(handle["data"].attrs["n_features"])
            for raw_index in indices:
                index = int(raw_index)
                sequence = handle["data"][index].reshape(-1, n_features)
                if cutoff_days is not None:
                    grouped_mjd = handle["grouped_mjd"][index]
                    mask = grouped_mjd <= self.t0[index] + float(cutoff_days)
                    sequence = sequence[mask]
                if not len(sequence):
                    ineligible += 1
                    continue
                records.append(
                    (self._normalize(sequence), int(self.targets[index]), str(self.ids[index]))
                )
        return records, ineligible


def _resolve_device(config: SuperNNovaTrainingConfig) -> tuple[str, bool]:
    """Resolve auto/CPU/CUDA selection without silently falling back from CUDA."""
    import torch

    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
    available = torch.cuda.is_available()
    if config.device == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if available and config.device in {"auto", "cuda"} else "cpu"
    return device, device == "cuda"


def _model_settings(config: SuperNNovaTrainingConfig, use_cuda: bool) -> SimpleNamespace:
    """Build the subset of settings consumed by SuperNNova's VanillaRNN."""
    return SimpleNamespace(
        layer_type="lstm",
        nb_classes=len(workflow.FINAL_CLASSES),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        use_cuda=use_cuda,
        rnn_output_option=config.rnn_output_option,
    )


def supernnova_source_versions() -> dict[str, str]:
    """Return package versions plus provenance for the imported SuperNNova."""

    return {
        **workflow.source_versions(extra_packages=("supernnova", "h5py")),
        **workflow.module_source_provenance("supernnova"),
    }


def _probabilities_for_indices(
    model: Any,
    store: HDF5SequenceStore,
    indices: Sequence[int],
    batch_size: int,
    use_cuda: bool,
    cutoff_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Predict probabilities in database order using packed SuperNNova batches."""
    import torch
    from supernnova.utils import training_utils

    settings = store.settings
    settings.random_length = False
    settings.use_cuda = use_cuda
    ids: list[str] = []
    probabilities = []
    ineligible = 0
    model.eval()
    for start in range(0, len(indices), batch_size):
        records, missing = store.fetch(indices[start : start + batch_size], cutoff_days)
        ineligible += missing
        if not records:
            continue
        packed, _, _, reverse_sort = training_utils.get_data_batch(
            records, np.arange(len(records)), settings
        )
        with torch.no_grad():
            logits = model(packed)
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()[reverse_sort]
        probabilities.append(batch_probabilities)
        ids.extend(record[2] for record in records)
    matrix = (
        np.concatenate(probabilities, axis=0)
        if probabilities
        else np.empty((0, len(workflow.FINAL_CLASSES)))
    )
    return np.asarray(ids, dtype=str), matrix, ineligible


def _class_weight_tensor(targets: np.ndarray, device: str) -> Any:
    """Build inverse-frequency class weights in the fixed output-class order."""
    import torch

    counts = np.bincount(targets, minlength=len(workflow.FINAL_CLASSES)).astype(float)
    if (counts == 0).any():
        raise ValueError("Every final class must occur in SuperNNova training data")
    weights = len(targets) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    config: SuperNNovaTrainingConfig,
    store: HDF5SequenceStore,
    history: list[dict[str, Any]],
    epoch: int,
    best_loss: float,
    unimproved_epochs: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Create a complete resumable checkpoint payload."""
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.normalized(),
        "database_identity": store.database_identity,
        "feature_names": list(feature_names(config.redshift_mode)),
        "class_order": list(workflow.FINAL_CLASSES),
        "history": history,
        "epoch": int(epoch),
        "best_validation_log_loss": float(best_loss),
        "unimproved_epochs": int(unimproved_epochs),
        "elapsed_seconds": float(elapsed_seconds),
        "source_versions": supernnova_source_versions(),
    }


def train_supernnova(
    database_path: str | Path,
    output_dir: str | Path,
    config: SuperNNovaTrainingConfig,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train, resume, or reload one classifier with optional notebook progress."""
    import torch
    from supernnova.training.vanilla_rnn import VanillaRNN
    from supernnova.utils import training_utils

    device, use_cuda = _resolve_device(config)
    torch.set_num_threads(max(1, int(config.threads)))
    workflow.set_random_seed(config.seed)
    store = HDF5SequenceStore(database_path, config)
    train_indices = store.indices("train")
    validation_indices = store.indices("validation")
    if not len(validation_indices):
        raise ValueError("SuperNNova training requires a validation partition")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.json"
    complete_path = output_dir / "complete.json"
    expected_identity = workflow.configuration_hash(
        {**config.normalized(), "database_identity": store.database_identity}
    )
    if complete_path.exists() and best_path.exists() and history_path.exists() and not force:
        completion = json.loads(complete_path.read_text())
        if completion.get("training_identity") != expected_identity:
            raise FileExistsError(f"Completed run has incompatible configuration: {output_dir}")
        if show_progress:
            print(f"Using completed SuperNNova cache: {output_dir}")
        return json.loads(history_path.read_text())

    settings = _model_settings(config, use_cuda)
    model = VanillaRNN(len(feature_names(config.redshift_mode)), settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(
        weight=_class_weight_tensor(store.targets[train_indices], device)
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    unimproved_epochs = 0
    start_epoch = 0
    elapsed_before = 0.0

    if last_path.exists() and not force and not complete_path.exists():
        resumed = torch.load(last_path, map_location=device, weights_only=False)
        if resumed.get("database_identity") != store.database_identity:
            raise ValueError("Incomplete checkpoint belongs to another sequence database")
        if resumed.get("config") != config.normalized():
            raise ValueError("Incomplete checkpoint belongs to another training configuration")
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        history = list(resumed["history"])
        start_epoch = int(resumed["epoch"]) + 1
        best_loss = float(resumed["best_validation_log_loss"])
        unimproved_epochs = int(resumed["unimproved_epochs"])
        elapsed_before = float(resumed.get("elapsed_seconds", 0.0))

    started = time.perf_counter()
    settings_for_batches = build_supernnova_settings(database_path, config, use_cuda=use_cuda)
    rng = np.random.default_rng(config.seed + start_epoch)
    epoch_iterator: Any = range(start_epoch, config.epochs)
    epoch_progress = None
    progress_factory = None
    if show_progress:
        # The text progress bar works in Jupyter even when widget extensions are absent.
        from tqdm import tqdm

        progress_factory = tqdm
        resume_note = f", resuming at epoch {start_epoch + 1}" if start_epoch else ""
        print(
            f"Training {config.redshift_mode} on {device}: "
            f"{len(train_indices):,} train, {len(validation_indices):,} validation, "
            f"{config.epochs} epoch(s){resume_note}"
        )
        epoch_progress = tqdm(
            epoch_iterator,
            total=config.epochs - start_epoch,
            desc=f"{config.redshift_mode} epochs",
            unit="epoch",
            dynamic_ncols=True,
        )
        epoch_iterator = epoch_progress

    for epoch in epoch_iterator:
        model.train()
        shuffled = rng.permutation(train_indices)
        batch_losses = []
        batch_starts = range(0, len(shuffled), config.batch_size)
        batch_iterator: Any = batch_starts
        batch_progress = None
        if progress_factory is not None:
            # The nested bar makes slow CPU epochs visibly advance batch by batch.
            batch_progress = progress_factory(
                batch_starts,
                total=len(batch_starts),
                desc=f"Epoch {epoch + 1}/{config.epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            batch_iterator = batch_progress
        for start in batch_iterator:
            records, _ = store.fetch(shuffled[start : start + config.batch_size])
            settings_for_batches.random_length = config.random_length
            packed, _, targets, _ = training_utils.get_data_batch(
                records, np.arange(len(records)), settings_for_batches
            )
            optimizer.zero_grad()
            logits = model(packed)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            if batch_progress is not None:
                batch_progress.set_postfix(
                    weighted_loss=f"{batch_losses[-1]:.4f}",
                    refresh=False,
                )

        _, validation_probabilities, _ = _probabilities_for_indices(
            model,
            store,
            validation_indices,
            config.batch_size,
            use_cuda,
        )
        validation_labels = np.asarray(
            [workflow.FINAL_CLASSES[index] for index in store.targets[validation_indices]]
        )
        validation_loss = workflow.class_balanced_log_loss(
            validation_labels, validation_probabilities
        )
        row = {
            "epoch": epoch + 1,
            "train_weighted_loss": float(np.mean(batch_losses)),
            "validation_class_balanced_log_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if epoch_progress is not None and progress_factory is not None:
            epoch_progress.set_postfix(
                train=f"{row['train_weighted_loss']:.4f}",
                validation=f"{validation_loss:.4f}",
                lr=f"{row['learning_rate']:.2g}",
                refresh=False,
            )
            progress_factory.write(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"train loss {row['train_weighted_loss']:.4f}, "
                f"validation loss {validation_loss:.4f}, "
                f"lr {row['learning_rate']:.2g}"
            )
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            unimproved_epochs = 0
        else:
            unimproved_epochs += 1
            if unimproved_epochs % config.lr_patience == 0:
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] *= config.lr_factor

        elapsed = elapsed_before + time.perf_counter() - started
        payload = _checkpoint_payload(
            model,
            optimizer,
            config,
            store,
            history,
            epoch,
            best_loss,
            unimproved_epochs,
            elapsed,
        )
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
        workflow.write_json_once(
            {"elapsed_seconds": elapsed, "epochs": history},
            history_path,
            overwrite=True,
        )
        if unimproved_epochs >= config.early_stopping_patience:
            if show_progress:
                print(
                    f"Early stopping after epoch {epoch + 1}; "
                    f"best validation loss {best_loss:.4f}."
                )
            break

    if not best_path.exists():
        raise RuntimeError("SuperNNova training produced no best checkpoint")
    elapsed = elapsed_before + time.perf_counter() - started
    result = {"elapsed_seconds": elapsed, "epochs": history}
    workflow.write_json_once(result, history_path, overwrite=True)
    workflow.write_json_once(
        {
            "status": "complete",
            "training_identity": expected_identity,
            "best_validation_log_loss": best_loss,
            "completed_epochs": len(history),
            "device": device,
        },
        complete_path,
        overwrite=True,
    )
    if show_progress:
        print(
            f"Finished {config.redshift_mode}: {len(history)} epoch(s), "
            f"{elapsed / 60:.2f} min, best validation loss {best_loss:.4f}"
        )
    return result


def load_supernnova_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[Any, SuperNNovaTrainingConfig]:
    """Reconstruct a VanillaRNN and load a saved backend checkpoint."""
    import torch
    from supernnova.training.vanilla_rnn import VanillaRNN

    requested = device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint loading was requested but CUDA is unavailable")
    payload = torch.load(checkpoint_path, map_location=requested, weights_only=False)
    config = SuperNNovaTrainingConfig(**payload["config"])
    settings = _model_settings(config, requested == "cuda")
    model = VanillaRNN(len(feature_names(config.redshift_mode)), settings).to(requested)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def predict_supernnova(
    checkpoint_path: str | Path,
    database_path: str | Path,
    role: str,
    cutoff_days: float | None = None,
    device: str = "auto",
) -> PredictionResult:
    """Predict one database role with complete or peak-relative partial sequences."""
    from astropy.table import Table

    model, config = load_supernnova_checkpoint(checkpoint_path, device=device)
    resolved_device, use_cuda = _resolve_device(
        SuperNNovaTrainingConfig(**{**config.normalized(), "device": device})
    )
    model = model.to(resolved_device)
    store = HDF5SequenceStore(database_path, config)
    indices = store.indices(role)
    ids, probabilities, ineligible = _probabilities_for_indices(
        model,
        store,
        indices,
        config.batch_size,
        use_cuda,
        cutoff_days=cutoff_days,
    )
    if not np.isfinite(probabilities).all():
        raise ValueError("SuperNNova produced non-finite probabilities")
    if len(probabilities) and not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("SuperNNova probabilities do not sum to one")
    classifications = Table({"object_id": ids})
    for index, label in enumerate(workflow.FINAL_CLASSES):
        classifications[label] = probabilities[:, index]
    return PredictionResult(
        classifications=classifications,
        eligible_objects=len(ids),
        ineligible_objects=ineligible,
        cutoff_days=cutoff_days,
    )


def paired_group_bootstrap_differences(
    first_predictions: pd.DataFrame,
    second_predictions: pd.DataFrame,
    split_manifest: pd.DataFrame,
    repeats: int = 1000,
    seed: int = workflow.DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Estimate paired second-minus-first metric differences by split group."""
    probability_columns = [f"prob_{label}" for label in workflow.FINAL_CLASSES]
    selected_columns = ["object_id", "true_class", "predicted_class", *probability_columns]
    first = first_predictions[selected_columns].copy()
    second = second_predictions[selected_columns].copy()
    common = sorted(set(first["object_id"]) & set(second["object_id"]))
    if not common:
        raise ValueError("Paired comparison has no common predicted objects")
    first = first.set_index("object_id").loc[common].reset_index()
    second = second.set_index("object_id").loc[common].reset_index()
    if not np.array_equal(first["true_class"], second["true_class"]):
        raise ValueError("Paired predictions disagree on true classes")
    groups = split_manifest.set_index("object_id").loc[common, "group_id"].astype(str)
    unique_groups = groups.unique()
    group_positions = {
        group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups
    }
    metric_names = (
        "class_balanced_log_loss",
        "balanced_accuracy",
        "macro_f1",
        "top_2_accuracy",
        "multiclass_brier",
    )
    samples = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        positions = np.concatenate([group_positions[group] for group in drawn])
        first_metrics = workflow.compute_classification_metrics(first.iloc[positions])
        second_metrics = workflow.compute_classification_metrics(second.iloc[positions])
        for name in metric_names:
            samples[name].append(second_metrics[name] - first_metrics[name])
    return {
        "direction": "second_minus_first",
        "common_objects": len(common),
        "intervals": {
            name: {
                "lower_95": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
            for name, values in samples.items()
        },
    }
