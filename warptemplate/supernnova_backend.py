"""Prepare, train, and evaluate the Warp recurrent light-curve classifier.

This module implements the complete data and experiment path around
:class:`warptemplate.sequence_rnn.WarpSequenceRNN`.  It follows SuperNNova's useful
sequence conventions—grouped epochs, log normalization, packed variable-length
batches, and random light-curve prefixes—while keeping Warp's seven-class taxonomy,
group-safe splits, artifact layout, and evaluation metrics.

Reading the file from top to bottom mirrors the actual workflow:

1. :func:`build_supernnova_sequences` turns long-form observations into one
   ``(epochs, features)`` matrix per object.
2. :func:`prepare_supernnova_database` streams these matrices into an HDF5 cache and
   learns normalization constants from the training role only.
3. :class:`HDF5SequenceStore` reads and normalizes records lazily.
4. :func:`_make_packed_batch` collates differently sized light curves for the LSTM.
5. :func:`train_supernnova` optimizes the local recurrent model, selects a checkpoint
   on validation loss, and optionally calibrates its probabilities.
6. The prediction functions evaluate full or time-truncated light curves and can
   average multiple independently seeded models.

Important distinction: the HDF5 file is a preprocessed *dataset cache*, not a trained
model.  Learned neural-network weights and optimizer state are stored separately in
``.pt`` checkpoint files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from . import classification as workflow


# -----------------------------------------------------------------------------
# Fixed feature and storage contracts
# -----------------------------------------------------------------------------

SUPERNOVA_ZEROPOINT = 27.5
DATABASE_SCHEMA_VERSION = 3
FLUX_NORMALIZATION_FLOOR = -2000.0
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
REDSHIFT_MODES = ("photometry_only", "photometry_plus_truth_z")
FLUX_FEATURES = tuple(f"FLUXCAL_{band}" for band in workflow.EXPECTED_BANDS)
FLUXERR_FEATURES = tuple(f"FLUXCALERR_{band}" for band in workflow.EXPECTED_BANDS)
PRESENCE_FEATURES = tuple(f"PRESENT_{band}" for band in workflow.EXPECTED_BANDS)
SNR_FEATURES = tuple(f"ASINH_SNR_{band}" for band in workflow.EXPECTED_BANDS)
DETECTION_FEATURES = tuple(f"DETECTED_5SIGMA_{band}" for band in workflow.EXPECTED_BANDS)
LIMITING_MAG_FEATURES = tuple(
    f"LIMITING_MAG_5SIGMA_SCALED_{band}" for band in workflow.EXPECTED_BANDS
)
BASE_PHOTOMETRY_FEATURES = (
    *FLUX_FEATURES,
    *FLUXERR_FEATURES,
    *PRESENCE_FEATURES,
    "delta_time",
)
ENGINEERED_FEATURES = (
    *SNR_FEATURES,
    *DETECTION_FEATURES,
    *LIMITING_MAG_FEATURES,
    "log1p_time_since_first",
    "observed_band_fraction",
)
PHOTOMETRY_FEATURES = (*BASE_PHOTOMETRY_FEATURES, *ENGINEERED_FEATURES)
REDSHIFT_FEATURES = ("HOSTGAL_SPECZ",)
ALL_FEATURES = PHOTOMETRY_FEATURES
NORMALIZED_FEATURES = (*FLUX_FEATURES, *FLUXERR_FEATURES, "delta_time")
POOLING_OPTIONS = ("mean", "standard", "attention")


@dataclass(frozen=True)
class SuperNNovaTrainingConfig:
    """Configure one reproducible recurrent-classifier training run.

    The fields fall into four groups:

    - model capacity: ``hidden_dim``, ``num_layers``, ``bidirectional``, ``dropout``;
    - sequence representation: ``rnn_output_option``, ``engineered_features``, and
      ``redshift_mode``;
    - optimization and stopping: epochs, batch size, learning rate, scheduler
      settings, minimum improvement, gradient clipping, and calibration; and
    - execution: random seed, device selection, and CPU thread count.

    ``random_length`` is a training augmentation: on each visit, an eligible light
    curve may be shortened to a random prefix of at least three epochs.  It teaches
    the model to classify partial as well as complete observations.  Validation and
    prediction always use deterministic lengths.

    ``redshift_mode="photometry_plus_truth_z"`` treats redshift as one object-level
    auxiliary variable fused after the LSTM.  It is not a per-epoch measurement.
    """

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
    minimum_improvement: float = 1e-4
    gradient_clip_norm: float = 5.0
    calibrate_probabilities: bool = True
    engineered_features: bool = False
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
        if self.rnn_output_option not in POOLING_OPTIONS:
            raise ValueError(f"Unsupported RNN pooling: {self.rnn_output_option!r}")
        if self.minimum_improvement < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("Improvement threshold must be non-negative and clipping positive")

    def normalized(self) -> dict[str, Any]:
        """Return a stable JSON-compatible configuration mapping."""
        return asdict(self)


@dataclass
class PredictionResult:
    """Bundle predictions with bookkeeping for a possible time cutoff.

    ``classifications`` uses the Astropy-table format expected by the shared
    prediction standardizer.  An object is ineligible when a requested cutoff leaves
    it with no observed epoch; such objects are counted instead of receiving invented
    probabilities.
    """

    classifications: Any
    eligible_objects: int
    ineligible_objects: int
    cutoff_days: float | None
    cutoff_reference: str | None


def feature_names(
    redshift_mode: str,
    engineered_features: bool = False,
) -> tuple[str, ...]:
    """Return the exact model-input feature order for one experiment variant.

    Feature order is part of the checkpoint contract.  The default uses measured
    flux, uncertainty, band-presence masks, and time gaps.  Engineered S/N, detection,
    limiting-magnitude, and coverage features are opt-in.  Truth redshift, when used,
    appears last so batching can separate it for late fusion.
    """
    if redshift_mode not in REDSHIFT_MODES:
        raise ValueError(f"Unsupported redshift mode: {redshift_mode!r}")
    selected = PHOTOMETRY_FEATURES if engineered_features else BASE_PHOTOMETRY_FEATURES
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
    """Assign database roles for either a full run or a balanced smoke experiment.

    Full modes preserve the persistent train/validation/test split.  Smoke mode draws
    small class-balanced samples from otherwise training-designated folds and assigns
    temporary roles.  The original persistent test and validation folds therefore
    remain untouched, and a final group-overlap check guards the temporary roles.
    """
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
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot one object's grouped long table into an epoch-by-feature matrix.

    Every row of the returned matrix is an observing epoch and every passband owns
    separate flux, uncertainty, and presence columns.  Structural zeros for bands not
    observed in an epoch are distinguishable through the presence mask.  Additional
    engineered columns are computed here even if a particular run later opts out;
    this permits one database cache to support both feature configurations.

    Returns the feature matrix and its absolute grouped MJDs.  Absolute times are kept
    outside the model features for scientifically explicit partial-curve cutoffs.
    """
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

    presence = sequence[:, [ALL_FEATURES.index(name) for name in PRESENCE_FEATURES]]
    flux = sequence[:, [ALL_FEATURES.index(name) for name in FLUX_FEATURES]]
    flux_error = sequence[:, [ALL_FEATURES.index(name) for name in FLUXERR_FEATURES]]
    observed = presence.astype(bool)
    snr = np.zeros_like(flux)
    snr[observed] = flux[observed] / flux_error[observed]
    # The asinh transform remains signed, is nearly linear near zero, and prevents a
    # few extreme S/N values from controlling optimization.  Five is the detection
    # threshold and therefore also a natural dimensionless scale.
    sequence[:, [ALL_FEATURES.index(name) for name in SNR_FEATURES]] = np.arcsinh(
        snr / 5.0
    )
    sequence[:, [ALL_FEATURES.index(name) for name in DETECTION_FEATURES]] = (
        observed & (snr >= 5.0)
    )
    limiting_magnitude = np.zeros_like(flux)
    limiting_magnitude[observed] = SUPERNOVA_ZEROPOINT - 2.5 * np.log10(
        5.0 * flux_error[observed]
    )
    # Centering on 25 mag and scaling by 5 mag gives an order-unity input while the
    # presence flag disambiguates the structural zero used for an unobserved band.
    limiting_magnitude[observed] = (limiting_magnitude[observed] - 25.0) / 5.0
    sequence[:, [ALL_FEATURES.index(name) for name in LIMITING_MAG_FEATURES]] = (
        limiting_magnitude
    )

    # The first delta is zero by definition; later values measure grouped-epoch gaps.
    sequence[:, PHOTOMETRY_FEATURES.index("delta_time")] = np.concatenate(
        ([0.0], np.diff(grouped_times))
    )
    sequence[:, ALL_FEATURES.index("log1p_time_since_first")] = np.log1p(
        grouped_times - grouped_times[0]
    )
    sequence[:, ALL_FEATURES.index("observed_band_fraction")] = presence.mean(axis=1)
    if not np.isfinite(sequence).all():
        raise ValueError("SuperNNova sequence contains non-finite values")
    if (sequence[:, PHOTOMETRY_FEATURES.index("delta_time")] < 0).any():
        raise ValueError("SuperNNova sequence contains negative delta times")
    return sequence, grouped_times.astype(np.float64)


def build_supernnova_sequences(
    truth: pd.DataFrame,
    observations: pd.DataFrame,
    object_ids: Sequence[str] | None = None,
    epoch_window_days: float = workflow.DEFAULT_EPOCH_WINDOW_DAYS,
    duplicate_strategy: str = "inverse_variance",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert Warp light curves into validated variable-length feature sequences.

    Photometry is first expressed at :data:`SUPERNOVA_ZEROPOINT`, then nearby visits
    are grouped into epochs.  Repeated measurements of the same band and epoch are
    combined according to ``duplicate_strategy``.  The result maps each requested
    object ID to ``(feature_matrix, grouped_mjd)`` in requested order.

    The feature matrix initially contains :data:`ALL_FEATURES`; model configuration
    selects the actual subset later.  Redshift is validated here as object metadata
    but is not inserted into the temporal matrix by this function.
    """
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
    grouped = workflow.group_observing_epochs(
        converted,
        window_days=epoch_window_days,
        duplicate_strategy=duplicate_strategy,
    )
    redshifts = selected_truth.set_index("object_id")["z"].astype(float)
    sequences: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for object_id, rows in grouped.groupby("object_id", sort=False):
        object_id = str(object_id)
        if object_id not in redshifts.index:
            raise ValueError(f"Observations have no matching truth row: {object_id}")
        if not np.isfinite(redshifts.loc[object_id]):
            raise ValueError(f"Object has non-finite redshift truth: {object_id}")
        sequences[object_id] = _sequence_from_grouped_rows(rows)
    if object_ids is not None:
        requested_order = list(map(str, object_ids))
        missing_ids = [object_id for object_id in requested_order if object_id not in sequences]
        if missing_ids:
            raise ValueError(f"Selected objects lack observations: {missing_ids[:5]}")
        sequences = {object_id: sequences[object_id] for object_id in requested_order}
    return sequences


# -----------------------------------------------------------------------------
# HDF5 sequence database and training-only normalization
# -----------------------------------------------------------------------------


def _create_database_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    """Create the resizable HDF5 schema for sequences and object metadata.

    Variable-length matrices are flattened on disk and reshaped using the stored
    feature count.  This avoids padding the entire database to its longest light curve.
    ``split`` contains compact integer role codes, whereas ``role`` remains readable.
    """
    string_dtype = h5py.string_dtype(encoding="utf-8")
    float_array_dtype = h5py.vlen_dtype(np.dtype("float32"))
    time_array_dtype = h5py.vlen_dtype(np.dtype("float64"))
    datasets = {
        "data": handle.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=float_array_dtype
        ),
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
    """Append one flattened light curve plus its target and provenance metadata."""
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
    """Fit log-normalization constants using measured values in training objects.

    Validation and test data are excluded to prevent preprocessing leakage.  Missing
    band slots are also excluded: their zeros mean "not observed", not zero physical
    flux.  For each normalized group this stores ``(minimum, log_mean, log_std)`` for
    the transformation ``(log(clip(x, min) - min + eps) - mean) / std``.

    The two-pass streaming calculation bounds memory use: the first pass obtains the
    log shift and the second accumulates moments.  Flux/error bands share global
    constants so their relative scale remains comparable across filters.
    """
    split_codes = handle["dataset_photometry_7classes"][:]
    train_indices = np.flatnonzero(split_codes == SPLIT_CODES["train"])
    if not len(train_indices):
        raise ValueError("Cannot normalize a SuperNNova database without training objects")
    n_features = int(handle["data"].attrs["n_features"])
    delta_index = ALL_FEATURES.index("delta_time")
    presence_slice = slice(
        len(FLUX_FEATURES) + len(FLUXERR_FEATURES),
        len(FLUX_FEATURES) + len(FLUXERR_FEATURES) + len(PRESENCE_FEATURES),
    )

    # A first pass obtains the shift required by SuperNNova's logarithm without
    # materializing every training time step in memory.
    minima = {"FLUXCAL": np.inf, "FLUXCALERR": np.inf, "delta_time": np.inf}
    for index in train_indices:
        sequence = handle["data"][index].reshape(-1, n_features)
        observed = sequence[:, presence_slice].astype(bool)
        minima["FLUXCAL"] = min(
            minima["FLUXCAL"], float(sequence[:, : len(FLUX_FEATURES)][observed].min())
        )
        minima["FLUXCALERR"] = min(
            minima["FLUXCALERR"],
            float(sequence[:, len(FLUX_FEATURES) : 2 * len(FLUX_FEATURES)][observed].min()),
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
        observed = sequence[:, presence_slice].astype(bool)
        values_by_group = {
            "FLUXCAL": sequence[:, : len(FLUX_FEATURES)][observed],
            "FLUXCALERR": sequence[
                :, len(FLUX_FEATURES) : 2 * len(FLUX_FEATURES)
            ][observed],
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


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash one source artifact without loading it fully into memory."""
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _database_identity(
    sample_dir: str | Path,
    truth: pd.DataFrame,
    role_manifest: pd.DataFrame,
    minimum_training_epochs: int,
    epoch_window_days: float,
    duplicate_strategy: str,
) -> str:
    """Hash every input that can change the scientific content of the HDF5 cache.

    The identity covers selected objects and roles, relevant truth values, observation
    file contents, preprocessing settings, feature schema, and the source of this
    adapter.  A stale cache is rejected even when its path happens to be unchanged.
    """
    columns = ["object_id", "role", "final_label", "group_id"]
    ordered = role_manifest[columns].astype(str).sort_values("object_id")
    selected_ids = set(ordered["object_id"])
    truth_columns = [name for name in ("object_id", "z", "t0") if name in truth]
    selected_truth = truth[truth["object_id"].astype(str).isin(selected_ids)][truth_columns]
    selected_truth = selected_truth.sort_values("object_id")
    digest = sha256()
    digest.update(ordered.to_csv(index=False).encode())
    digest.update(selected_truth.to_csv(index=False).encode())
    for path in workflow.sample_table_files(sample_dir, "observations"):
        digest.update(str(path.relative_to(Path(sample_dir))).encode())
        digest.update(_file_sha256(path).encode())
    preprocessing = {
        "sequence_features": list(ALL_FEATURES),
        "context_features": list(REDSHIFT_FEATURES),
        "schema": DATABASE_SCHEMA_VERSION,
        "zeropoint": SUPERNOVA_ZEROPOINT,
        "minimum_epochs": minimum_training_epochs,
        "epoch_window_days": epoch_window_days,
        "duplicate_strategy": duplicate_strategy,
        "flux_floor": FLUX_NORMALIZATION_FLOOR,
    }
    digest.update(json.dumps(preprocessing, sort_keys=True).encode())
    # Code fingerprints prevent stale reuse when behavior changes without a schema bump.
    for path in (Path(__file__), Path(workflow.__file__)):
        digest.update(_file_sha256(path).encode())
    return digest.hexdigest()[:16]


def prepare_supernnova_database(
    sample_dir: str | Path,
    truth: pd.DataFrame,
    role_manifest: pd.DataFrame,
    output_path: str | Path,
    minimum_training_epochs: int = 3,
    epoch_window_days: float = workflow.DEFAULT_EPOCH_WINDOW_DAYS,
    duplicate_strategy: str = "inverse_variance",
) -> dict[str, Any]:
    """Create one new HDF5 sequence database for a role manifest.

    Observation Parquet files are filtered and processed one fragment at a time.  An
    object is written only after validation and is rejected if it crosses fragments.
    Curves shorter than ``minimum_training_epochs`` are recorded as exclusions.
    Normalization is derived after all records are present, from the training role
    only.

    The database is first written to a sibling temporary path and then atomically
    renamed.  Readers therefore see either no database or the new complete database,
    never a partially written HDF5 file.  Existing output or temporary files are
    refused; loading an existing database is the separate, explicit
    :func:`load_supernnova_database_summary` operation.

    Returns a JSON-compatible summary of object counts, feature order, preprocessing,
    and normalization constants.
    """
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
    identity = _database_identity(
        sample_dir,
        truth,
        role_manifest,
        minimum_training_epochs,
        epoch_window_days,
        duplicate_strategy,
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace sequence database: {output_path}")
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
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Remove interrupted database temporary file: {temporary_path}")
    # Write to a temporary sibling so os.replace remains atomic on the same filesystem.
    with h5py.File(temporary_path, "w") as handle:
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
            sequences = build_supernnova_sequences(
                file_truth,
                frame,
                epoch_window_days=epoch_window_days,
                duplicate_strategy=duplicate_strategy,
            )
            for object_id, (sequence, grouped_mjd) in sequences.items():
                if object_id in written:
                    raise ValueError(f"Object crosses observation files: {object_id}")
                if len(sequence) < minimum_training_epochs:
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
        handle.attrs["epoch_window_days"] = epoch_window_days
        handle.attrs["duplicate_strategy"] = duplicate_strategy
        handle.attrs["class_order_json"] = json.dumps(list(workflow.FINAL_CLASSES))
        summary = {
            "database_identity": identity,
            "database_schema_version": DATABASE_SCHEMA_VERSION,
            "objects": len(written),
            "skipped_short_objects": len(skipped_short),
            "role_counts": {
                role: int((handle["dataset_photometry_7classes"][:] == code).sum())
                for role, code in SPLIT_CODES.items()
            },
            "normalization": {
                key: dict(zip(("min", "mean", "std"), values))
                for key, values in normalization.items()
            },
            "feature_order": list(ALL_FEATURES),
            "context_feature_order": list(REDSHIFT_FEATURES),
            "zeropoint": SUPERNOVA_ZEROPOINT,
            "epoch_window_days": epoch_window_days,
            "duplicate_strategy": duplicate_strategy,
        }
        handle.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        handle.flush()
    os.replace(temporary_path, output_path)
    return summary


def load_supernnova_database_summary(database_path: str | Path) -> dict[str, Any]:
    """Load the persisted summary from one existing sequence database.

    No alternate location is searched and no database is generated.  HDF5 therefore
    raises its normal error when the configured file is absent or unreadable.
    """
    with h5py.File(database_path, "r") as handle:
        if int(handle.attrs["database_schema_version"]) != DATABASE_SCHEMA_VERSION:
            raise ValueError("Sequence database schema does not match this backend")
        if tuple(handle["features"][:].astype(str)) != ALL_FEATURES:
            raise ValueError("Sequence database feature order does not match this backend")
        return json.loads(str(handle.attrs["summary_json"]))


def build_supernnova_settings(
    database_path: str | Path,
    config: SuperNNovaTrainingConfig,
    use_cuda: bool = False,
) -> SimpleNamespace:
    """Translate the typed run config and HDF5 metadata into loader settings.

    The namespace records feature indices and normalization parameters once, avoiding
    repeated string lookups for every sequence.  It deliberately selects only the
    configured temporal features; object-level redshift is appended by the store and
    then separated for late fusion by the batch collator.
    """
    database_path = Path(database_path)
    selected_features = (
        PHOTOMETRY_FEATURES if config.engineered_features else BASE_PHOTOMETRY_FEATURES
    )
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
        idx_specz=[],
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
        normalization_presence_indices=[
            all_features.index(PRESENCE_FEATURES[index % len(PRESENCE_FEATURES)])
            if index < len(FLUX_FEATURES) + len(FLUXERR_FEATURES)
            else -1
            for index in range(len(NORMALIZED_FEATURES))
        ],
        arr_norm=np.asarray(arr_norm, dtype=float),
        norm="global",
        random_length=config.random_length,
        random_redshift=False,
        redshift="none",
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
    """Provide lazy, process-safe access to normalized sequence records.

    Small fixed metadata arrays are loaded once at construction.  Variable-length
    light curves remain on disk and are read only when requested by a training batch
    or prediction call.  The HDF5 handle is opened lazily per process, which allows a
    store to be pickled for data-loader workers without sharing an unsafe file handle.
    """

    def __init__(self, database_path: str | Path, config: SuperNNovaTrainingConfig):
        """Open database metadata and initialize a process-local lazy reader."""
        self.database_path = Path(database_path)
        self.config = config
        self.settings = build_supernnova_settings(self.database_path, config)
        with h5py.File(self.database_path, "r") as handle:
            self.ids = handle["SNID"][:].astype(str)
            self.targets = handle["target_7classes"][:].astype(int)
            self.split_codes = handle["dataset_photometry_7classes"][:].astype(int)
            self.t0 = handle["t0"][:].astype(float)
            self.redshift = handle["z"][:].astype(float)
            self.database_identity = str(handle.attrs["database_identity"])
        self._handle: h5py.File | None = None
        self._handle_pid: int | None = None

    def _reader(self) -> h5py.File:
        """Return one reusable read-only HDF5 handle for the current process."""
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            self.close()
            self._handle = h5py.File(self.database_path, "r")
            self._handle_pid = pid
        return self._handle

    def close(self) -> None:
        """Close the process-local HDF5 handle if it is open."""
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
        self._handle = None
        self._handle_pid = None

    def __del__(self) -> None:
        """Release the read handle during normal object cleanup."""
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Drop non-picklable handles when workers receive the sequence store."""
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def indices(self, role: str) -> np.ndarray:
        """Return database indices for a named partition role."""
        if role not in SPLIT_CODES:
            raise ValueError(f"Unknown sequence-store role: {role!r}")
        return np.flatnonzero(self.split_codes == SPLIT_CODES[role])

    def _normalize(self, sequence: np.ndarray, redshift: float) -> np.ndarray:
        """Apply training-fitted scaling and select the configured input features.

        Missing flux/error slots are reset to zero *after* normalization.  They are
        structural padding within an epoch, while corresponding presence flags tell
        the model why the value is zero.  In the redshift variant, one constant column
        is appended temporarily so the collator can extract it as auxiliary context.
        """
        normalized = sequence.copy()
        selected = normalized[:, self.settings.idx_features_to_normalize]
        minimum = self.settings.arr_norm[:, 0]
        mean = self.settings.arr_norm[:, 1]
        std = self.settings.arr_norm[:, 2]
        selected = np.clip(selected, minimum, np.inf)
        selected = (np.log(selected - minimum + 1e-5) - mean) / std
        presence = normalized[:, [ALL_FEATURES.index(name) for name in PRESENCE_FEATURES]]
        # Missing flux/error entries are structural padding, not measurements.  Zero is
        # the neutral value after standardization and presence flags retain missingness.
        selected[:, : len(FLUX_FEATURES)][presence == 0] = 0.0
        selected[:, len(FLUX_FEATURES) : 2 * len(FLUX_FEATURES)][presence == 0] = 0.0
        normalized[:, self.settings.idx_features_to_normalize] = selected
        normalized = normalized[:, self.settings.idx_features]
        if self.config.redshift_mode == "photometry_plus_truth_z":
            redshift_column = np.full((len(normalized), 1), redshift, dtype=np.float32)
            normalized = np.concatenate((normalized, redshift_column), axis=1)
        if not np.isfinite(normalized).all():
            raise ValueError("Normalized SuperNNova sequence contains non-finite values")
        return normalized.astype(np.float32, copy=False)

    def fetch(
        self,
        indices: Sequence[int],
        cutoff_days: float | None = None,
        cutoff_reference: str = "truth_t0",
    ) -> tuple[list[tuple[np.ndarray, int, str]], int]:
        """Load selected records, optionally truncating them at an absolute cutoff.

        ``cutoff_reference="truth_t0"`` interprets ``cutoff_days`` as phase relative
        to simulated peak time.  ``"first_observation"`` measures elapsed observing
        time and does not require peak knowledge.  An object with no retained epoch is
        omitted and counted as ineligible; no empty tensor is sent to the LSTM.

        Each returned record is ``(sequence, integer_target, object_id)``.
        """
        if cutoff_reference not in {"truth_t0", "first_observation"}:
            raise ValueError(f"Unsupported cutoff reference: {cutoff_reference!r}")
        records = []
        ineligible = 0
        handle = self._reader()
        n_features = int(handle["data"].attrs["n_features"])
        for raw_index in indices:
            index = int(raw_index)
            sequence = handle["data"][index].reshape(-1, n_features)
            if cutoff_days is not None:
                grouped_mjd = handle["grouped_mjd"][index]
                reference_mjd = (
                    self.t0[index]
                    if cutoff_reference == "truth_t0"
                    else float(grouped_mjd[0])
                )
                mask = grouped_mjd <= reference_mjd + float(cutoff_days)
                sequence = sequence[mask]
            if not len(sequence):
                ineligible += 1
                continue
            records.append(
                (
                    self._normalize(sequence, self.redshift[index]),
                    int(self.targets[index]),
                    str(self.ids[index]),
                )
            )
        return records, ineligible


class SequenceDataset:
    """Expose one train/validation/test role with a dataset-style interface.

    The class intentionally holds stable database indices rather than materialized
    arrays.  Accessing an item delegates normalization and I/O to the shared store.
    """

    def __init__(self, store: HDF5SequenceStore, role: str):
        """Bind the dataset to the stable database indices of one role."""
        self.store = store
        self.database_indices = store.indices(role)

    def __len__(self) -> int:
        """Return the number of eligible database records in this role."""
        return len(self.database_indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, str]:
        """Load one normalized record by role-relative index."""
        records, ineligible = self.store.fetch([self.database_indices[index]])
        if ineligible or not records:
            raise IndexError("Database role unexpectedly contains an empty sequence")
        return records[0]


def _resolve_device(config: SuperNNovaTrainingConfig) -> tuple[str, bool]:
    """Resolve ``auto``/CPU/CUDA selection without hiding an unavailable request."""
    import torch

    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
    available = torch.cuda.is_available()
    if config.device == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if available and config.device in {"auto", "cuda"} else "cpu"
    return device, device == "cuda"


def _model_settings(config: SuperNNovaTrainingConfig, use_cuda: bool) -> SimpleNamespace:
    """Build the architecture namespace consumed by :class:`WarpSequenceRNN`."""
    return SimpleNamespace(
        layer_type="lstm",
        nb_classes=len(workflow.FINAL_CLASSES),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        use_cuda=use_cuda,
        rnn_output_option=config.rnn_output_option,
        auxiliary_dim=1 if config.redshift_mode == "photometry_plus_truth_z" else 0,
    )


def _make_packed_batch(
    records: Sequence[tuple[np.ndarray, int, str]],
    config: SuperNNovaTrainingConfig,
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[Any, Any | None, Any, np.ndarray]:
    """Collate variable-length records into a length-sorted PyTorch packed batch.

    Packed sequences avoid treating padding as data, but PyTorch requires descending
    lengths.  This function therefore sorts sequences and targets together and returns
    ``reverse_sort`` so predictions can be restored to caller order.

    During training, a supplied ``rng`` may shorten each light curve to a random prefix
    of at least three epochs.  With truth redshift enabled, the repeated last column is
    removed from the temporal sequence and returned once per object as ``auxiliary``.

    Returns
    -------
    packed, auxiliary, targets, reverse_sort:
        A packed temporal batch, optional ``(batch, 1)`` late-fusion tensor, integer
        class targets in sorted order, and the permutation that undoes length sorting.
    """
    import torch

    if not records:
        raise ValueError("Cannot collate an empty SuperNNova batch")
    sequences = []
    auxiliaries = []
    lengths = []
    targets = []
    use_redshift = config.redshift_mode == "photometry_plus_truth_z"
    for sequence, target, _ in records:
        selected = sequence
        if rng is not None and config.random_length and len(selected) > 3:
            selected = selected[: int(rng.integers(3, len(selected) + 1))]
        if use_redshift:
            # Redshift is object-level context.  Feeding the repeated scalar through
            # every recurrent step would give longer light curves disproportionate weight.
            auxiliaries.append(float(selected[0, -1]))
            selected = selected[:, :-1]
        sequences.append(np.asarray(selected, dtype=np.float32))
        lengths.append(len(selected))
        targets.append(int(target))

    sort_order = np.argsort(lengths)[::-1].copy()
    reverse_sort = np.argsort(sort_order)
    sorted_lengths = [lengths[index] for index in sort_order]
    max_length = max(sorted_lengths)
    feature_count = sequences[0].shape[1]
    padded = torch.zeros((max_length, len(records), feature_count), dtype=torch.float32)
    # Padding is only an intermediate representation; pack_padded_sequence removes it
    # from recurrent computation using the true sorted lengths below.
    for batch_index, record_index in enumerate(sort_order):
        values = torch.from_numpy(sequences[record_index])
        padded[: len(values), batch_index] = values
    padded = padded.to(device)
    packed = torch.nn.utils.rnn.pack_padded_sequence(padded, sorted_lengths)
    target_tensor = torch.tensor(
        [targets[index] for index in sort_order], dtype=torch.long, device=device
    )
    auxiliary_tensor = None
    if use_redshift:
        auxiliary_tensor = torch.tensor(
            [[auxiliaries[index]] for index in sort_order],
            dtype=torch.float32,
            device=device,
        )
    return packed, auxiliary_tensor, target_tensor, reverse_sort


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
    cutoff_reference: str = "truth_t0",
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run deterministic batched inference and restore database object order.

    The model receives length-sorted batches, but returned IDs and probability rows
    follow the input database indices.  Logits are divided by the fitted temperature
    before softmax; a temperature above one softens overconfident predictions.
    """
    import torch

    device = "cuda" if use_cuda else "cpu"
    ids: list[str] = []
    probabilities = []
    ineligible = 0
    model.eval()
    for start in range(0, len(indices), batch_size):
        records, missing = store.fetch(
            indices[start : start + batch_size],
            cutoff_days,
            cutoff_reference=cutoff_reference,
        )
        ineligible += missing
        if not records:
            continue
        packed, auxiliary, _, reverse_sort = _make_packed_batch(
            records,
            SuperNNovaTrainingConfig(
                **{**store.config.normalized(), "random_length": False}
            ),
            device,
        )
        with torch.no_grad():
            logits = model(packed, auxiliary=auxiliary)
            batch_probabilities = (
                torch.softmax(logits / float(temperature), dim=1)
                .cpu()
                .numpy()[reverse_sort]
            )
        probabilities.append(batch_probabilities)
        ids.extend(record[2] for record in records)
    matrix = (
        np.concatenate(probabilities, axis=0)
        if probabilities
        else np.empty((0, len(workflow.FINAL_CLASSES)))
    )
    return np.asarray(ids, dtype=str), matrix, ineligible


def _class_weight_tensor(targets: np.ndarray, device: str) -> Any:
    """Build inverse-frequency loss weights so every class has equal total influence."""
    import torch

    counts = np.bincount(targets, minlength=len(workflow.FINAL_CLASSES)).astype(float)
    if (counts == 0).any():
        raise ValueError("Every final class must occur in SuperNNova training data")
    weights = len(targets) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _capture_rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """Capture every random stream needed for an exact interrupted-run continuation."""
    import torch

    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_generator": rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any], rng: np.random.Generator) -> None:
    """Restore Python, NumPy, generator, and PyTorch random streams."""
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write a checkpoint to a temporary sibling and publish it atomically."""
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_temperature_scaling(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Fit one positive calibration temperature on validation predictions.

    Temperature scaling changes confidence but not the winning class: probabilities
    are converted back to log scores, divided by a single optimized scalar, and
    renormalized.  The scalar minimizes class-balanced negative log likelihood so the
    calibration objective matches model selection without favoring abundant classes.
    """
    from scipy.optimize import minimize_scalar

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), len(workflow.FINAL_CLASSES)):
        raise ValueError("Temperature calibration received incompatible arrays")
    counts = np.bincount(labels, minlength=len(workflow.FINAL_CLASSES)).astype(float)
    if (counts == 0).any():
        raise ValueError("Temperature calibration requires every output class")
    sample_weights = len(labels) / (len(counts) * counts[labels])
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))

    def objective(log_temperature: float) -> float:
        """Return weighted NLL for a log-parameterized temperature."""
        scaled = log_probabilities / np.exp(log_temperature)
        scaled -= scaled.max(axis=1, keepdims=True)
        calibrated = np.exp(scaled)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        losses = -np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1.0))
        return float(np.average(losses, weights=sample_weights))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature calibration failed: {result.message}")
    return float(np.exp(result.x))


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: SuperNNovaTrainingConfig,
    store: HDF5SequenceStore,
    history: list[dict[str, Any]],
    epoch: int,
    best_loss: float,
    unimproved_epochs: int,
    elapsed_seconds: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Create the model, optimizer, scheduler, history, and RNG state for resuming."""
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config.normalized(),
        "database_identity": store.database_identity,
        "feature_names": list(feature_names(config.redshift_mode, config.engineered_features)),
        "class_order": list(workflow.FINAL_CLASSES),
        "history": history,
        "epoch": int(epoch),
        "best_validation_log_loss": float(best_loss),
        "unimproved_epochs": int(unimproved_epochs),
        "elapsed_seconds": float(elapsed_seconds),
        "rng_state": _capture_rng_state(rng),
        "temperature": 1.0,
        "source_versions": supernnova_source_versions(),
    }


def train_supernnova(
    database_path: str | Path,
    output_dir: str | Path,
    config: SuperNNovaTrainingConfig,
    action: str = "run",
    show_progress: bool = True,
    max_epochs_this_call: int | None = None,
) -> dict[str, Any]:
    """Explicitly start or resume one RNN training run.

    Training uses class-weighted cross entropy, optional random-prefix augmentation,
    Adam optimization, gradient clipping, and a plateau learning-rate scheduler.  At
    the end of every epoch ``last.pt`` captures all state needed for an exact resume;
    ``best.pt`` is replaced only after a meaningful validation improvement.  Early
    stopping and model selection use class-balanced validation log loss.  Test objects
    are never read in this function.

    ``action="run"`` requires a new output directory.  ``action="resume"`` requires
    an incomplete ``last.pt`` checkpoint and restores all optimizer, scheduler, and
    random state.  Loading a completed model is deliberately handled by
    :func:`load_supernnova_checkpoint`, so this function never decides implicitly
    between training and loading.  ``max_epochs_this_call`` deliberately pauses a run
    after a bounded number of epochs and is primarily useful for resume tests.  Once
    training completes, validation-only temperature scaling is stored in the best
    checkpoint when requested.

    Returns a JSON-compatible dictionary containing elapsed time and per-epoch metrics.
    """
    import torch
    from .sequence_rnn import WarpSequenceRNN

    if action not in {"run", "resume"}:
        raise ValueError("Training action must be 'run' or 'resume'")
    if max_epochs_this_call is not None and max_epochs_this_call < 1:
        raise ValueError("max_epochs_this_call must be positive when provided")

    device, use_cuda = _resolve_device(config)
    torch.set_num_threads(max(1, int(config.threads)))
    workflow.set_random_seed(config.seed)
    store = HDF5SequenceStore(database_path, config)
    train_indices = store.indices("train")
    validation_indices = store.indices("validation")
    if not len(validation_indices):
        raise ValueError("SuperNNova training requires a validation partition")

    output_dir = Path(output_dir)
    if action == "run":
        output_dir.mkdir(parents=True, exist_ok=False)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.json"
    complete_path = output_dir / "complete.json"
    expected_identity = workflow.configuration_hash(
        {**config.normalized(), "database_identity": store.database_identity}
    )
    if action == "resume" and complete_path.exists():
        raise RuntimeError("A completed SuperNNova run cannot be resumed; load it instead")

    settings = _model_settings(config, use_cuda)
    recurrent_features = feature_names(config.redshift_mode, config.engineered_features)
    recurrent_input_size = len(recurrent_features) - int(
        config.redshift_mode == "photometry_plus_truth_z"
    )
    model = WarpSequenceRNN(recurrent_input_size, settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        # PyTorch reduces only after ``patience + 1`` bad epochs.  Subtract one so
        # the public setting retains its literal "after N bad epochs" meaning.
        patience=config.lr_patience - 1,
        threshold=config.minimum_improvement,
        threshold_mode="abs",
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=_class_weight_tensor(store.targets[train_indices], device)
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    unimproved_epochs = 0
    start_epoch = 0
    elapsed_before = 0.0
    rng = np.random.default_rng(config.seed)

    if action == "resume":
        # Resume optimizer, scheduler, counters, and random streams together.  Loading
        # weights alone would change subsequent shuffling and prefix augmentation.
        resumed = torch.load(last_path, map_location=device, weights_only=False)
        if resumed.get("database_identity") != store.database_identity:
            raise ValueError("Incomplete checkpoint belongs to another sequence database")
        if resumed.get("config") != config.normalized():
            raise ValueError("Incomplete checkpoint belongs to another training configuration")
        model.load_state_dict(resumed["model_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        scheduler.load_state_dict(resumed["scheduler_state"])
        history = list(resumed["history"])
        start_epoch = int(resumed["epoch"]) + 1
        best_loss = float(resumed["best_validation_log_loss"])
        unimproved_epochs = int(resumed["unimproved_epochs"])
        elapsed_before = float(resumed.get("elapsed_seconds", 0.0))
        if "rng_state" not in resumed:
            raise ValueError("Checkpoint predates reproducible RNG-state persistence")
        _restore_rng_state(resumed["rng_state"], rng)

    started = time.perf_counter()
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

    paused = False
    epochs_run_this_call = 0
    for epoch in epoch_iterator:
        model.train()
        # The explicit NumPy generator makes object shuffling and random prefixes part
        # of checkpointed state instead of relying on an implicit global stream.
        shuffled = rng.permutation(train_indices)
        batch_losses = []
        batch_gradient_norms = []
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
            packed, auxiliary, targets, _ = _make_packed_batch(
                records,
                config,
                device,
                rng=rng,
            )
            optimizer.zero_grad()
            logits = model(packed, auxiliary=auxiliary)
            loss = criterion(logits, targets)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            batch_gradient_norms.append(float(gradient_norm.detach().cpu()))
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
        improved = validation_loss < best_loss - config.minimum_improvement
        if improved:
            best_loss = validation_loss
            unimproved_epochs = 0
        else:
            unimproved_epochs += 1
        scheduler.step(validation_loss)
        row["gradient_norm_mean"] = float(np.mean(batch_gradient_norms))
        row["gradient_norm_max"] = float(np.max(batch_gradient_norms))

        elapsed = elapsed_before + time.perf_counter() - started
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            config,
            store,
            history,
            epoch,
            best_loss,
            unimproved_epochs,
            elapsed,
            rng,
        )
        _atomic_torch_save(payload, last_path)
        # ``last`` supports resumption; ``best`` is the validation-selected model used
        # for calibration and all final predictions.
        if improved:
            _atomic_torch_save(payload, best_path)
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
        epochs_run_this_call += 1
        if (
            max_epochs_this_call is not None
            and epochs_run_this_call >= max_epochs_this_call
            and epoch + 1 < config.epochs
        ):
            paused = True
            break

    if not best_path.exists():
        raise RuntimeError("SuperNNova training produced no best checkpoint")
    elapsed = elapsed_before + time.perf_counter() - started
    result = {"elapsed_seconds": elapsed, "epochs": history}
    workflow.write_json_once(result, history_path, overwrite=True)
    if paused:
        if show_progress:
            print(f"Paused {config.redshift_mode} after epoch {len(history)}.")
        return result

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    temperature = 1.0
    if config.calibrate_probabilities:
        _, validation_probabilities, _ = _probabilities_for_indices(
            model,
            store,
            validation_indices,
            config.batch_size,
            use_cuda,
        )
        temperature = fit_temperature_scaling(
            store.targets[validation_indices], validation_probabilities
        )
    best_payload["temperature"] = temperature
    _atomic_torch_save(best_payload, best_path)
    workflow.write_json_once(
        {
            "status": "complete",
            "training_identity": expected_identity,
            "best_validation_log_loss": best_loss,
            "completed_epochs": len(history),
            "device": device,
            "temperature": temperature,
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


def load_supernnova_training_history(output_dir: str | Path) -> dict[str, Any]:
    """Load a completed or paused training history from its fixed JSON path."""
    return json.loads((Path(output_dir) / "history.json").read_text())


def load_supernnova_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[Any, SuperNNovaTrainingConfig]:
    """Reconstruct an RNN and validate a saved checkpoint's feature contracts.

    The checkpoint config determines the architecture.  Feature and class order are
    checked before weights are loaded, preventing a numerically compatible but
    semantically wrong output layer from being used with changed code or taxonomy.
    """
    import torch
    from .sequence_rnn import WarpSequenceRNN

    requested = device
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint loading was requested but CUDA is unavailable")
    payload = torch.load(checkpoint_path, map_location=requested, weights_only=False)
    config = SuperNNovaTrainingConfig(**payload["config"])
    expected_features = list(feature_names(config.redshift_mode, config.engineered_features))
    if payload.get("feature_names") != expected_features:
        raise ValueError("Checkpoint feature order does not match its configuration")
    if payload.get("class_order") != list(workflow.FINAL_CLASSES):
        raise ValueError("Checkpoint class order does not match the active taxonomy")
    settings = _model_settings(config, requested == "cuda")
    recurrent_features = feature_names(config.redshift_mode, config.engineered_features)
    recurrent_input_size = len(recurrent_features) - int(
        config.redshift_mode == "photometry_plus_truth_z"
    )
    model = WarpSequenceRNN(recurrent_input_size, settings).to(requested)
    model.load_state_dict(payload["model_state"])
    model.temperature = float(payload.get("temperature", 1.0))
    model.database_identity = str(payload["database_identity"])
    model.eval()
    return model, config


def predict_supernnova(
    checkpoint_path: str | Path,
    database_path: str | Path,
    role: str,
    cutoff_days: float | None = None,
    cutoff_reference: str = "truth_t0",
    device: str = "auto",
) -> PredictionResult:
    """Predict one database role from complete or time-truncated light curves.

    Checkpoint and HDF5 identities must match.  Returned probability columns follow
    :data:`workflow.FINAL_CLASSES` and are validated for finiteness and unit sums.
    See :meth:`HDF5SequenceStore.fetch` for cutoff semantics and eligibility.
    """
    from astropy.table import Table

    model, config = load_supernnova_checkpoint(checkpoint_path, device=device)
    resolved_device, use_cuda = _resolve_device(
        SuperNNovaTrainingConfig(**{**config.normalized(), "device": device})
    )
    model = model.to(resolved_device)
    store = HDF5SequenceStore(database_path, config)
    if model.database_identity != store.database_identity:
        raise ValueError("Checkpoint and sequence database identities do not match")
    indices = store.indices(role)
    ids, probabilities, ineligible = _probabilities_for_indices(
        model,
        store,
        indices,
        config.batch_size,
        use_cuda,
        cutoff_days=cutoff_days,
        cutoff_reference=cutoff_reference,
        temperature=float(model.temperature),
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
        cutoff_reference=cutoff_reference if cutoff_days is not None else None,
    )


def predict_supernnova_ensemble(
    checkpoint_paths: Sequence[str | Path],
    database_path: str | Path,
    role: str,
    cutoff_days: float | None = None,
    cutoff_reference: str = "truth_t0",
    device: str = "auto",
) -> PredictionResult:
    """Average calibrated probabilities from several independently trained seeds.

    This is a simple probability ensemble, not a second learned model.  Every member
    must predict the same objects in exactly the same order, making row misalignment a
    hard error rather than silently averaging unrelated transients.
    """
    if not checkpoint_paths:
        raise ValueError("An ensemble requires at least one checkpoint")
    results = [
        predict_supernnova(
            path,
            database_path,
            role,
            cutoff_days=cutoff_days,
            cutoff_reference=cutoff_reference,
            device=device,
        )
        for path in checkpoint_paths
    ]
    reference_ids = np.asarray(results[0].classifications["object_id"], dtype=str)
    matrices = []
    for result in results:
        ids = np.asarray(result.classifications["object_id"], dtype=str)
        if not np.array_equal(ids, reference_ids):
            raise ValueError("Ensemble checkpoints produced different object orders")
        matrices.append(
            np.column_stack(
                [
                    np.asarray(result.classifications[label], dtype=float)
                    for label in workflow.FINAL_CLASSES
                ]
            )
        )
    from astropy.table import Table

    mean_probabilities = np.mean(matrices, axis=0)
    classifications = Table({"object_id": reference_ids})
    for index, label in enumerate(workflow.FINAL_CLASSES):
        classifications[label] = mean_probabilities[:, index]
    return PredictionResult(
        classifications=classifications,
        eligible_objects=results[0].eligible_objects,
        ineligible_objects=results[0].ineligible_objects,
        cutoff_days=cutoff_days,
        cutoff_reference=cutoff_reference if cutoff_days is not None else None,
    )


def paired_group_bootstrap_differences(
    first_predictions: pd.DataFrame,
    second_predictions: pd.DataFrame,
    split_manifest: pd.DataFrame,
    repeats: int = 1000,
    seed: int = workflow.DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Estimate uncertainty on paired model differences by provenance group.

    Only objects predicted by both models are compared.  Each bootstrap draw reuses
    the same sampled group positions for both prediction tables, preserving their
    pairing and much of the shared-sample covariance.  Intervals report
    ``second - first``; the desirable sign therefore depends on the metric (negative
    for losses, positive for accuracies and F1).
    """
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
