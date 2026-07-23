"""Shared splitting, ParSNIP adaptation, evaluation, and experiment utilities.

The functions in this module deliberately keep the observation Parquet tree as the
single source of photometry.  Split manifests contain only object-level metadata;
notebooks select observations by ``object_id`` when a partition is needed.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FINAL_CLASSES = (
    "SLSN",
    "SN IIP",
    "SN IIb",
    "SN IIn",
    "SN Ib",
    "SN Ic",
    "SN Ic-BL",
)
CLASS_MAP = {
    "SLSN-I": "SLSN",
    "SLSN-II": "SLSN",
    "SN IIP": "SN IIP",
    "SN IIb": "SN IIb",
    "SN IIn": "SN IIn",
    "SN Ib": "SN Ib",
    "SN Ic": "SN Ic",
    "SN Ic-BL": "SN Ic-BL",
}
EXPECTED_BANDS = (
    "lsstu",
    "lsstg",
    "lsstr",
    "lssti",
    "lsstz",
    "lssty",
    "ztfg",
    "ztfr",
    "ztfi",
)
SPLIT_NAMES = {0: "test", 1: "validation"}
DEFAULT_SPLIT_SEED = 20260721
DEFAULT_N_SPLITS = 10
DEFAULT_MIN_GROUPED_EPOCHS = 3
DEFAULT_EPOCH_WINDOW_DAYS = 0.33
PARSNIP_ZEROPOINT = 25.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable identity and configuration for one classifier experiment."""

    training_sample: str
    evaluation_sample: str
    backend: str = "parsnip"
    redshift_mode: str = "truth_z"
    split_strategy: str = "template_key"
    seed: int = DEFAULT_SPLIT_SEED
    model_config: Mapping[str, Any] | None = None
    run_id: str | None = None

    def normalized(self) -> dict[str, Any]:
        """Return a stable JSON-compatible configuration mapping."""
        payload = asdict(self)
        payload["model_config"] = dict(self.model_config or {})
        payload["run_id"] = self.run_id or make_run_id(payload)
        return payload


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when PyTorch is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def merge_fitclasses(labels: Iterable[str]) -> pd.Series:
    """Map raw fit classes into the fixed seven-class taxonomy."""
    series = pd.Series(labels, copy=False)
    unknown = sorted(set(series.dropna()) - set(CLASS_MAP))
    if unknown:
        raise ValueError(f"Unmapped fitclass values: {unknown}")
    return series.map(CLASS_MAP)


def _read_parquet_files(
    files: Sequence[Path],
    columns: Sequence[str] | None = None,
    object_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read selected Parquet files, optionally retaining only requested objects."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if not files:
        raise FileNotFoundError("No Parquet files were provided")
    requested_columns = list(columns) if columns is not None else None
    read_columns = requested_columns
    selected_ids = None
    if object_ids is not None:
        selected_ids = {str(object_id) for object_id in object_ids}
        if not selected_ids:
            return pd.DataFrame(columns=requested_columns or [])
        if read_columns is not None and "object_id" not in read_columns:
            read_columns = [*read_columns, "object_id"]

    tables = []
    value_set = pa.array(sorted(selected_ids)) if selected_ids is not None else None
    for file in files:
        table = pq.ParquetFile(file).read(columns=read_columns)
        if value_set is not None:
            table = table.filter(pc.is_in(table["object_id"], value_set=value_set))
        if table.num_rows:
            tables.append(table)
    if not tables:
        return pd.DataFrame(columns=requested_columns or read_columns or [])
    try:
        combined = pa.concat_tables(tables, promote_options="permissive")
    except TypeError:
        # PyArrow before 14 used the boolean ``promote`` spelling.
        combined = pa.concat_tables(tables, promote=True)
    frame = combined.to_pandas()
    if requested_columns is not None:
        frame = frame[requested_columns]
    return frame


def load_parquet_tree(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read one conventional partitioned Parquet tree without Hive conflicts."""
    path = Path(path)
    files = sorted(path.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found below {path}")
    return _read_parquet_files(files, columns=columns)


def sample_table_files(sample_dir: str | Path, table: str) -> list[Path]:
    """Return schema-5 or schema-6 ensemble Parquet files for one logical table."""
    if table not in {"truth", "observations"}:
        raise ValueError("table must be 'truth' or 'observations'")
    sample_dir = Path(sample_dir)
    direct = sorted((sample_dir / table).glob("**/*.parquet"))
    ensemble = sorted(sample_dir.glob(f"*/{table}/**/*.parquet"))
    if direct and ensemble:
        raise ValueError(f"Sample mixes direct and ensemble {table} layouts: {sample_dir}")
    files = direct or ensemble
    if not files:
        raise FileNotFoundError(f"No {table} Parquet files found below {sample_dir}")
    return files


def load_sample_manifest(sample_dir: str | Path) -> dict[str, Any]:
    """Load the schema-5 run manifest or schema-6 top-level ensemble manifest."""
    sample_dir = Path(sample_dir)
    candidates = [sample_dir / "ensemble_manifest.json", sample_dir / "manifest.json"]
    existing = [path for path in candidates if path.exists()]
    if len(existing) > 1:
        raise ValueError(f"Sample has ambiguous top-level manifests: {existing}")
    return json.loads(existing[0].read_text()) if existing else {}


def load_sample_truth(
    sample_dir: str | Path,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load object-level truth from a direct run or realization ensemble."""
    return _read_parquet_files(sample_table_files(sample_dir, "truth"), columns=columns)


def load_sample_observations(
    sample_dir: str | Path,
    columns: Sequence[str] | None = None,
    object_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load all or selected observations from either supported sample layout."""
    return _read_parquet_files(
        sample_table_files(sample_dir, "observations"),
        columns=columns,
        object_ids=object_ids,
    )


def load_training_sample(
    sample_dir: str | Path,
    truth_columns: Sequence[str] | None = None,
    observation_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load all truth and observations from a direct run or realization ensemble."""
    truth = load_sample_truth(sample_dir, truth_columns)
    observations = load_sample_observations(sample_dir, observation_columns)
    source_manifest = load_sample_manifest(sample_dir)
    return truth, observations, source_manifest


def audit_sample_observations(sample_dir: str | Path) -> dict[str, Any]:
    """Scan observation files with bounded memory and summarize their input contract."""
    import pyarrow.parquet as pq

    row_count = 0
    object_ids: set[str] = set()
    band_counts: Counter[str] = Counter()
    invalid_rows = 0
    for file in sample_table_files(sample_dir, "observations"):
        frame = pq.ParquetFile(file).read(
            columns=["object_id", "band", "mjd", "flux", "fluxerr", "zp"]
        ).to_pandas()
        row_count += len(frame)
        object_ids.update(frame["object_id"].astype(str).unique())
        band_counts.update(frame["band"].astype(str).value_counts().to_dict())
        numeric = frame[["mjd", "flux", "fluxerr", "zp"]].to_numpy(dtype=float)
        invalid_rows += int((~np.isfinite(numeric).all(axis=1) | (frame["fluxerr"] <= 0)).sum())
    return {
        "observation_rows": int(row_count),
        "objects_with_observations": int(len(object_ids)),
        "band_counts": {band: int(band_counts[band]) for band in sorted(band_counts)},
        "invalid_rows": int(invalid_rows),
    }


def rescale_flux_to_zeropoint(
    observations: pd.DataFrame,
    target_zeropoint: float = PARSNIP_ZEROPOINT,
) -> pd.DataFrame:
    """Rescale flux and uncertainty while preserving the corresponding AB magnitude."""
    required = {"flux", "fluxerr", "zp"}
    missing = required - set(observations)
    if missing:
        raise ValueError(f"Missing zeropoint-conversion columns: {sorted(missing)}")
    converted = observations.copy()
    # Multiplying both quantities preserves S/N and converts the reported flux to the
    # target magnitude convention: m_AB = target_zp - 2.5 log10(flux_target).
    scale = 10.0 ** ((target_zeropoint - converted["zp"].to_numpy()) / 2.5)
    converted["flux"] = converted["flux"].to_numpy() * scale
    converted["fluxerr"] = converted["fluxerr"].to_numpy() * scale
    converted["zp"] = float(target_zeropoint)
    return converted


def group_observing_epochs(
    observations: pd.DataFrame,
    window_days: float = DEFAULT_EPOCH_WINDOW_DAYS,
) -> pd.DataFrame:
    """Apply SuperNNova-style 0.33-day grouping and duplicate-band selection."""
    required = {"object_id", "mjd", "band", "fluxerr"}
    missing = required - set(observations)
    if missing:
        raise ValueError(f"Missing epoch-grouping columns: {sorted(missing)}")
    ordered = observations.sort_values(["object_id", "mjd", "fluxerr"]).copy()
    grouped_mjd = np.empty(len(ordered), dtype=float)

    # SuperNNova starts a new epoch when the time since the last group anchor exceeds
    # the window.  Comparing with the anchor avoids chained measurements extending a
    # single night indefinitely.
    cursor = 0
    for _, group in ordered.groupby("object_id", sort=False):
        times = group["mjd"].to_numpy(dtype=float)
        anchor = times[0]
        for offset, time in enumerate(times):
            if offset == 0 or time - anchor > window_days:
                anchor = time
            grouped_mjd[cursor + offset] = anchor
        cursor += len(group)
    ordered["grouped_mjd"] = grouped_mjd

    # Retain the statistically most precise measurement if a band repeats in an epoch.
    grouped = (
        ordered.sort_values("fluxerr")
        .drop_duplicates(["object_id", "grouped_mjd", "band"], keep="first")
        .sort_values(["object_id", "grouped_mjd", "band"])
        .reset_index(drop=True)
    )
    return grouped


def grouped_epoch_counts(
    observations: pd.DataFrame,
    window_days: float = DEFAULT_EPOCH_WINDOW_DAYS,
) -> pd.Series:
    """Count grouped observing epochs per object."""
    grouped = group_observing_epochs(observations, window_days=window_days)
    return grouped.groupby("object_id")["grouped_mjd"].nunique().rename("grouped_epochs")


def grouped_epoch_counts_from_sample(
    sample_dir: str | Path,
    window_days: float = DEFAULT_EPOCH_WINDOW_DAYS,
) -> pd.Series:
    """Count grouped epochs across a large sample without loading it into memory."""
    import pyarrow.parquet as pq

    counts: dict[str, int] = {}
    for file in sample_table_files(sample_dir, "observations"):
        frame = pq.ParquetFile(file).read(columns=["object_id", "mjd"]).to_pandas()
        for object_id, rows in frame.groupby("object_id", sort=False):
            object_id = str(object_id)
            if object_id in counts:
                raise ValueError(
                    f"Object {object_id!r} crosses observation files; streaming epoch "
                    "counting requires each object to remain in one batch"
                )
            times = np.sort(rows["mjd"].to_numpy(dtype=float))
            if not len(times):
                counts[object_id] = 0
                continue
            anchor = times[0]
            epochs = 1
            # Use the same anchored 0.33-day convention as group_observing_epochs.
            for time_value in times[1:]:
                if time_value - anchor > window_days:
                    anchor = time_value
                    epochs += 1
            counts[object_id] = epochs
    result = pd.Series(counts, name="grouped_epochs", dtype=int)
    result.index.name = "object_id"
    return result


def _active_groups(truth: pd.DataFrame, strategy: str) -> pd.Series:
    """Build leakage groups for a supported split strategy."""
    if strategy == "template_key":
        if "template_key" not in truth:
            raise ValueError("template_key is required for template_key splitting")
        return truth["template_key"].astype(str)
    if strategy == "basis_sn":
        if "basis_sn" not in truth:
            raise ValueError("basis_sn is required for basis_sn splitting")
        # Basis names can recur across physical classes, so class-qualify the ID.
        return truth["fitclass"].astype(str) + "|" + truth["basis_sn"].astype(str)
    raise ValueError("split_strategy must be 'template_key' or 'basis_sn'")


def create_grouped_split_manifest(
    truth: pd.DataFrame,
    observations: pd.DataFrame | None = None,
    strategy: str = "template_key",
    seed: int = DEFAULT_SPLIT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
    min_grouped_epochs: int = DEFAULT_MIN_GROUPED_EPOCHS,
    epoch_window_days: float = DEFAULT_EPOCH_WINDOW_DAYS,
    epoch_counts: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic stratified group folds and an exclusion manifest."""
    from sklearn.model_selection import StratifiedGroupKFold

    if truth["object_id"].duplicated().any():
        raise ValueError("Truth table contains duplicate object_id values")
    prepared = truth.copy()
    prepared["raw_label"] = prepared["fitclass"].astype(str)
    prepared["final_label"] = merge_fitclasses(prepared["fitclass"]).to_numpy()
    prepared["group_id"] = _active_groups(prepared, strategy).to_numpy()
    if (observations is None) == (epoch_counts is None):
        raise ValueError("Provide exactly one of observations or epoch_counts")
    counts = (
        grouped_epoch_counts(observations, window_days=epoch_window_days)
        if epoch_counts is None
        else epoch_counts.rename("grouped_epochs")
    )
    prepared = prepared.merge(counts, how="left", left_on="object_id", right_index=True)
    prepared["grouped_epochs"] = prepared["grouped_epochs"].fillna(0).astype(int)

    retained_mask = prepared["grouped_epochs"] >= min_grouped_epochs
    excluded = prepared.loc[~retained_mask].copy()
    excluded["exclusion_reason"] = f"fewer_than_{min_grouped_epochs}_grouped_epochs"
    retained = prepared.loc[retained_mask].copy().reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    retained["fold"] = -1
    dummy = np.zeros(len(retained), dtype=np.uint8)
    for fold, (_, held_out) in enumerate(
        splitter.split(dummy, retained["final_label"], retained["group_id"])
    ):
        retained.loc[held_out, "fold"] = fold
    retained["fold"] = retained["fold"].astype(int)
    retained["split"] = retained["fold"].map(SPLIT_NAMES).fillna("train")
    retained["split_strategy"] = strategy
    retained["split_seed"] = int(seed)

    columns = [
        "object_id",
        "raw_label",
        "final_label",
        "template_key",
        "basis_sn",
        "group_id",
        "grouped_epochs",
        "fold",
        "split",
        "split_strategy",
        "split_seed",
    ]
    optional = [
        column
        for column in ("z", "t0", "ra", "dec", "survey_realization_id")
        if column in retained
    ]
    manifest = retained[columns + optional].sort_values("object_id").reset_index(drop=True)
    excluded_columns = [
        column
        for column in (
            "object_id",
            "raw_label",
            "final_label",
            "template_key",
            "basis_sn",
            "survey_realization_id",
        )
        if column in excluded
    ]
    excluded = excluded[excluded_columns + ["grouped_epochs", "exclusion_reason"]]
    excluded = excluded.sort_values("object_id").reset_index(drop=True)
    validate_split_manifest(manifest)
    return manifest, excluded


def validate_split_manifest(manifest: pd.DataFrame) -> None:
    """Assert object uniqueness, partition completeness, and zero group leakage."""
    required = {"object_id", "group_id", "fold", "split", "final_label"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    if manifest["object_id"].isna().any() or manifest["object_id"].duplicated().any():
        raise ValueError("Every retained object must occur exactly once")
    if not set(manifest["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Split manifest contains an unknown partition")
    group_partition_counts = manifest.groupby("group_id")["split"].nunique()
    leaking = group_partition_counts[group_partition_counts > 1]
    if len(leaking):
        raise ValueError(f"Groups cross partitions: {leaking.index[:5].tolist()}")
    if (manifest.loc[manifest["fold"] == 0, "split"] != "test").any():
        raise ValueError("Fold 0 must be test")
    if (manifest.loc[manifest["fold"] == 1, "split"] != "validation").any():
        raise ValueError("Fold 1 must be validation")
    if (manifest.loc[manifest["fold"] >= 2, "split"] != "train").any():
        raise ValueError("Folds 2 and above must be training")


def prepare_grouped_epoch_counts(
    sample_dir: str | Path,
    artifact_dir: str | Path,
    overwrite: bool = False,
) -> pd.Series:
    """Load or persist the sample-wide grouped-epoch count used by every split."""
    artifact_dir = Path(artifact_dir)
    count_path = artifact_dir / "grouped_epoch_counts.parquet"
    if count_path.exists() and not overwrite:
        frame = pd.read_parquet(count_path)
        if frame["object_id"].duplicated().any():
            raise ValueError("Grouped-epoch cache contains duplicate object IDs")
        return frame.set_index("object_id")["grouped_epochs"].astype(int)

    counts = grouped_epoch_counts_from_sample(sample_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_table_once(counts.reset_index(), count_path, overwrite=overwrite)
    return counts


def prepare_persistent_split(
    sample_dir: str | Path,
    artifact_dir: str | Path,
    strategy: str = "template_key",
    seed: int = DEFAULT_SPLIT_SEED,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a saved split or create, validate, and persist it once."""
    artifact_dir = Path(artifact_dir)
    split_path = artifact_dir / f"split_{strategy}_seed{seed}.parquet"
    excluded_path = artifact_dir / "excluded_objects.parquet"
    if split_path.exists() and excluded_path.exists() and not overwrite:
        manifest = pd.read_parquet(split_path)
        excluded = pd.read_parquet(excluded_path)
        validate_split_manifest(manifest)
        return manifest, excluded
    if split_path.exists() and not excluded_path.exists() and not overwrite:
        raise RuntimeError(
            "Split artifacts are incomplete; inspect them and use overwrite=True to rebuild"
        )

    truth_columns = [
        "object_id",
        "fitclass",
        "template_key",
        "basis_sn",
        "z",
        "t0",
        "ra",
        "dec",
    ]
    truth = load_sample_truth(sample_dir)
    if "survey_realization_id" in truth:
        truth_columns.append("survey_realization_id")
    truth = truth[truth_columns]
    counts = prepare_grouped_epoch_counts(
        sample_dir, artifact_dir, overwrite=overwrite
    )
    manifest, excluded = create_grouped_split_manifest(
        truth, strategy=strategy, seed=seed, epoch_counts=counts
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_table_once(manifest, split_path, overwrite=overwrite)
    # Exclusions are independent of strategy; both strategies must reproduce them.
    if excluded_path.exists() and not overwrite:
        existing = pd.read_parquet(excluded_path).sort_values("object_id").reset_index(drop=True)
        if not existing.equals(excluded):
            raise ValueError("Existing excluded-object manifest does not match this sample")
    else:
        write_table_once(excluded, excluded_path, overwrite=overwrite)
    return manifest, excluded


def select_partition_rows(
    observations: pd.DataFrame | str | Path,
    split_manifest: pd.DataFrame,
    partition: str | Sequence[str],
) -> pd.DataFrame:
    """Select observation rows through object IDs without copying source storage."""
    partitions = {partition} if isinstance(partition, str) else set(partition)
    ids = set(split_manifest.loc[split_manifest["split"].isin(partitions), "object_id"])
    if isinstance(observations, (str, Path)):
        return load_sample_observations(observations, object_ids=sorted(ids))
    return observations.loc[observations["object_id"].isin(ids)].copy()


def sample_balanced_object_ids(
    split_manifest: pd.DataFrame,
    partition: str = "train",
    per_class: int = 64,
    seed: int = DEFAULT_SPLIT_SEED,
    folds: Sequence[int] | None = None,
) -> list[str]:
    """Draw a deterministic class-balanced object subset from one partition."""
    selected = split_manifest.loc[split_manifest["split"] == partition]
    if folds is not None:
        selected = selected.loc[selected["fold"].isin(folds)]
    if selected.empty:
        raise ValueError(f"Partition {partition!r} with folds {folds} is empty")
    rng = np.random.default_rng(seed)
    result: list[str] = []
    for label in FINAL_CLASSES:
        candidates = selected.loc[selected["final_label"] == label, "object_id"].to_numpy()
        if not len(candidates):
            raise ValueError(f"No {label} objects are available in {partition}")
        count = min(per_class, len(candidates))
        result.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return result


def _select_lcdata_metadata(
    truth: pd.DataFrame,
    object_ids: Sequence[str] | None,
) -> pd.DataFrame:
    """Select and format object metadata required by lcdata and ParSNIP."""
    selected = truth.copy()
    if object_ids is not None:
        id_order = {str(object_id): index for index, object_id in enumerate(object_ids)}
        selected = selected[selected["object_id"].isin(id_order)].copy()
        selected["_order"] = selected["object_id"].map(id_order)
        selected = selected.sort_values("_order").drop(columns="_order")
    if selected.empty:
        raise ValueError("Cannot create lcdata from an empty truth selection")
    selected["type"] = merge_fitclasses(selected["fitclass"]).to_numpy()
    selected["redshift"] = selected["z"].astype(float)
    metadata_columns = ["object_id", "type", "redshift"]
    metadata_columns += [column for column in ("ra", "dec") if column in selected]
    return selected[metadata_columns].copy()


def _prepare_lcdata_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Apply the ParSNIP zeropoint, band, finiteness, and ordering contract."""
    converted = rescale_flux_to_zeropoint(observations, PARSNIP_ZEROPOINT)
    unknown_bands = sorted(set(converted["band"]) - set(EXPECTED_BANDS))
    if unknown_bands:
        raise ValueError(f"Unexpected bandpasses: {unknown_bands}")
    converted = converted.rename(columns={"mjd": "time"})
    converted = converted[["object_id", "time", "band", "flux", "fluxerr"]]
    converted = converted.sort_values(["object_id", "time", "band"]).reset_index(drop=True)
    numeric = converted[["time", "flux", "fluxerr"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (converted["fluxerr"] <= 0).any():
        raise ValueError("lcdata observations must have finite values and positive errors")
    if (converted.groupby("object_id")["time"].diff().dropna() < 0).any():
        raise ValueError("Light curves must be chronologically ordered")
    return converted


def to_lcdata_from_sample(
    truth: pd.DataFrame,
    sample_dir: str | Path,
    object_ids: Sequence[str] | None = None,
):
    """Build lcdata batchwise from a large direct or ensemble Parquet sample."""
    from astropy.table import Table, vstack
    import lcdata
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    metadata = _select_lcdata_metadata(truth, object_ids)
    selected_ids = set(metadata["object_id"].astype(str))
    value_set = pa.array(sorted(selected_ids))
    seen_ids: set[str] = set()
    metadata_tables = []
    light_curves = []
    columns = ["object_id", "mjd", "band", "flux", "fluxerr", "zp"]
    for file in sample_table_files(sample_dir, "observations"):
        table = pq.ParquetFile(file).read(columns=columns)
        table = table.filter(pc.is_in(table["object_id"], value_set=value_set))
        if not table.num_rows:
            continue
        observations = _prepare_lcdata_observations(table.to_pandas())
        file_ids = set(observations["object_id"].astype(str).unique())
        overlap = seen_ids & file_ids
        if overlap:
            raise ValueError(f"Objects cross observation files: {sorted(overlap)[:5]}")
        seen_ids.update(file_ids)
        file_metadata = metadata[metadata["object_id"].isin(file_ids)]
        child = lcdata.from_observations(
            Table.from_pandas(file_metadata), Table.from_pandas(observations)
        )
        metadata_tables.append(child.meta)
        light_curves.extend(child.light_curves)

    missing_ids = selected_ids - seen_ids
    if missing_ids:
        raise ValueError(f"Objects without observations: {sorted(missing_ids)[:5]}")
    # Construct once after streaming so repeated Dataset addition cannot become quadratic.
    combined_metadata = vstack(metadata_tables, metadata_conflicts="silent")
    return lcdata.Dataset(combined_metadata, light_curves)


def to_lcdata(
    truth: pd.DataFrame,
    observations: pd.DataFrame | str | Path,
    object_ids: Sequence[str] | None = None,
):
    """Convert selected simulated photometry to a validated ParSNIP/lcdata dataset."""
    from astropy.table import Table
    import lcdata

    if isinstance(observations, (str, Path)):
        return to_lcdata_from_sample(truth, observations, object_ids=object_ids)

    metadata = _select_lcdata_metadata(truth, object_ids)
    selected_obs = observations.copy()
    if object_ids is not None:
        selected_obs = selected_obs[selected_obs["object_id"].isin(object_ids)].copy()
    converted = _prepare_lcdata_observations(selected_obs)
    missing_ids = set(metadata["object_id"]) - set(converted["object_id"])
    if missing_ids:
        raise ValueError(f"Objects without observations: {sorted(missing_ids)[:5]}")

    dataset = lcdata.from_observations(Table.from_pandas(metadata), Table.from_pandas(converted))
    return dataset


def dataset_for_partition(
    truth: pd.DataFrame,
    observations: pd.DataFrame | str | Path,
    split_manifest: pd.DataFrame,
    partition: str | Sequence[str],
):
    """Build an lcdata dataset for one or more named partitions."""
    partitions = {partition} if isinstance(partition, str) else set(partition)
    ids = split_manifest.loc[split_manifest["split"].isin(partitions), "object_id"].tolist()
    return to_lcdata(truth, observations, object_ids=ids)


@contextmanager
def _parsnip_lightgbm_compatibility():
    """Adapt ParSNIP's legacy fit call to LightGBM 4 without editing ParSNIP."""

    import lightgbm

    original_fit = lightgbm.LGBMClassifier.fit
    if "verbose" in inspect.signature(original_fit).parameters:
        yield
        return

    def compatible_fit(classifier, *args, **kwargs):
        """Discard the fit-level verbose option removed by LightGBM 4."""

        kwargs.pop("verbose", None)
        return original_fit(classifier, *args, **kwargs)

    # ParSNIP owns the outer training loop, so narrowly patch its dependency for
    # the duration of that call and always restore the kernel-installed class.
    lightgbm.LGBMClassifier.fit = compatible_fit
    try:
        yield
    finally:
        lightgbm.LGBMClassifier.fit = original_fit


def tune_parsnip_classifier(
    train_representations,
    validation_representations,
    min_child_weights: Sequence[float] = (1.0, 10.0, 30.0),
) -> tuple[Any, pd.DataFrame]:
    """Select ParSNIP's class-reweighted LightGBM model on validation log loss."""
    import parsnip

    labels = np.asarray(train_representations["type"]).astype(str)
    validation_labels = np.asarray(validation_representations["type"]).astype(str)
    rows = []
    models = []
    for min_child_weight in min_child_weights:
        classifier = parsnip.Classifier()
        with _parsnip_lightgbm_compatibility():
            classifier.train(
                train_representations.copy(),
                num_folds=1,
                labels=labels,
                reweight=True,
                min_child_weight=float(min_child_weight),
            )
        classified = classifier.classify(validation_representations)
        probabilities = _classification_probabilities(classified, FINAL_CLASSES)
        score = class_balanced_log_loss(validation_labels, probabilities, FINAL_CLASSES)
        rows.append({"min_child_weight": float(min_child_weight), "validation_log_loss": score})
        models.append(classifier)
    scores = pd.DataFrame(rows).sort_values("validation_log_loss").reset_index(drop=True)
    best_weight = scores.loc[0, "min_child_weight"]
    return models[list(min_child_weights).index(best_weight)], scores


def refit_parsnip_classifier(
    train_representations,
    validation_representations,
    min_child_weight: float,
):
    """Refit the selected ParSNIP LightGBM classifier on train plus validation."""
    from astropy.table import vstack
    import parsnip

    combined = vstack([train_representations, validation_representations], metadata_conflicts="silent")
    classifier = parsnip.Classifier()
    with _parsnip_lightgbm_compatibility():
        classifier.train(
            combined.copy(),
            num_folds=1,
            labels=np.asarray(combined["type"]).astype(str),
            reweight=True,
            min_child_weight=float(min_child_weight),
        )
    return classifier


def _classification_probabilities(classifications, class_order: Sequence[str]) -> np.ndarray:
    """Extract probabilities from an Astropy table in a fixed class order."""
    missing = set(class_order) - set(classifications.colnames)
    if missing:
        raise ValueError(f"Classifier omitted probability columns: {sorted(missing)}")
    return np.column_stack([np.asarray(classifications[label], dtype=float) for label in class_order])


def standardize_predictions(
    classifications,
    split_manifest: pd.DataFrame,
    config: ExperimentConfig | Mapping[str, Any],
    partition: str = "test",
    class_order: Sequence[str] = FINAL_CLASSES,
) -> pd.DataFrame:
    """Create the model-neutral per-object prediction table."""
    payload = config.normalized() if isinstance(config, ExperimentConfig) else dict(config)
    ids = np.asarray(classifications["object_id"]).astype(str)
    probabilities = _classification_probabilities(classifications, class_order)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("Probability rows must sum to one")

    metadata = split_manifest.set_index("object_id")
    unknown = set(ids) - set(metadata.index.astype(str))
    if unknown:
        raise ValueError(f"Predictions contain unknown objects: {sorted(unknown)[:5]}")
    selected = metadata.loc[ids]
    if (selected["split"] != partition).any():
        raise ValueError(f"Predictions contain objects outside the {partition} partition")

    output = pd.DataFrame({"object_id": ids, "true_class": selected["final_label"].to_numpy()})
    output["predicted_class"] = np.asarray(class_order)[probabilities.argmax(axis=1)]
    for index, label in enumerate(class_order):
        output[f"prob_{label}"] = probabilities[:, index]
    output["backend"] = payload["backend"]
    output["redshift_mode"] = payload["redshift_mode"]
    output["training_sample"] = payload["training_sample"]
    output["evaluation_sample"] = payload["evaluation_sample"]
    output["split_strategy"] = payload["split_strategy"]
    output["run_id"] = payload.get("run_id") or make_run_id(payload)
    return output


def inverse_frequency_weights(labels: Sequence[str]) -> np.ndarray:
    """Return weights giving every represented class equal aggregate weight."""
    labels = np.asarray(labels).astype(str)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = {label: len(labels) / (len(classes) * count) for label, count in zip(classes, counts)}
    return np.asarray([per_class[label] for label in labels], dtype=float)


def class_balanced_log_loss(
    true_labels: Sequence[str],
    probabilities: np.ndarray,
    class_order: Sequence[str] = FINAL_CLASSES,
) -> float:
    """Compute multiclass log loss with equal total weight per represented class."""
    from sklearn.metrics import log_loss

    true_labels = np.asarray(true_labels).astype(str)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return float(
        log_loss(
            true_labels,
            clipped,
            labels=list(class_order),
            sample_weight=inverse_frequency_weights(true_labels),
        )
    )


def compute_classification_metrics(
    predictions: pd.DataFrame,
    class_order: Sequence[str] = FINAL_CLASSES,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Compute the shared scalar, per-class, Brier, and calibration metrics."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    class_order = list(class_order)
    true = predictions["true_class"].to_numpy().astype(str)
    predicted = predictions["predicted_class"].to_numpy().astype(str)
    probabilities = predictions[[f"prob_{label}" for label in class_order]].to_numpy(dtype=float)
    true_indices = np.asarray([class_order.index(label) for label in true])
    one_hot = np.eye(len(class_order))[true_indices]
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=class_order, zero_division=0
    )
    top_two = np.argpartition(probabilities, -2, axis=1)[:, -2:]
    top_two_correct = np.asarray([index in row for index, row in zip(true_indices, top_two)])
    confidence = probabilities.max(axis=1)
    correct = (true == predicted).astype(float)
    bins = np.linspace(0.0, 1.0, calibration_bins + 1)
    bin_index = np.minimum(np.digitize(confidence, bins[1:], right=True), calibration_bins - 1)
    calibration = []
    expected_calibration_error = 0.0
    for index in range(calibration_bins):
        mask = bin_index == index
        if not mask.any():
            continue
        bin_weight = float(mask.mean())
        mean_confidence = float(confidence[mask].mean())
        mean_accuracy = float(correct[mask].mean())
        expected_calibration_error += bin_weight * abs(mean_accuracy - mean_confidence)
        calibration.append(
            {
                "lower": float(bins[index]),
                "upper": float(bins[index + 1]),
                "count": int(mask.sum()),
                "confidence": mean_confidence,
                "accuracy": mean_accuracy,
            }
        )

    return {
        "class_order": class_order,
        "n_objects": int(len(predictions)),
        "class_balanced_log_loss": class_balanced_log_loss(true, probabilities, class_order),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=class_order, average="macro")),
        "top_1_accuracy": float(accuracy_score(true, predicted)),
        "top_2_accuracy": float(top_two_correct.mean()),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error": float(expected_calibration_error),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(class_order)
        },
        "calibration": calibration,
    }


def group_bootstrap_confidence_intervals(
    predictions: pd.DataFrame,
    split_manifest: pd.DataFrame,
    class_order: Sequence[str] = FINAL_CLASSES,
    repeats: int = 1000,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, dict[str, float]]:
    """Bootstrap evaluation groups and return 95% intervals for primary metrics."""
    merged = predictions.merge(split_manifest[["object_id", "group_id"]], on="object_id", how="left")
    if merged["group_id"].isna().any():
        raise ValueError("Every prediction must map to a split group")
    groups = merged["group_id"].unique()
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "class_balanced_log_loss": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "top_2_accuracy": [],
        "multiclass_brier": [],
    }
    grouped = {group: rows for group, rows in merged.groupby("group_id", sort=False)}
    for _ in range(repeats):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        bootstrap = pd.concat([grouped[group] for group in drawn], ignore_index=True)
        metrics = compute_classification_metrics(bootstrap, class_order=class_order)
        for key in samples:
            samples[key].append(metrics[key])
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in samples.items()
    }


def summarize_light_curves(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize cadence, S/N, and survey coverage for stratified evaluation."""
    data = observations.copy()
    data["snr"] = np.abs(data["flux"]) / data["fluxerr"]
    data["survey"] = np.where(data["band"].astype(str).str.startswith("ztf"), "ZTF", "LSST")
    cadence = data.sort_values(["object_id", "mjd"]).groupby("object_id")["mjd"].apply(
        lambda times: float(np.median(np.diff(np.unique(times)))) if times.nunique() > 1 else np.nan
    )
    summary = data.groupby("object_id").agg(
        observations=("mjd", "size"),
        epochs=("mjd", "nunique"),
        median_snr=("snr", "median"),
        max_snr=("snr", "max"),
    )
    coverage = pd.crosstab(data["object_id"], data["survey"])
    for survey in ("ZTF", "LSST"):
        if survey not in coverage:
            coverage[survey] = 0
    summary["median_cadence_days"] = cadence
    summary["ztf_observations"] = coverage["ZTF"]
    summary["lsst_observations"] = coverage["LSST"]
    summary["coverage"] = np.select(
        [
            (summary["ztf_observations"] > 0) & (summary["lsst_observations"] > 0),
            summary["ztf_observations"] > 0,
        ],
        ["ZTF+LSST", "ZTF-only"],
        default="LSST-only",
    )
    return summary.reset_index()


def metric_breakdowns(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    observations: pd.DataFrame,
    class_order: Sequence[str] = FINAL_CLASSES,
) -> pd.DataFrame:
    """Evaluate primary metrics versus redshift, cadence, S/N, and coverage."""
    features = summarize_light_curves(observations).merge(
        truth[["object_id", "z"]], on="object_id", how="left"
    )
    merged = predictions.merge(features, on="object_id", how="left")
    rows = []
    numeric_columns = {
        "redshift": "z",
        "cadence": "median_cadence_days",
        "snr": "median_snr",
    }
    for dimension, column in numeric_columns.items():
        valid = merged[column].notna()
        if valid.sum() < 4:
            continue
        # Quantile bins retain useful counts even when the simulated distributions are skewed.
        bins = pd.qcut(merged.loc[valid, column], q=4, duplicates="drop")
        for interval, indices in bins.groupby(bins, observed=True).groups.items():
            subset = merged.loc[indices]
            metrics = compute_classification_metrics(subset, class_order)
            rows.append(
                {
                    "dimension": dimension,
                    "bin": str(interval),
                    "n_objects": len(subset),
                    "class_balanced_log_loss": metrics["class_balanced_log_loss"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
    for coverage, subset in merged.groupby("coverage", dropna=False):
        metrics = compute_classification_metrics(subset, class_order)
        rows.append(
            {
                "dimension": "coverage",
                "bin": str(coverage),
                "n_objects": len(subset),
                "class_balanced_log_loss": metrics["class_balanced_log_loss"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def configuration_hash(config: Mapping[str, Any]) -> str:
    """Hash an experiment configuration using canonical JSON serialization."""
    # A user-facing run label does not change the scientific configuration identity.
    canonical = {key: value for key, value in config.items() if key != "run_id"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(encoded).hexdigest()[:16]


def make_run_id(config: Mapping[str, Any]) -> str:
    """Create a readable, deterministic run identifier."""
    backend = str(config.get("backend", "run"))
    redshift = str(config.get("redshift_mode", "unknown-z"))
    return f"{backend}_{redshift}_{configuration_hash(config)}"


def run_directory(root: str | Path, config: ExperimentConfig | Mapping[str, Any]) -> Path:
    """Return the standard output directory for an experiment."""
    payload = config.normalized() if isinstance(config, ExperimentConfig) else dict(config)
    run_id = payload.get("run_id") or make_run_id(payload)
    return (
        Path(root)
        / payload["evaluation_sample"]
        / payload["split_strategy"]
        / payload["backend"]
        / run_id
    )


def module_source_provenance(module_name: str) -> dict[str, str]:
    """Return the loaded module path and a deterministic Python-source hash."""

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return {
            f"{module_name}_module_path": "unavailable",
            f"{module_name}_source_hash": "unavailable",
        }
    module_file = Path(module.__file__).resolve()
    source_root = module_file.parent if module_file.name == "__init__.py" else module_file
    sources = (
        sorted(source_root.rglob("*.py"))
        if source_root.is_dir()
        else [source_root]
    )
    digest = sha256()
    for source in sources:
        relative = source.relative_to(source_root) if source_root.is_dir() else Path(source.name)
        digest.update(str(relative).encode())
        digest.update(source.read_bytes())
    return {
        f"{module_name}_module_path": str(module_file),
        f"{module_name}_source_hash": digest.hexdigest()[:16],
    }


def source_versions(extra_packages: Sequence[str] = ("parsnip", "lcdata", "lightgbm")) -> dict[str, str]:
    """Collect reproducibility versions without requiring every optional package."""
    packages = ["numpy", "pandas", "scikit-learn", "astropy", "torch", *extra_packages]
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "local-or-unavailable"
    try:
        package_root = Path(__file__).resolve().parent
        versions["git_commit"] = subprocess.check_output(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        versions["git_commit"] = "unavailable"
    # The distribution name "parsnip" is ambiguous, so fingerprint the module
    # that this kernel actually imported instead of assuming a sibling checkout.
    versions.update(module_source_provenance("parsnip"))
    return versions


def build_experiment_metadata(
    config: ExperimentConfig,
    split_manifest: pd.DataFrame,
    status: str = "configured",
) -> dict[str, Any]:
    """Build the frozen metadata record shared by classifier backends."""
    payload = config.normalized()
    return {
        **payload,
        "configuration_hash": configuration_hash(payload),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "class_order": list(FINAL_CLASSES),
        "split_seed": int(split_manifest["split_seed"].iloc[0]),
        "partition_counts": split_manifest["split"].value_counts().sort_index().to_dict(),
        "source_versions": source_versions(),
        "command": " ".join(sys.argv),
    }


def write_table_once(table: pd.DataFrame, path: str | Path, overwrite: bool = False) -> Path:
    """Write a Parquet artifact and require explicit permission to replace it."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def write_json_once(payload: Mapping[str, Any], path: str | Path, overwrite: bool = False) -> Path:
    """Write a JSON artifact and require explicit permission to replace it."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def assert_comparable_experiments(
    first: ExperimentConfig | Mapping[str, Any],
    second: ExperimentConfig | Mapping[str, Any],
) -> None:
    """Reject direct comparisons with different evaluation samples or split strategies."""
    left = first.normalized() if isinstance(first, ExperimentConfig) else first
    right = second.normalized() if isinstance(second, ExperimentConfig) else second
    mismatches = [
        key
        for key in ("evaluation_sample", "split_strategy")
        if left.get(key) != right.get(key)
    ]
    if mismatches:
        raise ValueError(f"Experiments are not directly comparable; mismatched {mismatches}")


def detect_held_out_group_leakage(
    training_manifest: pd.DataFrame,
    evaluation_manifest: pd.DataFrame,
    evaluation_partition: str = "test",
) -> set[str]:
    """Return groups occurring in both training objects and a frozen evaluation set."""
    training_groups = set(
        training_manifest.loc[training_manifest["split"] == "train", "group_id"].astype(str)
    )
    evaluation_groups = set(
        evaluation_manifest.loc[
            evaluation_manifest["split"] == evaluation_partition, "group_id"
        ].astype(str)
    )
    return training_groups & evaluation_groups


def assert_no_held_out_group_leakage(
    training_manifest: pd.DataFrame,
    evaluation_manifest: pd.DataFrame,
    evaluation_partition: str = "test",
) -> None:
    """Reject a training sample that reuses frozen held-out evaluation groups."""
    leaking = detect_held_out_group_leakage(
        training_manifest, evaluation_manifest, evaluation_partition=evaluation_partition
    )
    if leaking:
        raise ValueError(f"Training sample leaks {len(leaking)} held-out groups")


def estimate_parsnip_training_time(
    smoke_seconds: float,
    smoke_objects: int,
    smoke_epochs: int,
    full_objects: int,
    full_epochs: int,
) -> float:
    """Scale smoke-test timing into a transparent first-order full-run estimate."""
    if min(smoke_seconds, smoke_objects, smoke_epochs, full_objects, full_epochs) <= 0:
        raise ValueError("Timing inputs must be positive")
    return smoke_seconds * (full_objects / smoke_objects) * (full_epochs / smoke_epochs)
