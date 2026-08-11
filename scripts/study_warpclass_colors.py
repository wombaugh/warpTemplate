#!/usr/bin/env python
"""Systematic testing of warptemplate model construction parameters.

Tests combinations of template selection, SN basis selection, fit quality tiers,
and color band choices to characterize their impact on EMG fits and E(B-V)
correlations. Designed for batch execution and comparative analysis.
"""

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# Import from your existing module
from templatecreation_III_colors import (
    CLASS_MAP,
    get_class_name,
    register_all,
    run_analysis,
)


# -----------------------------------------------------------------------------
# Test configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TemplateConfig:
    """Immutable configuration for a single warptemplate model test."""
    category: str
    cid: int
    template_selection: str | int
    snbasis_selection: str | int
    min_fit_quality: str | None
    colband: tuple[str, str]
    n_draws: int
    random_state: int

    @property
    def class_name(self) -> str:
        return get_class_name(self.category, self.cid)

    @property
    def colband_str(self) -> str:
        return ",".join(self.colband)

    def to_key(self) -> str:
        """Unique identifier for this configuration."""
        ts = str(self.template_selection).replace("-", "neg")
        ss = str(self.snbasis_selection)
        q = self.min_fit_quality or "none"
        return f"{self.class_name}_ts{ts}_ss{ss}_q{q}_{self.colband[0]}{self.colband[1]}_nd{self.n_draws}_rs{self.random_state}"

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "category": self.category,
            "cid": self.cid,
            "template_selection": self.template_selection,
            "snbasis_selection": self.snbasis_selection,
            "min_fit_quality": self.min_fit_quality,
            "colband1": self.colband[0],
            "colband2": self.colband[1],
            "n_draws": self.n_draws,
            "random_state": self.random_state,
        }


# -----------------------------------------------------------------------------
# Parameter grid generators
# -----------------------------------------------------------------------------

def template_selection_values() -> list[str | int]:
    """Template sampling strategies to test."""
    return ["all", 1, 3, 5, -3, -5]  # all, fixed N, random N


def snbasis_selection_values() -> list[str | int]:
    """SN basis sampling strategies to test."""
    return ["all", 1, 3, 5]


def quality_tier_values(include_none: bool = True) -> list[str | None]:
    """Fit quality thresholds to test."""
    tiers = ["gold", "silver", "bronze"]
    if include_none:
        tiers = [None] + tiers
    return tiers


def color_band_pairs() -> list[tuple[str, str]]:
    """Color band combinations to test."""
    return [
        ("ztfg", "ztfr"),  # g-r optical
        ("ztfg", "ztfi"),  # g-i optical
        ("ztfr", "ztfi"),  # r-i optical
        ("ps1::g", "ps1::r"),  # Pan-STARRS comparison
    ]


def n_draws_values() -> list[int]:
    """E(B-V) sampling densities to test."""
    return [10, 50, 100, 200]


def random_state_values() -> list[int]:
    """Seeds for reproducibility testing."""
    return [41, 42, 123, 999]


# -----------------------------------------------------------------------------
# Grid generation with filtering
# -----------------------------------------------------------------------------

def generate_test_grid(
    categories: list[str] | None = None,
    cids: dict[str, list[int]] | None = None,
    restrict_bands: list[tuple[str, str]] | None = None,
    restrict_qualities: list[str | None] | None = None,
    max_configs: int | None = None,
) -> Iterator[TemplateConfig]:
    """Generate filtered parameter combinations for testing."""

    if categories is None:
        categories = list(CLASS_MAP.keys())

    if cids is None:
        cids = {cat: list(range(len(CLASS_MAP[cat]))) for cat in categories}

    bands = restrict_bands or color_band_pairs()
    qualities = restrict_qualities or quality_tier_values()

    count = 0
    for cat in categories:
        for cid in cids.get(cat, []):
            for ts in template_selection_values():
                for ss in snbasis_selection_values():
                    # Skip incompatible combinations: "all" with large N
                    if ts == "all" and ss != "all":
                        continue  # Test all templates with all SN basis only

                    for q in qualities:
                        for band1, band2 in bands:
                            for nd in n_draws_values():
                                for rs in random_state_values():
                                    # Subsample: test fewer random states for expensive configs
                                    if nd >= 200 and rs != 42:
                                        continue

                                    config = TemplateConfig(
                                        category=cat,
                                        cid=cid,
                                        template_selection=ts,
                                        snbasis_selection=ss,
                                        min_fit_quality=q,
                                        colband=(band1, band2),
                                        n_draws=nd,
                                        random_state=rs,
                                    )
                                    yield config
                                    count += 1
                                    if max_configs and count >= max_configs:
                                        return


def generate_focused_grid(
    class_name: str | None = None,
    vary_one: str | None = None,
    base_config: TemplateConfig | None = None,
) -> Iterator[TemplateConfig]:
    """Vary one parameter while holding others fixed."""

    if base_config is None:
        base_config = TemplateConfig(
            category="n",
            cid=11,  # SN II
            template_selection="all",
            snbasis_selection="all",
            min_fit_quality=None,
            colband=("ztfg", "ztfr"),
            n_draws=100,
            random_state=42,
        )

    if class_name:
        # Find category/cid for class_name
        for cat, classes in CLASS_MAP.items():
            if class_name in classes:
                base_config = TemplateConfig(
                    category=cat,
                    cid=classes.index(class_name),
                    template_selection=base_config.template_selection,
                    snbasis_selection=base_config.snbasis_selection,
                    min_fit_quality=base_config.min_fit_quality,
                    colband=base_config.colband,
                    n_draws=base_config.n_draws,
                    random_state=base_config.random_state,
                )
                break

    vary_map = {
        "template_selection": template_selection_values(),
        "snbasis_selection": snbasis_selection_values(),
        "min_fit_quality": quality_tier_values(),
        "colband": color_band_pairs(),
        "n_draws": n_draws_values(),
        "random_state": random_state_values(),
    }

    if vary_one not in vary_map:
        raise ValueError(f"vary_one must be one of {list(vary_map.keys())}")

    for val in vary_map[vary_one]:
        kwargs = {
            "category": base_config.category,
            "cid": base_config.cid,
            "template_selection": base_config.template_selection,
            "snbasis_selection": base_config.snbasis_selection,
            "min_fit_quality": base_config.min_fit_quality,
            "colband": base_config.colband,
            "n_draws": base_config.n_draws,
            "random_state": base_config.random_state,
        }
        kwargs[vary_one] = val

        yield TemplateConfig(**kwargs)


# -----------------------------------------------------------------------------
# Test execution with error handling
# -----------------------------------------------------------------------------

@dataclass
class TestResult:
    """Outcome of a single configuration test."""
    config: TemplateConfig
    success: bool
    emg_params: dict | None = None
    ebv_poly: list | None = None
    n_templates: int = 0
    n_colors: int = 0
    error: str | None = None
    runtime_seconds: float = 0.0


def run_single_test(
    config: TemplateConfig,
    warpdir: Path,
    outdir: Path,
    version: str = "4",
    toskip: list[str] | None = None,
) -> TestResult:
    """Execute one test configuration with full error capture."""
    import time

    start = time.time()
    toskip = toskip or []

    # Build synthetic args namespace
    class Args:
        pass

    args = Args()
    args.category = config.category
    args.cid = config.cid
    args.warpdir = warpdir
    args.outdir = outdir / config.to_key()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.template_selection = config.template_selection
    args.snbasis_selection = config.snbasis_selection
    args.min_fit_quality = config.min_fit_quality
    args.exclude_input = []
    args.toskip = toskip
    args.colband = config.colband_str
    args.version = version
    args.fit_csv = args.outdir / "emg_fits.csv"
    args.no_progress = True
    args.n_draws = config.n_draws

    try:
        results = run_analysis(args)

        return TestResult(
            config=config,
            success=True,
            emg_params=results['emg_params'],
            ebv_poly=results['ebv_poly'],
            n_templates=results['n_templates'],
            n_colors=results['n_colors'],
            runtime_seconds=time.time() - start,
        )

    except Exception as e:
        return TestResult(
            config=config,
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            runtime_seconds=time.time() - start,
        )


# -----------------------------------------------------------------------------
# Comparative analysis
# -----------------------------------------------------------------------------

def compare_results(results: list[TestResult], outdir: Path) -> pd.DataFrame:
    """Generate comparison table and sensitivity metrics."""

    records = []
    for r in results:
        base = r.config.to_dict()
        base["success"] = r.success
        base["runtime_seconds"] = r.runtime_seconds

        if r.success and r.emg_params:
            base["K"] = r.emg_params["K"]
            base["loc"] = r.emg_params["loc"]
            base["scale"] = r.emg_params["scale"]
            base["n_templates"] = r.n_templates
            base["n_colors"] = r.n_colors

            # E(B-V) polynomial sensitivity: coefficient of variation
            if r.ebv_poly:
                base["ebv_poly_range"] = max(r.ebv_poly) - min(r.ebv_poly)
                base["ebv_poly_mean"] = np.mean(np.abs(r.ebv_poly))
            else:
                base["ebv_poly_range"] = np.nan
                base["ebv_poly_mean"] = np.nan
        else:
            base["K"] = np.nan
            base["loc"] = np.nan
            base["scale"] = np.nan
            base["error"] = r.error

        records.append(base)

    df = pd.DataFrame(records)

    # Save full results
    df.to_csv(outdir / "comparison_results.csv", index=False)

    # Generate sensitivity report
    if len(df) > 1 and df["success"].sum() > 1:
        report = generate_sensitivity_report(df)
        with open(outdir / "sensitivity_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

    return df


def generate_sensitivity_report(df: pd.DataFrame) -> dict:
    """Compute parameter sensitivity metrics from results."""

    report = {
        "n_total": len(df),
        "n_success": int(df["success"].sum()),
        "n_failed": int((~df["success"]).sum()),
        "by_parameter": {},
    }

    # Group by each parameter and compute variance in outputs
    for param in ["template_selection", "snbasis_selection",
                  "min_fit_quality", "colband1", "n_draws", "random_state"]:
        if param not in df.columns:
            continue

        grouped = df[df["success"]].groupby(param).agg({
            "K": ["mean", "std", "count"],
            "loc": ["mean", "std"],
            "scale": ["mean", "std"],
            "n_templates": ["mean", "std"],
        })

        # Flatten column names
        grouped.columns = ["_".join(c).strip() for c in grouped.columns.values]

        report["by_parameter"][param] = {
            "n_groups": len(grouped),
            "K_cv_max": float(grouped["K_std"].max() / grouped["K_mean"].abs().max()) if grouped["K_mean"].abs().max() > 0 else 0,
            "loc_std_max": float(grouped["loc_std"].max()),
            "scale_cv_max": float(grouped["scale_std"].max() / grouped["scale_mean"].max()) if grouped["scale_mean"].max() > 0 else 0,
            "group_summary": grouped.to_dict(),
        }

    # Identify most sensitive parameter
    sensitivities = {
        p: d["K_cv_max"] + d["loc_std_max"]
        for p, d in report["by_parameter"].items()
    }
    if sensitivities:
        report["most_sensitive_parameter"] = max(sensitivities, key=sensitivities.get)

    return report


# -----------------------------------------------------------------------------
# CLI for test execution
# -----------------------------------------------------------------------------

def build_test_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Systematic testing of warptemplate model parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode", choices=["grid", "focused", "single"], default="focused",
        help="Test mode: full grid, vary-one-parameter, or single config"
    )

    # Class selection (same as base)
    g_class = parser.add_argument_group("class selection")
    g_class.add_argument(
        "-c", "--category", choices=["n", "e", "w", "a"], default="n",
    )
    g_class.add_argument("--cid", type=int, default=11)
    g_class.add_argument("--class-name", default=None,
                        help="Override category/cid with explicit class name")

    # Parameter restriction
    g_restrict = parser.add_argument_group("parameter restriction")
    g_restrict.add_argument(
        "--vary-one", default="template_selection",
        choices=["template_selection", "snbasis_selection",
                "min_fit_quality", "colband", "n_draws", "random_state"],
        help="Parameter to vary in focused mode"
    )
    g_restrict.add_argument(
        "--template-selection", default="all",
        help="Fixed value for non-varied parameter"
    )
    g_restrict.add_argument(
        "--snbasis-selection", default="all",
    )
    g_restrict.add_argument(
        "--min-fit-quality", default=None,
        choices=[None, "gold", "silver", "bronze"],
    )
    g_restrict.add_argument(
        "--colband", default="ztfg,ztfr",
    )
    g_restrict.add_argument(
        "--n-draws", type=int, default=100,
    )
    g_restrict.add_argument(
        "--random-state", type=int, default=42,
    )

    # Grid mode options
    g_grid = parser.add_argument_group("grid mode options")
    g_grid.add_argument(
        "--max-configs", type=int, default=None,
        help="Limit total configurations (for testing)"
    )
    g_grid.add_argument(
        "--categories", nargs="+", default=None,
        choices=["n", "e", "w", "a"],
    )

    # Execution
    g_exec = parser.add_argument_group("execution")
    g_exec.add_argument(
        "--warpdir", type=Path,
        default=Path("/Users/jnordin/data/models/sncosmo/warpmod/v4"),
    )
    g_exec.add_argument(
        "--outdir", type=Path, default=Path("./warp_test_results"),
    )
    g_exec.add_argument(
        "--version", default="4",
    )
    g_exec.add_argument(
        "--parallel", type=int, default=1,
        help="Parallel workers (not yet implemented)"
    )
    g_exec.add_argument(
        "--resume", action="store_true",
        help="Skip existing result directories"
    )
    g_exec.add_argument(
        "--dry-run", action="store_true",
        help="Print configurations without executing"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_test_parser()
    args = parser.parse_args(argv)

    register_all()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Build configuration generator
    if args.mode == "single":
        band1, band2 = args.colband.split(",")
        configs = [TemplateConfig(
            category=args.category,
            cid=args.cid,
            template_selection=args.template_selection,
            snbasis_selection=args.snbasis_selection,
            min_fit_quality=args.min_fit_quality,
            colband=(band1, band2),
            n_draws=args.n_draws,
            random_state=args.random_state,
        )]

    elif args.mode == "focused":
        base = TemplateConfig(
            category=args.category,
            cid=args.cid,
            template_selection=args.template_selection,
            snbasis_selection=args.snbasis_selection,
            min_fit_quality=args.min_fit_quality,
            colband=tuple(args.colband.split(",")),
            n_draws=args.n_draws,
            random_state=args.random_state,
        )
        configs = list(generate_focused_grid(
            class_name=args.class_name,
            vary_one=args.vary_one,
            base_config=base,
        ))

    else:  # grid
        cids = None
        if args.categories:
            cids = {cat: list(range(len(CLASS_MAP[cat]))) for cat in args.categories}
        configs = list(generate_test_grid(
            categories=args.categories,
            cids=cids,
            max_configs=args.max_configs,
        ))

    print(f"Generated {len(configs)} test configurations")

    if args.dry_run:
        for cfg in configs:
            print(f"  {cfg.to_key()}")
        return 0

    # Execute tests
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {cfg.to_key()}")

        # Resume check
        test_outdir = args.outdir / cfg.to_key()
        if args.resume and (test_outdir / "comparison_results.csv").exists():
            print("  Skipping (already exists)")
            continue

        result = run_single_test(cfg, args.warpdir, args.outdir, args.version)
        results.append(result)

        if result.success:
            print(f"  Success: K={result.emg_params['K']:.3f}, "
                  f"loc={result.emg_params['loc']:.3f}, "
                  f"n_templates={result.n_templates}")
        else:
            print(f"  FAILED: {result.error[:200]}")

    # Final comparison
    if len(results) > 1:
        df = compare_results(results, args.outdir)
        print(f"\nComparison saved to {args.outdir / 'comparison_results.csv'}")
        print(f"Success rate: {df['success'].mean():.1%}")

        if df["success"].sum() > 1:
            print("\nParameter sensitivity ranking:")
            with open(args.outdir / "sensitivity_report.json") as f:
                report = json.load(f)
            for param, metrics in report.get("by_parameter", {}).items():
                print(f"  {param}: K_cv_max={metrics['K_cv_max']:.3f}, "
                      f"loc_std_max={metrics['loc_std_max']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
