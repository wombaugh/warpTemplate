import importlib.util
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path


_MISSING_MODULE = object()
_ISOLATED_MODULES = (
    "warptemplate",
    "warptemplate.models",
    "warptemplate.loaders",
)


def load_loader_module():
    """Load the coefficient loader with a lightweight fake model module."""

    package = types.ModuleType("warptemplate")
    package_dir = Path(__file__).resolve().parents[1] / "warptemplate"
    package.__path__ = [str(package_dir)]

    models = types.ModuleType("warptemplate.models")

    def fake_get_warpedTimeSeriesModel(**kwargs):
        """Return constructor arguments in place of a heavy sncosmo model."""

        return kwargs

    models.get_warpedTimeSeriesModel = fake_get_warpedTimeSeriesModel

    sys.modules.pop("warptemplate.loaders", None)
    sys.modules["warptemplate"] = package
    sys.modules["warptemplate.models"] = models

    loaders_path = package_dir / "loaders.py"
    spec = importlib.util.spec_from_file_location("warptemplate.loaders", loaders_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["warptemplate.loaders"] = module
    spec.loader.exec_module(module)
    return module


class WarpfitTemplateLoaderTest(unittest.TestCase):
    def setUp(self):
        """Preserve real package modules before installing isolated test doubles."""

        self.saved_modules = {
            name: sys.modules.get(name, _MISSING_MODULE)
            for name in _ISOLATED_MODULES
        }
        self.tmpdir = tempfile.TemporaryDirectory()
        self.coeff_dir = Path(self.tmpdir.name)
        self.module = load_loader_module()
        self.write_coeffs()

    def tearDown(self):
        """Remove fixtures and restore package modules for later test files."""

        self.tmpdir.cleanup()
        for name, module in self.saved_modules.items():
            if module is _MISSING_MODULE:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def write_coeffs(self):
        data = {
            "model_colors": {
                "K": 0.5,
                "loc": 0.7,
                "scale": 0.1,
                "color1": "ztfg",
                "color2": "ztfr",
                "ebv_corr_func": [2.0, 0.0],
            },
            "warpcoeff": {
                "basis_a": [
                    {
                        "model": "tpl_bronze",
                        "z": 0.01,
                        "quality": "bronze",
                        "draw_prob": 1.0,
                        "peakcol": 0.2,
                        "mdict": {"corrmodel": {}},
                    },
                    {
                        "model": "tpl_gold",
                        "z": 0.02,
                        "quality": "gold",
                        "draw_prob": 2.0,
                        "peakcol": 0.4,
                        "mdict": {"corrmodel": {}},
                    },
                ],
                "basis_b": [
                    {
                        "model": "tpl_b",
                        "z": 0.03,
                        "quality": "gold",
                        "draw_prob": 1.0,
                        "peakcol": 0.1,
                        "mdict": {"corrmodel": {}},
                    },
                ],
            },
        }
        with open(self.coeff_dir / "warpcoeffs_v4_SN Test_col.pkl", "wb") as handle:
            pickle.dump(data, handle)

    def test_versioned_path_and_discovery_follow_loader_configuration(self):
        """Loader helpers must use the configured coefficient version and suffix."""

        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))
        self.assertEqual(
            loader.coefficient_path("SN/Test").name,
            "warpcoeffs_v4_SNTest_col.pkl",
        )
        self.assertEqual(loader.available_fitclasses(), ["SN Test"])

    def test_target_color_metadata_and_correction(self):
        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))

        templates = loader.get_templates(
            "SN Test",
            snbasis_selection="all",
            template_selection="all",
            min_fit_quality="gold",
            color_mode="target",
            target_peak_color=0.9,
        )

        self.assertEqual(len(templates), 2)
        first = templates[0]
        self.assertEqual(first["quality"], "gold")
        self.assertEqual(first["target_peak_color"], 0.9)
        self.assertAlmostEqual(first["samplecorr_ebv"], 2.0 * (0.9 - 0.4))
        self.assertEqual(first["model"]["samplecorr_bands"], ["ztfg", "ztfr"])
        self.assertEqual(first["model"]["samplecorr_ebv"], first["samplecorr_ebv"])

    def test_quality_filter_does_not_mutate_cache(self):
        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))

        filtered = loader.get_templates(
            "SN Test",
            snbasis_selection="all",
            template_selection="all",
            min_fit_quality="gold",
        )
        unfiltered = loader.get_templates(
            "SN Test",
            snbasis_selection="all",
            template_selection="all",
        )

        self.assertEqual(len(filtered), 2)
        self.assertEqual(len(unfiltered), 3)

    def test_get_model_colors_returns_copy(self):
        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))

        colors = loader.get_model_colors("SN Test")
        colors["loc"] = 99

        self.assertEqual(loader.get_model_colors("SN Test")["loc"], 0.7)

    def test_target_mode_requires_target_peak_color(self):
        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))

        with self.assertRaises(ValueError):
            loader.get_templates("SN Test", color_mode="target")

    def test_v4_library_supports_every_color_mode_reproducibly(self):
        """All public color modes must work with the v4 `_col` file convention."""

        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))
        options = {
            None: {},
            "harmonize": {},
            "draw": {},
            "target": {"target_peak_color": 0.9},
        }
        for mode, extra in options.items():
            with self.subTest(color_mode=mode):
                first = loader.get_templates(
                    "SN Test",
                    snbasis_selection=4,
                    template_selection=1,
                    random_seed=17,
                    color_mode=mode,
                    **extra,
                )
                second = loader.get_templates(
                    "SN Test",
                    snbasis_selection=4,
                    template_selection=1,
                    random_seed=17,
                    color_mode=mode,
                    **extra,
                )
                self.assertEqual(len(first), 4)
                self.assertEqual(
                    [entry["samplecorr_ebv"] for entry in first],
                    [entry["samplecorr_ebv"] for entry in second],
                )

    def test_default_loader_falls_back_to_flat_v3_library(self):
        """Existing v3 pickles remain readable during the v4 rollout."""

        legacy_dir = self.coeff_dir / "legacy"
        legacy_dir.mkdir()
        with (legacy_dir / "warpcoeffs_v3_SN Legacy.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "basis": [
                        {
                            "model": "legacy-template",
                            "z": 0.1,
                            "quality": "gold",
                            "draw_prob": 1.0,
                            "mdict": {"corrmodel": {}},
                        }
                    ]
                },
                handle,
            )
        loader = self.module.WarpfitTemplateLoader(str(legacy_dir))
        self.assertEqual(loader.available_fitclasses(), ["SN Legacy"])
        self.assertEqual(
            loader.get_templates(
                "SN Legacy",
                snbasis_selection="all",
                template_selection="all",
            )[0]["template_sn"],
            "legacy-template",
        )

    def test_descriptor_draw_does_not_materialize_models(self):
        """Descriptor selection must retain references without calling the model builder."""

        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))
        descriptors = loader.get_template_descriptors(
            "SN Test",
            snbasis_selection=20,
            template_selection=1,
            min_fit_quality="gold",
            random_seed=4,
            color_mode="target",
            target_peak_color=0.9,
        )

        self.assertEqual(len(descriptors), 20)
        self.assertTrue(all("mdict" not in item.to_dict() for item in descriptors))
        self.assertTrue(all(not hasattr(item, "model") for item in descriptors))
        self.assertTrue(all(item.template_key.startswith("SN Test|") for item in descriptors))

    def test_descriptor_can_be_materialized_later(self):
        """Stable descriptor indices must recover the selected coefficient entry."""

        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))
        descriptor = loader.get_template_descriptors(
            "SN Test",
            snbasis_selection=1,
            template_selection=1,
            min_fit_quality="gold",
            random_seed=8,
        )[0]
        record = loader.materialize_descriptor(descriptor)

        self.assertEqual(record["basis_sn"], descriptor.basis_sn)
        self.assertEqual(record["template_sn"], descriptor.template_sn)
        self.assertEqual(record["model"]["original_template_name"], descriptor.template_sn)

    def test_clear_fitclass_cache_is_selective(self):
        """Batch cleanup must remove only the completed fitclass coefficient file."""

        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))
        loader.get_model_colors("SN Test")
        self.assertIn("SN Test", loader._cache)
        loader.clear_fitclass_cache("SN Test")
        self.assertNotIn("SN Test", loader._cache)

    def test_joint_entry_probabilities_preserve_hierarchical_sampling(self):
        """Bases remain uniform while weights act only inside each basis."""

        loader = self.module.WarpfitTemplateLoader(str(self.coeff_dir))
        entries = loader.get_entry_probabilities(
            "SN Test", min_fit_quality="bronze"
        )
        probabilities = {
            descriptor.template_key: probability
            for descriptor, probability in entries
        }
        self.assertAlmostEqual(probabilities["SN Test|basis_a|0"], 1.0 / 6.0)
        self.assertAlmostEqual(probabilities["SN Test|basis_a|1"], 1.0 / 3.0)
        self.assertAlmostEqual(probabilities["SN Test|basis_b|0"], 1.0 / 2.0)


if __name__ == "__main__":
    unittest.main()
