import importlib.util
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path


def load_loader_module():
    package = types.ModuleType("warpTemplate")
    package.__path__ = [str(Path(__file__).resolve().parents[1])]

    models = types.ModuleType("warpTemplate.models")

    def fake_get_warpedTimeSeriesModel(**kwargs):
        return kwargs

    models.get_warpedTimeSeriesModel = fake_get_warpedTimeSeriesModel

    sys.modules.pop("warpTemplate.loaders", None)
    sys.modules["warpTemplate"] = package
    sys.modules["warpTemplate.models"] = models

    loaders_path = Path(__file__).resolve().parents[1] / "loaders.py"
    spec = importlib.util.spec_from_file_location("warpTemplate.loaders", loaders_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["warpTemplate.loaders"] = module
    spec.loader.exec_module(module)
    return module


class WarpfitTemplateLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.coeff_dir = Path(self.tmpdir.name)
        self.module = load_loader_module()
        self.write_coeffs()

    def tearDown(self):
        self.tmpdir.cleanup()

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
        with open(self.coeff_dir / "warpcoeffs_v3_SN Test.pkl", "wb") as handle:
            pickle.dump(data, handle)

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


if __name__ == "__main__":
    unittest.main()
