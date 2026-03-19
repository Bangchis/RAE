from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from src_jax.stage1_runtime import finalize_latent_stats, list_image_files
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    finalize_latent_stats = None
    list_image_files = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing dependency: {_IMPORT_ERROR}")
class Stage1RuntimeTests(unittest.TestCase):
    def test_finalize_latent_stats_matches_manual_values(self) -> None:
        latents = np.asarray(
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[5.0, 6.0], [7.0, 8.0]]],
            ],
            dtype=np.float32,
        )
        latent_sum = latents.sum(axis=0, dtype=np.float64)
        latent_sumsq = np.square(latents, dtype=np.float64).sum(axis=0, dtype=np.float64)

        mean, var = finalize_latent_stats(latent_sum, latent_sumsq, count=latents.shape[0])

        np.testing.assert_allclose(mean, latents.mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(var, latents.var(axis=0), atol=1e-6)

    def test_list_image_files_recurses_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a").mkdir()
            (root / "b").mkdir()
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / "b" / "second.png")
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / "a" / "first.jpg")

            image_paths = list_image_files(root)

            self.assertEqual(
                image_paths,
                sorted([root / "a" / "first.jpg", root / "b" / "second.png"]),
            )


if __name__ == "__main__":
    unittest.main()
