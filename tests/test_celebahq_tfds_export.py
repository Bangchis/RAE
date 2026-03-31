from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src_jax.export_celebahq_tfds import (
    _configure_tfds_runtime,
    build_split_specs,
    normalize_example_filename,
)


class CelebAHQExportTests(unittest.TestCase):
    def test_configure_tfds_runtime_defaults_to_python_protobuf_runtime(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _configure_tfds_runtime()

            self.assertEqual(os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"], "python")

    def test_configure_tfds_runtime_preserves_existing_protobuf_runtime(self) -> None:
        with patch.dict(os.environ, {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "cpp"}, clear=True):
            _configure_tfds_runtime()

            self.assertEqual(os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"], "cpp")

    def test_build_split_specs_uses_train_val_test_percentages(self) -> None:
        split_specs = build_split_specs(train_percent=90, val_percent=5)

        self.assertEqual(split_specs["train"], "train[:90%]")
        self.assertEqual(split_specs["val"], "train[90%:95%]")
        self.assertEqual(split_specs["test"], "train[95%:]")

    def test_build_split_specs_rejects_invalid_percentages(self) -> None:
        with self.assertRaises(ValueError):
            build_split_specs(train_percent=95, val_percent=5)

    def test_normalize_example_filename_handles_bytes_and_missing_suffix(self) -> None:
        self.assertEqual(normalize_example_filename(b"000123", index=0), "000123.png")
        self.assertEqual(normalize_example_filename("nested/path/000124.jpg", index=0), "000124.jpg")


if __name__ == "__main__":
    unittest.main()
