from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from src_jax.export_celebahq_hf import _coerce_image_to_pil, _export_examples, _resolve_example_filename
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

    def test_resolve_example_filename_prefers_explicit_key_then_image_path(self) -> None:
        self.assertEqual(
            _resolve_example_filename({"custom_name": "nested/example.png"}, index=0, filename_key="custom_name"),
            "example.png",
        )
        self.assertEqual(
            _resolve_example_filename({"image": {"path": "nested/from-image.jpg"}}, index=1, filename_key=None),
            "from-image.jpg",
        )

    def test_coerce_image_to_pil_accepts_bytes_dict(self) -> None:
        image = Image.new("RGB", (2, 2), color=(10, 20, 30))
        tmp_path = Path("/tmp/celebahq_hf_test_image.png")
        image.save(tmp_path)
        try:
            payload = {"bytes": tmp_path.read_bytes()}
            converted = _coerce_image_to_pil(payload)
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(converted.mode, "RGB")
        self.assertEqual(converted.size, (2, 2))

    def test_export_examples_writes_imagefolder_tree(self) -> None:
        examples = [
            {"image": Image.new("RGB", (2, 2), color=(255, 0, 0)), "file_name": "000001.png"},
            {"image": Image.new("RGB", (2, 2), color=(0, 255, 0)), "file_name": "000002.png"},
        ]

        with TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            written = _export_examples(
                examples=examples,
                split_name="train",
                output_root=output_root,
                class_name="face",
                overwrite=False,
                image_key="image",
                filename_key=None,
            )

            self.assertEqual(written, 2)
            self.assertTrue((output_root / "train" / "face" / "000001.png").exists())
            self.assertTrue((output_root / "train" / "face" / "000002.png").exists())


if __name__ == "__main__":
    unittest.main()
