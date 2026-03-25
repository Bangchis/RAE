from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src_jax.vendor import _apply_backend_compat_patches


class JaxVendorPatchTests(unittest.TestCase):
    def test_backend_patches_preserve_model_initialized_ema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend_dir = Path(tmp_dir)
            (backend_dir / "networks" / "encoders").mkdir(parents=True, exist_ok=True)
            (backend_dir / "utils").mkdir(parents=True, exist_ok=True)

            (backend_dir / "networks" / "encoders" / "dino.py").write_text(
                "from transformers import FlaxDinov2Model, AutoImageProcessor\n",
                encoding="utf-8",
            )
            (backend_dir / "networks" / "encoders" / "dino_w_register.py").write_text(
                "from transformers import Dinov2WithRegistersModel\n",
                encoding="utf-8",
            )
            (backend_dir / "networks" / "encoders" / "utils.py").write_text(
                "\"\"\"File containing utility functions for the encoder.\"\"\"\n\n"
                "# built-in libs\n"
                "import math\n\n"
                "# external libs\n"
                "from google.cloud import storage\n\n"
                "def download_blob(bucket_name, source_blob_name, destination_file_name):\n"
                "    \"\"\"Downloads a blob from the bucket.\"\"\"\n"
                "    storage_client = storage.Client()\n"
                "    bucket = storage_client.bucket(bucket_name)\n"
                "    blob = bucket.blob(source_blob_name)\n"
                "    blob.download_to_filename(destination_file_name)\n",
                encoding="utf-8",
            )
            (backend_dir / "utils" / "ema.py").write_text(
                "class EMA:\n"
                "    def __init__(self, net, decay):\n"
                "        self.ema = copy.deepcopy(net)\n"
                "        ema_state = jax.tree.map(lambda x: jnp.zeros_like(x), nnx.state(net, nnx.Param))\n"
                "        nnx.update(self.ema, ema_state)\n"
                "        self.ema.eval()\n"
                "        self.decay = decay\n",
                encoding="utf-8",
            )

            _apply_backend_compat_patches(backend_dir)

            ema_text = (backend_dir / "utils" / "ema.py").read_text(encoding="utf-8")
            self.assertIn("self.ema = copy.deepcopy(net)", ema_text)
            self.assertIn("self.ema.eval()", ema_text)
            self.assertNotIn("jnp.zeros_like", ema_text)
            self.assertNotIn("nnx.update(self.ema, ema_state)", ema_text)


if __name__ == "__main__":
    unittest.main()
