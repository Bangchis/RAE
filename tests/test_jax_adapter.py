from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from src_jax.config_adapter import build_backend_config_dict, load_repo_config, maybe_convert_fid_reference
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    build_backend_config_dict = None
    load_repo_config = None
    maybe_convert_fid_reference = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing optional dependency: {_IMPORT_ERROR}")
class JaxAdapterTests(unittest.TestCase):
    def test_build_backend_config_maps_sitdh_to_lightning_ddt(self) -> None:
        repo_cfg, config_path = load_repo_config("configs/stage2/training/ImageNet256/SiTDH-XL_DINOv2-B.yaml")
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=config_path,
            mode="train",
            data_path="/tmp/imagenet",
            precision="bf16",
            seed=7,
            num_train_samples=1281167,
            enable_eval=False,
        )

        self.assertEqual(backend_cfg["network_class"], "lightning_ddt")
        self.assertEqual(backend_cfg["network"]["num_encoder_blocks"], 28)
        self.assertEqual(backend_cfg["network"]["num_decoder_blocks"], 2)
        self.assertEqual(backend_cfg["network"]["encoder_hidden_size"], 1152)
        self.assertEqual(backend_cfg["network"]["decoder_hidden_size"], 2048)
        self.assertEqual(backend_cfg["network"]["encoder_num_heads"], 16)
        self.assertEqual(backend_cfg["network"]["decoder_num_heads"], 16)
        self.assertEqual(backend_cfg["interface_class"], "sit")
        self.assertEqual(backend_cfg["dtype"], "bfloat16")
        self.assertEqual(backend_cfg["data"]["data_dir"], "/tmp/imagenet")

    def test_npz_fid_reference_is_converted_to_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_path = Path(tmp_dir) / "ref_stats.npz"
            np.savez(ref_path, mu=np.zeros(2048, dtype=np.float64), sigma=np.eye(2048, dtype=np.float64))

            out_path = Path(maybe_convert_fid_reference(str(ref_path)))
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.suffix, ".pkl")

    def test_eval_block_maps_validation_and_fid_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_path = Path(tmp_dir) / "ref_stats.npz"
            np.savez(ref_path, mu=np.zeros(2048, dtype=np.float64), sigma=np.eye(2048, dtype=np.float64))

            repo_cfg, config_path = load_repo_config(
                "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
                overrides=[
                    "eval.data_path=/tmp/imagenet_val",
                    "eval.eval_every=5000",
                    f"eval.fid_ref={ref_path}",
                    "eval.fid_every=25000",
                    "eval.fid_num_samples=4096",
                    "eval.batch_size=8",
                    "eval.num_workers=2",
                    "eval.max_batches=16",
                    "eval.eval_model=true",
                ],
            )
            backend_cfg = build_backend_config_dict(
                repo_cfg,
                config_path=config_path,
                mode="train",
                data_path="/tmp/imagenet_train",
                precision="bf16",
                seed=7,
                num_train_samples=1281167,
                enable_eval=True,
            )

            self.assertTrue(backend_cfg["eval"]["on"])
            self.assertTrue(backend_cfg["eval"]["loss_on"])
            self.assertEqual(backend_cfg["eval"]["data_dir"], "/tmp/imagenet_val")
            self.assertEqual(backend_cfg["eval"]["loss_every_steps"], 5000)
            self.assertEqual(backend_cfg["eval"]["loss_batch_size"], 8)
            self.assertEqual(backend_cfg["eval"]["num_workers"], 2)
            self.assertEqual(backend_cfg["eval"]["max_batches"], 16)
            self.assertTrue(backend_cfg["eval"]["eval_model"])
            self.assertTrue(backend_cfg["eval"]["fid_on"])
            self.assertTrue(str(backend_cfg["data"]["stat_dir"]).endswith(".pkl"))

    def test_infer_network_class_accepts_sitdh_target(self) -> None:
        repo_cfg, config_path = load_repo_config(
            "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
        )
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=config_path,
            mode="train",
            data_path="/tmp/imagenet",
            precision="bf16",
            seed=7,
            num_train_samples=1281167,
            enable_eval=False,
        )

        self.assertEqual(backend_cfg["network_class"], "lightning_ddt")
        self.assertEqual(backend_cfg["network"]["num_encoder_blocks"], 12)
        self.assertEqual(backend_cfg["network"]["num_decoder_blocks"], 2)
        self.assertEqual(backend_cfg["network"]["encoder_hidden_size"], 384)
        self.assertEqual(backend_cfg["network"]["decoder_hidden_size"], 2048)
        self.assertEqual(backend_cfg["network"]["encoder_num_heads"], 6)
        self.assertEqual(backend_cfg["network"]["decoder_num_heads"], 16)


if __name__ == "__main__":
    unittest.main()
