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
    def _write_temp_config(self, text: str, *, filename: str = "config.yaml") -> str:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / filename
        config_path.write_text(text, encoding="utf-8")
        return str(config_path)

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

    def test_train_data_dir_normalizes_split_path_back_to_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "celebahq256_imgfolder"
            for split_name in ("train", "val", "test"):
                (root / split_name / "face").mkdir(parents=True, exist_ok=True)

            repo_cfg, config_path = load_repo_config(
                "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
            )
            backend_cfg = build_backend_config_dict(
                repo_cfg,
                config_path=config_path,
                mode="train",
                data_path=str(root / "train"),
                precision="bf16",
                seed=7,
                num_train_samples=1281167,
                enable_eval=False,
            )

            self.assertEqual(backend_cfg["data"]["data_dir"], str(root.resolve()))

    def test_training_diagnostics_flags_are_forwarded(self) -> None:
        repo_cfg, config_path = load_repo_config(
            "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
            overrides=[
                "training.log_rae_latent_stats=true",
                "training.log_activation_stats=true",
            ],
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

        self.assertTrue(backend_cfg["diagnostics"]["log_rae_latent_stats"])
        self.assertTrue(backend_cfg["diagnostics"]["log_activation_stats"])

    def test_prefetch_factors_are_forwarded(self) -> None:
        repo_cfg, config_path = load_repo_config(
            "configs/stage2/training/ImageNet256/SiTDH-S_DINOv2-B.yaml",
            overrides=[
                "training.prefetch_factor=8",
                "eval.data_path=/tmp/imagenet_val",
                "eval.eval_every=5000",
                "eval.prefetch_factor=3",
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

        self.assertEqual(backend_cfg["data"]["prefetch_factor"], 8)
        self.assertEqual(backend_cfg["eval"]["prefetch_factor"], 3)

    def test_random_flip_flags_are_forwarded(self) -> None:
        config_path = self._write_temp_config(
            """
stage_1:
  target: stage1.RAE
  params:
    encoder_input_size: 256
    decoder_patch_size: 16
    normalization_stat_path: /tmp/stat.pt
    pretrained_encoder_path: facebook/dinov2-base
    pretrained_decoder_path: /tmp/decoder.ckpt
stage_2:
  target: stage2.models.SiT.SiTDH
  params:
    input_size: 16
    patch_size: 1
    in_channels: 768
    hidden_size: [384, 2048]
    depth: [12, 2]
    num_heads: [6, 16]
misc:
  latent_size: [768, 16, 16]
training:
  random_flip: true
eval:
  data_path: /tmp/celebahq256/val
  eval_every: 5000
  random_flip: false
""",
            filename="CelebAHQ256_SiTDH-S_DINOv2-B.yaml",
        )
        repo_cfg, resolved_config_path = load_repo_config(config_path)
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=resolved_config_path,
            mode="train",
            data_path="/tmp/celebahq256_imgfolder",
            precision="bf16",
            seed=7,
            num_train_samples=30_000,
            enable_eval=True,
        )

        self.assertTrue(backend_cfg["data"]["random_flip"])
        self.assertFalse(backend_cfg["eval"]["random_flip"])

    def test_celebahq_configs_default_training_random_flip_to_true(self) -> None:
        config_path = self._write_temp_config(
            """
stage_1:
  target: stage1.RAE
  params:
    encoder_input_size: 256
    decoder_patch_size: 16
    normalization_stat_path: /tmp/stat.pt
    pretrained_encoder_path: facebook/dinov2-base
    pretrained_decoder_path: /tmp/decoder.ckpt
stage_2:
  target: stage2.models.SiT.SiTDH
  params:
    input_size: 16
    patch_size: 1
    in_channels: 768
    hidden_size: [384, 2048]
    depth: [12, 2]
    num_heads: [6, 16]
misc:
  latent_size: [768, 16, 16]
""",
            filename="CelebAHQ256_SiTDH-B_DINOv2-B.yaml",
        )
        repo_cfg, resolved_config_path = load_repo_config(config_path)
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=resolved_config_path,
            mode="train",
            data_path="/tmp/celebahq256_imgfolder",
            precision="bf16",
            seed=7,
            num_train_samples=30_000,
            enable_eval=False,
        )

        self.assertTrue(backend_cfg["data"]["random_flip"])

    def test_lightningdit_target_and_tfds_data_section_are_forwarded(self) -> None:
        config_path = self._write_temp_config(
            """
data:
  format: tfds
  dataset_name: celebahq256
  train_split: train[:95%]
  eval_split: train[95%:]
stage_1:
  target: stage1.RAE
  params:
    encoder_input_size: 256
    decoder_patch_size: 16
    normalization_stat_path: /tmp/stat.pt
    pretrained_decoder_path: /tmp/decoder.ckpt
stage_2:
  target: stage2.models.lightningDiT.LightningDiT
  params:
    input_size: 16
    patch_size: 1
    in_channels: 768
    hidden_size: 1152
    depth: 28
    num_heads: 16
misc:
  latent_size: [768, 16, 16]
  num_classes: 1
training:
  random_flip: true
eval:
  data_path: /tmp/tfds_root
  tfds_split: train[95%:]
  eval_every: 1000
""",
            filename="CelebAHQ256_LightningDiT.yaml",
        )
        repo_cfg, resolved_config_path = load_repo_config(config_path)
        backend_cfg = build_backend_config_dict(
            repo_cfg,
            config_path=resolved_config_path,
            mode="train",
            data_path="/tmp/tfds_root",
            precision="bf16",
            seed=11,
            num_train_samples=30_000,
            enable_eval=True,
        )

        self.assertEqual(backend_cfg["network_class"], "lightning_dit")
        self.assertEqual(backend_cfg["network"]["hidden_size"], 1152)
        self.assertEqual(backend_cfg["network"]["depth"], 28)
        self.assertEqual(backend_cfg["network"]["num_heads"], 16)
        self.assertEqual(backend_cfg["data"]["source"], "tfds")
        self.assertEqual(backend_cfg["data"]["dataset_name"], "celebahq256")
        self.assertEqual(backend_cfg["data"]["train_split"], "train[:95%]")
        self.assertEqual(backend_cfg["eval"]["tfds_split"], "train[95%:]")


if __name__ == "__main__":
    unittest.main()
