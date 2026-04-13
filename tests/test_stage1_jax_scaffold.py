from __future__ import annotations

import unittest

try:
    import jax.numpy as jnp

    from src_jax.stage1_jax_config import load_stage1_jax_runtime_config, runtime_config_to_dict
    from src_jax.stage1_jax_state import (
        apply_gradients,
        build_adamw_transform,
        build_learning_rate_schedule,
        create_model_train_state,
    )
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    jnp = None
    load_stage1_jax_runtime_config = None
    runtime_config_to_dict = None
    apply_gradients = None
    build_adamw_transform = None
    build_learning_rate_schedule = None
    create_model_train_state = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing optional dependency: {_IMPORT_ERROR}")
class Stage1JaxScaffoldTests(unittest.TestCase):
    def test_existing_stage1_tfds_config_maps_to_jax_runtime(self) -> None:
        runtime_cfg, _repo_cfg, _cfg_path = load_stage1_jax_runtime_config(
            "configs/stage1/training/CelebAHQ256_DINOv2-B_decXL_tfds.yaml",
        )

        self.assertEqual(runtime_cfg.data.source, "tfds")
        self.assertEqual(runtime_cfg.data.dataset_name, "celebahq256")
        self.assertEqual(runtime_cfg.data.image_size, 256)
        self.assertEqual(runtime_cfg.training.global_batch_size, 512)
        self.assertEqual(runtime_cfg.training.optimizer.lr, 2.0e-4)
        self.assertAlmostEqual(runtime_cfg.gan_loss.disc_weight, 0.75)
        self.assertEqual(runtime_cfg.discriminator.arch.recipe, "S_8")

        summary = runtime_config_to_dict(runtime_cfg)
        self.assertEqual(summary["data"]["dataset_name"], "celebahq256")
        self.assertEqual(summary["training"]["epochs"], 16)

    def test_cosine_schedule_warms_up_and_decays(self) -> None:
        runtime_cfg, _repo_cfg, _cfg_path = load_stage1_jax_runtime_config(
            "configs/stage1/training/CelebAHQ256_DINOv2-B_decXL_tfds.yaml",
        )
        schedule = build_learning_rate_schedule(
            runtime_cfg.training.scheduler,
            steps_per_epoch=10,
            total_epochs=runtime_cfg.training.epochs,
        )

        start = float(schedule(0))
        after_warmup = float(schedule(10))
        final = float(schedule(runtime_cfg.training.epochs * 10))

        self.assertAlmostEqual(start, 0.0, places=8)
        self.assertGreater(after_warmup, final)
        self.assertAlmostEqual(final, runtime_cfg.training.scheduler.final_lr, places=8)

    def test_optax_state_updates_step_and_ema(self) -> None:
        runtime_cfg, _repo_cfg, _cfg_path = load_stage1_jax_runtime_config(
            "configs/stage1/training/CelebAHQ256_DINOv2-B_decXL_tfds.yaml",
        )
        tx = build_adamw_transform(
            runtime_cfg.training.optimizer,
            learning_rate=lambda _step: jnp.asarray(0.1, dtype=jnp.float32),
            clip_grad=0.0,
        )
        params = {"w": jnp.asarray([1.0, -1.0], dtype=jnp.float32)}
        grads = {"w": jnp.asarray([0.5, -0.5], dtype=jnp.float32)}

        state = create_model_train_state(params, tx)
        next_state = apply_gradients(state, grads, tx=tx, ema_decay=0.5)

        self.assertEqual(next_state.step, 1)
        self.assertFalse(jnp.array_equal(next_state.params["w"], state.params["w"]))
        self.assertFalse(jnp.array_equal(next_state.params_ema["w"], state.params_ema["w"]))


if __name__ == "__main__":
    unittest.main()
