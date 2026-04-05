from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from src_jax.stage2_runtime import (
        _install_strict_wandb_initializer,
        _load_wandb_resume_metadata,
        _resolve_wandb_resume_binding,
        _sample_initial_latents,
        _wandb_resume_metadata_path,
    )
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    _install_strict_wandb_initializer = None
    _load_wandb_resume_metadata = None
    _resolve_wandb_resume_binding = None
    _sample_initial_latents = None
    _wandb_resume_metadata_path = None
    _IMPORT_ERROR = exc


class _ModelWithSource:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def sample_source_prior(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        self.calls.append(shape)
        return ("source", shape)


class _RngFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return "rng-key"


class _DummyConfig:
    def to_dict(self) -> dict[str, int]:
        return {"seed": 7}


class _FakeWandb:
    def __init__(self, generated_ids: list[str] | None = None) -> None:
        self._generated_ids = list(generated_ids or ["fresh123"])
        self.login_keys: list[str] = []
        self.calls: list[dict[str, object]] = []
        self.util = SimpleNamespace(generate_id=self._generate_id)

    def _generate_id(self) -> str:
        if not self._generated_ids:
            raise AssertionError("No generated W&B run ids left in test double.")
        return self._generated_ids.pop(0)

    def login(self, key: str) -> None:
        self.login_keys.append(key)

    def init(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        run_id = kwargs.get("id")
        if run_id is None and "resume_from" in kwargs:
            run_id = str(kwargs["resume_from"]).split("?", 1)[0]
        return SimpleNamespace(id=run_id)


class _FakeWandbUtils:
    def __init__(self, fake_wandb: _FakeWandb) -> None:
        self.wandb = fake_wandb

    @staticmethod
    def is_main_process() -> bool:
        return True


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing dependency: {_IMPORT_ERROR}")
class Stage2RuntimeTests(unittest.TestCase):
    def test_sample_initial_latents_prefers_source_prior(self) -> None:
        model = _ModelWithSource()
        rngs = _RngFactory()

        class _RandomStub:
            def normal(self, *_args, **_kwargs):  # pragma: no cover - should never run
                raise AssertionError("jax.random.normal should not be used when source prior exists")

        jax = types.SimpleNamespace(random=_RandomStub())
        jnp = types.SimpleNamespace(float32="float32")

        result = _sample_initial_latents(
            model,
            batch_size=2,
            input_size=32,
            in_channels=4,
            rngs=rngs,
            jax=jax,
            jnp=jnp,
        )

        self.assertEqual(result, ("source", (2, 32, 32, 4)))
        self.assertEqual(model.calls, [(2, 32, 32, 4)])
        self.assertEqual(rngs.calls, 0)

    def test_sample_initial_latents_falls_back_to_gaussian_noise(self) -> None:
        rngs = _RngFactory()
        recorded: dict[str, object] = {}

        class _RandomStub:
            def normal(self, key: str, shape: tuple[int, ...], dtype: str) -> tuple[str, str, tuple[int, ...], str]:
                recorded["key"] = key
                recorded["shape"] = shape
                recorded["dtype"] = dtype
                return ("normal", key, shape, dtype)

        jax = types.SimpleNamespace(random=_RandomStub())
        jnp = types.SimpleNamespace(float32="float32")

        result = _sample_initial_latents(
            object(),
            batch_size=3,
            input_size=16,
            in_channels=8,
            rngs=rngs,
            jax=jax,
            jnp=jnp,
        )

        self.assertEqual(result, ("normal", "rng-key", (3, 16, 16, 8), "float32"))
        self.assertEqual(recorded["key"], "rng-key")
        self.assertEqual(recorded["shape"], (3, 16, 16, 8))
        self.assertEqual(recorded["dtype"], "float32")
        self.assertEqual(rngs.calls, 1)


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing optional dependency: {_IMPORT_ERROR}")
class Stage2RuntimeWandbTests(unittest.TestCase):
    def test_new_workdir_generates_unique_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            metadata, init_kwargs = _resolve_wandb_resume_binding(
                workdir=workdir,
                entity="entity",
                project_name="project",
                exp_name="exp-train",
                explicit_run_id=None,
                wandb_utils=SimpleNamespace(wandb=_FakeWandb(["fresh123"])),
            )

            self.assertEqual(init_kwargs, {"id": "fresh123", "resume": "never"})
            self.assertEqual(metadata["run_id"], "fresh123")
            self.assertEqual(metadata["entity"], "entity")
            self.assertEqual(metadata["project"], "project")
            self.assertEqual(metadata["exp_name"], "exp-train")

    def test_legacy_resume_requires_explicit_run_id_when_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000010").mkdir()

            with self.assertRaises(FileNotFoundError):
                _resolve_wandb_resume_binding(
                    workdir=workdir,
                    entity="entity",
                    project_name="project",
                    exp_name="exp-train",
                    explicit_run_id=None,
                    wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
                )

    def test_legacy_resume_accepts_explicit_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000010").mkdir()

            metadata, init_kwargs = _resolve_wandb_resume_binding(
                workdir=workdir,
                entity="entity",
                project_name="project",
                exp_name="exp-train",
                explicit_run_id="legacy123",
                wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
            )

            self.assertEqual(init_kwargs, {"resume_from": "legacy123?_step=10"})
            self.assertEqual(metadata["run_id"], "legacy123")

    def test_stored_metadata_rewinds_to_latest_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000010").mkdir()
            (workdir / "checkpoint_000120").mkdir()
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity", "project": "project-a", "exp_name": "exp-a", "run_id": "saved123"}',
                encoding="utf-8",
            )

            metadata, init_kwargs = _resolve_wandb_resume_binding(
                workdir=workdir,
                entity="entity",
                project_name="project-a",
                exp_name="exp-a",
                explicit_run_id=None,
                wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
            )

            self.assertEqual(init_kwargs, {"resume_from": "saved123?_step=120"})
            self.assertEqual(metadata["run_id"], "saved123")

    def test_resume_binding_detects_entity_project_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000010").mkdir()
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity-a", "project": "project-a", "exp_name": "exp-a", "run_id": "saved123"}',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                _resolve_wandb_resume_binding(
                    workdir=workdir,
                    entity="entity-b",
                    project_name="project-a",
                    exp_name="exp-a",
                    explicit_run_id=None,
                    wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
                )

    def test_resume_binding_requires_checkpoint_when_metadata_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity", "project": "project", "exp_name": "exp-a", "run_id": "saved123"}',
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                _resolve_wandb_resume_binding(
                    workdir=workdir,
                    entity="entity",
                    project_name="project",
                    exp_name="exp-a",
                    explicit_run_id=None,
                    wandb_utils=SimpleNamespace(wandb=_FakeWandb(["unused123"])),
                )

    def test_initializer_persists_metadata_for_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            fake_wandb = _FakeWandb(["fresh999"])
            fake_utils = _FakeWandbUtils(fake_wandb)
            _install_strict_wandb_initializer(
                fake_utils,
                workdir=workdir,
                explicit_run_id=None,
            )

            with patch.dict(
                os.environ,
                {"WANDB_API_KEY": "secret-key", "WANDB_ENTITY": "entity"},
                clear=False,
            ):
                fake_utils.initialize(_DummyConfig(), exp_name="exp-train", project_name="project")

            self.assertEqual(fake_wandb.login_keys, ["secret-key"])
            self.assertEqual(fake_wandb.calls[0]["id"], "fresh999")
            metadata = _load_wandb_resume_metadata(_wandb_resume_metadata_path(workdir))
            self.assertEqual(metadata["run_id"], "fresh999")

    def test_initializer_persists_legacy_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000042").mkdir()
            fake_wandb = _FakeWandb(["unused123"])
            fake_utils = _FakeWandbUtils(fake_wandb)
            _install_strict_wandb_initializer(
                fake_utils,
                workdir=workdir,
                explicit_run_id="legacy999",
            )

            with patch.dict(
                os.environ,
                {"WANDB_API_KEY": "secret-key", "WANDB_ENTITY": "entity"},
                clear=False,
            ):
                fake_utils.initialize(_DummyConfig(), exp_name="exp-train", project_name="project")

            self.assertEqual(fake_wandb.calls[0]["resume_from"], "legacy999?_step=42")
            metadata = _load_wandb_resume_metadata(_wandb_resume_metadata_path(workdir))
            self.assertEqual(metadata["run_id"], "legacy999")

    def test_initializer_uses_resume_from_without_setting_resume_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000100").mkdir()
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity", "project": "project", "exp_name": "exp-train", "run_id": "resume123"}',
                encoding="utf-8",
            )
            fake_wandb = _FakeWandb(["unused123"])
            fake_utils = _FakeWandbUtils(fake_wandb)
            _install_strict_wandb_initializer(
                fake_utils,
                workdir=workdir,
                explicit_run_id=None,
            )

            with patch.dict(
                os.environ,
                {"WANDB_API_KEY": "secret-key", "WANDB_ENTITY": "entity"},
                clear=False,
            ):
                fake_utils.initialize(_DummyConfig(), exp_name="exp-train", project_name="project")

            self.assertEqual(fake_wandb.calls[0]["resume_from"], "resume123?_step=100")
            self.assertNotIn("resume", fake_wandb.calls[0])

    def test_initializer_rejects_explicit_run_id_mismatch_with_saved_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000100").mkdir()
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity", "project": "project", "exp_name": "exp-train", "run_id": "resume123"}',
                encoding="utf-8",
            )
            fake_utils = _FakeWandbUtils(_FakeWandb(["unused123"]))
            _install_strict_wandb_initializer(
                fake_utils,
                workdir=workdir,
                explicit_run_id="other999",
            )

            with self.assertRaises(ValueError):
                with patch.dict(
                    os.environ,
                    {"WANDB_API_KEY": "secret-key", "WANDB_ENTITY": "entity"},
                    clear=False,
                ):
                    fake_utils.initialize(_DummyConfig(), exp_name="exp-train", project_name="project")

    def test_initializer_migrates_legacy_manual_run_id_when_metadata_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workdir = Path(tmp_dir)
            (workdir / "checkpoint_000100").mkdir()
            metadata_path = _wandb_resume_metadata_path(workdir)
            metadata_path.write_text(
                '{"entity": "entity", "project": "project", "exp_name": "exp-train", "run_id": "resume123"}',
                encoding="utf-8",
            )
            fake_wandb = _FakeWandb(["unused123"])
            fake_utils = _FakeWandbUtils(fake_wandb)
            _install_strict_wandb_initializer(
                fake_utils,
                workdir=workdir,
                explicit_run_id="resume123",
            )

            with patch.dict(
                os.environ,
                {"WANDB_API_KEY": "secret-key", "WANDB_ENTITY": "entity"},
                clear=False,
            ):
                fake_utils.initialize(_DummyConfig(), exp_name="exp-train", project_name="project")

            self.assertEqual(fake_wandb.calls[0]["resume_from"], "resume123?_step=100")
