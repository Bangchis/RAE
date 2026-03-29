from __future__ import annotations

import unittest

try:
    from src.stage2.models.SiT import SiTDH
    from src.stage2.models.lightningDiT import LightningDiT
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    SiTDH = None
    LightningDiT = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing dependency: {_IMPORT_ERROR}")
class Stage2ModelTests(unittest.TestCase):
    def test_sitdh_wraps_lightning_dit_contract(self) -> None:
        model = SiTDH(
            input_size=16,
            patch_size=1,
            in_channels=4,
            hidden_size=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            num_classes=10,
        )

        self.assertIsInstance(model, LightningDiT)
        self.assertEqual(model.hidden_size, 32)
        self.assertEqual(model.depth, 2)
        self.assertEqual(model.num_heads, 4)


if __name__ == "__main__":
    unittest.main()
