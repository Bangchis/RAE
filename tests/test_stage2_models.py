from __future__ import annotations

import unittest

try:
    from src.stage2.models.SiT import SiTDH
    from src.stage2.models.DDT import DiTwDDTHead
    _IMPORT_ERROR = None
except Exception as exc:
    SiTDH = None
    DiTwDDTHead = None
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"Missing dependency: {_IMPORT_ERROR}")
class Stage2ModelTests(unittest.TestCase):
    def test_sitdh_wraps_ddt_contract(self) -> None:
        model = SiTDH(
            input_size=16,
            patch_size=1,
            in_channels=4,
            hidden_size=[32, 64],
            depth=[2, 1],
            num_heads=[4, 4],
            mlp_ratio=2.0,
            num_classes=10,
        )

        self.assertIsInstance(model, DiTwDDTHead)
        self.assertEqual(model.encoder_hidden_size, 32)
        self.assertEqual(model.decoder_hidden_size, 64)
        self.assertEqual(model.num_encoder_blocks, 2)
        self.assertEqual(model.num_decoder_blocks, 1)


if __name__ == "__main__":
    unittest.main()
