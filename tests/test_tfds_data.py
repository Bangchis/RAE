from __future__ import annotations

import unittest

from src_jax.tfds_data import build_label_proxy_dataset, infer_constant_label, infer_default_split, resolve_tfds_dataset_name


class TfdsDataTests(unittest.TestCase):
    def test_celebahq_aliases_resolve_to_expected_dataset(self) -> None:
        self.assertEqual(resolve_tfds_dataset_name("celebahq256"), "celebahq256")
        self.assertEqual(resolve_tfds_dataset_name("celeb_a_hq/256"), "celeb_a_hq/256")

    def test_default_splits_match_known_datasets(self) -> None:
        self.assertEqual(infer_default_split("celebahq256", is_train=True), "train")
        self.assertEqual(infer_default_split("imagenet256", is_train=False), "validation")

    def test_celebahq_uses_constant_zero_labels(self) -> None:
        self.assertEqual(infer_constant_label("celebahq256"), 0)
        dataset = build_label_proxy_dataset(
            num_examples=5,
            image_size=256,
            dataset_name="celebahq256",
            num_classes=1,
        )

        image, label = dataset[3]
        self.assertEqual(tuple(image.shape), (3, 256, 256))
        self.assertEqual(label, 0)

    def test_multiclass_proxy_cycles_labels_when_no_constant_label_exists(self) -> None:
        dataset = build_label_proxy_dataset(
            num_examples=6,
            image_size=32,
            dataset_name="custom-imagenet",
            num_classes=3,
        )

        labels = [dataset[index][1] for index in range(6)]
        self.assertEqual(labels, [0, 1, 2, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
