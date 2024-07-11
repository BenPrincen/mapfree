from __future__ import annotations

from config.default import cfg
import numpy as np
import os
from pathlib import Path
from typing import Type
import unittest

from lib.dataset.dataset import DataModule


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.data_root = Path("data")
        self.dataset_split = Path("val")
        self.cfg = cfg
        self.cfg.merge_from_file("config/datasets/mapfree.yaml")
        self.cfg.merge_from_file("config/MicKey/curriculum_learning.yaml")

    def test_length(self):
        scenes = os.listdir(self.data_root / self.dataset_split)
        total = 0
        for scene in scenes:
            total += len(os.listdir(self.data_root / self.dataset_split / scene))
        loader = DataModule(self.cfg).val_dataloader()
        self.assertEqual(total, len(loader))


if __name__ == "__main__":
    unittest.main()
