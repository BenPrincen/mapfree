from lib.dataset.mapfree import MapFreeDataset
import unittest
from lib.dataset.mapfree import MapFreeDataset
import os
from torch.utils.data import DataLoader
from yacs.config import CfgNode as cfg


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.data_dir = "lib/tests/test_data"
        config_path = os.path.join(self.data_dir, "testset.yaml")
        self.config = cfg()
        self.config.set_new_allowed(True)
        self.config.DEBUG = False

        if os.path.exists(config_path):
            self.config.merge_from_file(config_path)
            # explicitely setting to None because if loading from yaml it's a string
            self.config.DATASET.SCENES = None
            self.config.DATASET.AUGMENTATION_TYPE = None
        else:
            self.config = None

    def test_dataset_creation(self):
        self.assertTrue(self.config)
        dataset = MapFreeDataset(self.config, "val")
        self.assertTrue(isinstance(dataset, MapFreeDataset))
        self.assertEqual(len(dataset), 3)

    def test_data(self):
        dataset = MapFreeDataset(self.config, "val")
        sample = dataset[0]
        self.assertEqual(sample["image0"].shape, (3, 720, 540))
        self.assertEqual(sample["image1"].shape, (3, 720, 540))
        self.assertEqual(sample["T_0to1"].shape, (4, 4))
        self.assertEqual(sample["K_color0"].shape, (3, 3))
        self.assertEqual(sample["K_color1"].shape, (3, 3))

    def test_dataloader(self):
        dataset = MapFreeDataset(self.config, "val")
        dataloader = DataLoader(
            dataset,
            batch_size=1,
        )
        self.assertTrue(isinstance(dataloader, DataLoader))

        counter = 0
        for data in dataloader:
            self.assertEqual(data["image0"].squeeze().shape, (3, 720, 540))
            self.assertEqual(data["image1"].squeeze().shape, (3, 720, 540))
            self.assertEqual(data["T_0to1"].squeeze().shape, (4, 4))
            self.assertEqual(data["K_color0"].squeeze().shape, (3, 3))
            self.assertEqual(data["K_color1"].squeeze().shape, (3, 3))
            counter += 1

        self.assertEqual(counter, 3)
