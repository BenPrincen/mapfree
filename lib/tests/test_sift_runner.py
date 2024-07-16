import unittest
import os
from lib.eval.sift_runner import SiftRunner
from yacs.config import CfgNode as cfg
from lib.dataset.mapfree import MapFreeDataset
from lib.camera import Camera
import numpy as np


class TestSiftRunner(unittest.TestCase):
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
        self._dataset = MapFreeDataset(self.config, "val")

    def test_creation(self):
        config_path = os.path.join("config/sift", "sift_config.yaml")
        sift_runner = SiftRunner(config_path)
        self.assertTrue(isinstance(sift_runner, SiftRunner))

    def test_run_one(self):
        config_path = os.path.join("config/sift", "sift_config.yaml")
        sift_runner = SiftRunner(config_path)
        # Get an item from the data set
        data = self._dataset[0]
        img1 = (np.transpose(data["image0"].numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        img2 = (np.transpose(data["image1"].numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        camera1 = Camera.from_K(data["K_color0"].numpy(), img1.shape[1], img1.shape[0])
        camera2 = Camera.from_K(data["K_color1"].numpy(), img2.shape[1], img2.shape[0])
        R, t, inliers = sift_runner.run_one(
            img1,
            img2,
            data["depth0"],
            data["depth1"],
            camera1,
            camera2,
        )
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual(t.shape, (3,))
        self.assertTrue(inliers > 0)

    def test_run(self):
        pass


unittest.main(argv=[""], verbosity=2, exit=False)
