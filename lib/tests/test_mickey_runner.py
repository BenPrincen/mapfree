from lib.eval.mickey_runner import MicKeyRunner
from lib.dataset.mapfree import MapFreeDataset
from lib.pose3 import Pose3
import os
from pathlib import Path
from torch.utils.data import DataLoader
import unittest
from yacs.config import CfgNode as cfg
from config.test.test_config import test_config


@unittest.skipIf(test_config.ONLINE, "Should not run online")
class TestMickeyRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = "lib/tests/test_data"
        cls.cl_config_path = "config/MicKey/curriculum_learning.yaml"
        cls.checkpoint_path = "weights/mickey.ckpt"

        dataset_config_path = Path(cls.data_dir) / "testset.yaml"
        cls.config = cfg()
        cls.config.set_new_allowed(True)
        cls.config.DEBUG = False

        if os.path.exists(dataset_config_path):
            cls.config.merge_from_file(dataset_config_path)
            # explicitely setting to None because if loaded from yaml file, they are strings
            cls.config.DATASET.SCENES = None
            cls.config.DATASET.AUGMENTATION_TYPE = None

            dataset = MapFreeDataset(cls.config, "val")
            cls.loader = DataLoader(dataset, batch_size=1)

        else:
            cls.config = None
            cls.loader = None

    def test_runner(self):
        runner = MicKeyRunner(self.cl_config_path, self.checkpoint_path)
        self.assertTrue(runner)
        self.assertTrue(self.config)
        self.assertTrue(self.loader)

        estimated_poses = runner.run(self.loader)
        self.assertEqual(len(estimated_poses), 1)

        for k, v in estimated_poses.items():
            for pose_info in v:
                pose, inliers, frame_num = pose_info
                self.assertTrue(isinstance(pose, Pose3))
                self.assertTrue(isinstance(inliers, float))
                self.assertTrue(isinstance(frame_num, int))
