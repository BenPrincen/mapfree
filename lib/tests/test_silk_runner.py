import os
import unittest

from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN

from config.test.test_config import test_config
from lib.camera import Camera
from lib.common import SILK_MATCHER, get_model
from lib.dataset.mapfree import MapFreeDataset
from lib.eval.silk_runner import SilkRunner
from lib.pose3 import Pose3
from lib.utils.data import data_to_model_device


@unittest.skipIf(test_config.ONLINE, "Should not run online")
class TestSilkRunner(unittest.TestCase):

    @classmethod
    def setUp(cls):

        def initConfig(dataset_config: str) -> CN:
            node = CN()
            node.set_new_allowed(True)
            node.merge_from_file(dataset_config)
            node.DEBUG = False

            # explicitely setting to None because if loading from yaml it's a string
            node.DATASET.SCENES = None
            node.DATASET.AUGMENTATION_TYPE = None
            return node

        cls.sift_config = "config/sift/sift_config.yaml"
        cls.checkpoint = "../silk/assets/models/silk/coco-rgb-aug.ckpt"
        cls.data_dir = "lib/tests/test_data"
        cls.dataset_config = os.path.join(cls.data_dir, "testset.yaml")

        paths = [cls.sift_config, cls.checkpoint, cls.data_dir, cls.dataset_config]
        paths_exist = [os.path.exists(i) for i in paths]

        if sum(paths_exist) < len(paths):
            print("Not all paths exist :(")
            exit(1)  # exit failure

        cls.config = initConfig(cls.dataset_config)
        cls.dataset = MapFreeDataset(cls.config, "val")
        cls.model = get_model(
            checkpoint=cls.checkpoint,
            default_outputs=("sparse_positions", "sparse_descriptors"),
        )

    def test_creation(self):
        silk_runner = SilkRunner(self.sift_config, self.checkpoint)
        self.assertTrue(isinstance(silk_runner, SilkRunner))
        self.assertTrue(isinstance(self.dataset, MapFreeDataset))

    def test_run_one(self):
        silk_runner = SilkRunner(self.sift_config, self.checkpoint)
        data = data_to_model_device(self.dataset[0], self.model)
        img1 = data["image0"]
        img2 = data["image1"]
        depth1 = data["depth0"]
        depth2 = data["depth1"]
        camera1 = Camera.from_K(
            data["K_color0"].detach().cpu().numpy(), img1.shape[1], img1.shape[0]
        )
        camera2 = Camera.from_K(
            data["K_color1"].detach().cpu().numpy(), img2.shape[1], img2.shape[0]
        )
        R, t, inliers = silk_runner.run_one(
            img1, img2, depth1, depth2, camera1, camera2, depth_scale=1.0
        )
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual(t.shape, (3,))
        self.assertTrue(inliers > 0)

    def test_run(self):
        silk_runner = SilkRunner(self.sift_config, self.checkpoint)
        self.assertTrue(silk_runner)
        self.assertTrue(self.dataset)
        self.assertTrue(self.sift_config)

        loader = DataLoader(self.dataset, batch_size=1)
        estimated_poses = silk_runner.run(loader)
        self.assertEqual(len(estimated_poses), 1)

        for k, v in estimated_poses.items():
            for pose_info in v:
                pose, inliers, frame_num = pose_info
                self.assertTrue(isinstance(pose, Pose3))
                self.assertTrue(isinstance(inliers, float))
                self.assertTrue(isinstance(frame_num, int))
