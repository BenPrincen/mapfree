import os
import unittest
from pathlib import Path

from torch.utils.data import DataLoader
from yacs.config import CfgNode as cfg

from config.test.test_config import test_config
from lib.dataset.mapfree import MapFreeDataset
from lib.eval.eval import Eval
from lib.eval.mickey_runner import MicKeyRunner
from lib.gen_utils import load_K, load_poses


@unittest.skipIf(test_config.ONLINE, "Should not run online")
class TestEval(unittest.TestCase):
    @classmethod
    def collectGT(cls, dataset_split: str, scenes: list) -> list:
        # def populate()
        gt_poses_by_scene, K_by_scene, W_by_scene, H_by_scene = {}, {}, {}, {}
        for scene in scenes:
            scene_path = Path(cls.data_dir) / dataset_split / scene
            K, W, H = load_K(scene_path / "intrinsics.txt")
            with (scene_path / "poses.txt").open(
                "r", encoding="utf-8"
            ) as gt_poses_file:
                gt_poses = load_poses(gt_poses_file, load_confidence=False)
            # gt_scene_info = [gt_poses, K, W, H]

            gt_poses_by_scene[scene] = gt_poses
            K_by_scene[scene] = K
            W_by_scene[scene] = W
            H_by_scene[scene] = H
        return (gt_poses_by_scene, K_by_scene, W_by_scene, H_by_scene)
        # gt_info # [gt_poses_by_scene, K_by_scene, W_by_scene, H_by_scene]
        # (gt_poses_by_scene, K_by_scene, W_by_scene, H_by_scene)

    @classmethod
    def setUpClass(cls):
        cls.data_dir = "lib/tests/test_data/mapfree"
        cls.cl_config_path = "config/MicKey/curriculum_learning.yaml"
        cls.checkpoint_path = "weights/mickey.ckpt"

        dataset_config_path = "lib/tests/test_data/testset_nodepth.yaml"
        cls.config = cfg()
        cls.config.set_new_allowed(True)
        cls.config.DEBUG = False

        if os.path.exists(dataset_config_path):
            cls.config.merge_from_file(dataset_config_path)
            # cls.config.DATASET.DATA_ROOT = cls.data_dir
            # explicitely setting to None because if loaded from yaml file, they are strings
            cls.config.DATASET.SCENES = None
            cls.config.DATASET.AUGMENTATION_TYPE = None

            cls.dataset = MapFreeDataset(cls.config, "val")
            cls.loader = DataLoader(cls.dataset, batch_size=1)
            cls.runner = MicKeyRunner(cls.cl_config_path, cls.checkpoint_path)
            cls.estimated_poses = cls.runner.run(cls.loader)

        else:
            cls.config = None
            cls.loader = None

    def test_creation(self):
        self.assertTrue(self.estimated_poses)
        eval = Eval.fromMapFree(self.estimated_poses, self.dataset)
        self.assertTrue(eval)
