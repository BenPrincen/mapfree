import numpy as np
import os
from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from typing import Tuple
from lib.find_matches import find_keypoints, find_matches
from lib.find_pose import find_relative_pose
from lib.unproject_points import unproject_points


class SiftRunner(object):
    def __init__(self, config: str) -> None:
        self._config = CN()
        self._config.set_new_allowed(True)
        if os.path.exists(config):
            self._config.merge_from_file(config)
        else:
            raise AttributeError("Config file does not exist")
        self._ransac_iters = self._config.SIFT.RANSAC_ITERATIONS
        self._inlier_thresh = self._config.SIFT.INLIER_THRESHOLD
        self._num_matches = self._config.SIFT.NUM_MATCHES
        self._min_num_matches = self._config.SIFT.MIN_NUM_MATCHES

    def run_once(
        self, img1, img2, img1_depth, img2_depth, camera1, camera2
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run the algorithm on a pair of images"""
        kps = find_keypoints([img1, img2])
        kp1, des1 = kps[0]
        kp2, des2 = kps[1]
        pts1, pts2 = find_matches(
            kp1, des1, kp2, des2, self._config.SIFT.MIN_NUM_MATCHES
        )
        # TODO::: Need to handle the error case of insufficient matching points
        pts1_3d = unproject_points(pts1, img1_depth, camera1)
        pts2_3d = unproject_points(pts2, img2_depth, camera2)
        (R, t), inliers = find_relative_pose(
            pts1_3d,
            pts2_3d,
            ransac_iterations=self._config.SIFT.RANSAC_ITERATIONS,
            inlier_threshold=self._config.SIFT.INLIER_THRESHOLD,
            num_matches=self._config.SIFT.NUM_MATCHES,
        )
        return R, t, inliers

    def run(self, data_loader: DataLoader) -> dict:
        pass
