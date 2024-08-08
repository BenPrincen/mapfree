import os
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from yacs.config import CfgNode as CN

from lib.camera import Camera
from lib.common import SILK_MATCHER, get_model
from lib.find_pose import find_relative_pose
from lib.pose3 import Pose3
from lib.rot3 import Rot3
from lib.unproject_points import unproject_points


class SilkRunner:
    def __init__(self, config: str, checkpoint: str) -> None:
        self._config = CN()
        self._config.set_new_allowed(True)
        if os.path.exists(config):
            self._config.merge_from_file(config)
        if os.path.exists(checkpoint):
            print("Found checkpoint path.")
            self._model = get_model(
                checkpoint=checkpoint,
                default_outputs=("sparse_positions", "sparse_descriptors"),
            )
        else:
            print("Didn't find checkpoint path. Using default.")
            self._model = get_model(
                default_outputs=("sparse_positions", "sparse_descriptors")
            )  # use default otherwise

        self._ransac_iters = self._config.SILK.RANSAC_ITERATIONS
        self._num_matches = self._config.SILK.NUM_MATCHES
        self._inlier_threshold = self._config.SILK.INLIER_THRESHOLD
        self._min_num_matches = self._config.SILK.MIN_NUM_MATCHES

        self._model.eval()

    def _convert_to_grayscale(self, *args) -> List[torch.Tensor]:
        grayscale = transforms.Compose(
            [
                transforms.Grayscale(
                    num_output_channels=1
                ),  # Convert image to grayscale
            ]
        )
        return [grayscale(img) for img in args]

    def _tensors_to_device(
        self, device: Union[torch.device, str], *args
    ) -> List[torch.Tensor]:
        tensors = [None] * len(args)
        for i, t in enumerate(args):
            if isinstance(t, torch.Tensor):
                tensors[i] = t.to(device)
        return tensors

    def _get_device(self) -> Union[str, torch.device]:
        try:
            device = next(self._model.parameters()).device
        except:
            # in case the model has no parameters (baseline models)
            device = "cpu"
        return device

    def _process_sparse_positions(self, *args) -> List[torch.Tensor]:
        sparse_positions = [None] * len(args)
        for i, positions in enumerate(args):
            sparse_positions[i] = from_feature_coords_to_image_coords(
                self._model, positions
            )
        return sparse_positions

    def _get_cameras(
        self, Ks: List[np.ndarray], Ws: List[int], Hs: List[int]
    ) -> List[Camera]:
        return [Camera.from_K(K, W, H) for K, W, H in zip(Ks, Ws, Hs)]

    def _get_points_from_matches(
        self, pts0: torch.Tensor, pts1: torch.Tensor, matches: torch.Tensor
    ) -> tuple:
        # get matches
        pts0 = pts0[matches[:, 0]]
        pts1 = pts1[matches[:, 1]]
        # remove confidence
        pts0 = pts0[:, :-1]
        pts1 = pts1[:, :-1]

        pts0 = pts0[:, [1, 0]]
        pts1 = pts1[:, [1, 0]]

        return pts0, pts1

    def run_one(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        img1_depth: np.ndarray,
        img2_depth: np.ndarray,
        camera1: Camera,
        camera2: Camera,
        depth_scale: float,
    ) -> tuple:

        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        device = self._get_device()
        img1, img2 = self._tensors_to_device(device, img1, img2)
        # convert to grayscale first
        grayscale = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
            ]
        )
        img1_gray = grayscale(img1).unsqueeze(0)
        img2_gray = grayscale(img2).unsqueeze(0)

        with torch.no_grad():
            sparse_positions_1, sparse_descriptors_1 = self._model(img1_gray)
            sparse_positions_2, sparse_descriptors_2 = self._model(img2_gray)

        sparse_positions_1 = from_feature_coords_to_image_coords(
            self._model, sparse_positions_1
        )
        sparse_positions_2 = from_feature_coords_to_image_coords(
            self._model, sparse_positions_2
        )

        matches = SILK_MATCHER(sparse_descriptors_1[0], sparse_descriptors_2[0])

        pts1, pts2 = self._get_points_from_matches(
            sparse_positions_1[0], sparse_positions_2[0], matches
        )
        pts1 = pts1.detach().cpu().numpy()
        pts2 = pts2.detach().cpu().numpy()
        pts1_3d = unproject_points(pts1, img1_depth, camera1, depth_scale)
        pts2_3d = unproject_points(pts2, img2_depth, camera2, depth_scale)

        output = find_relative_pose(
            pts1_3d,
            pts2_3d,
            ransac_iterations=self._ransac_iters,
            inlier_threshold=self._inlier_threshold,
            num_matches=self._num_matches,
        )
        if output == None:
            return None
        (R, t), inliers, _, _ = output
        return R, t, inliers

    def run(self, loader: DataLoader) -> dict:
        needed_keys = [
            "image0",
            "image1",
            "depth0",
            "depth1",
            "K_color0",
            "K_color1",
            "pair_names",
            "scene_id",
        ]
        estimated_poses = defaultdict(list)
        for data in tqdm(loader):

            # get data from dictionary
            img1, img2, img1_depth, img2_depth, Ks1, Ks2, pair_names, scenes = [
                data[k] for k in needed_keys
            ]

            device = self._get_device()
            img1, img2 = self._tensors_to_device(device, img1, img2)
            gray_img1, gray_img2 = self._convert_to_grayscale(img1, img2)

            img1_depth = img1_depth.detach().numpy()
            img2_depth = img2_depth.detach().numpy()

            with torch.no_grad():
                output1 = self._model(gray_img1)
                output2 = self._model(gray_img2)

            sparse_positions1, sparse_descriptors1 = output1
            sparse_positions2, sparse_descriptors2 = output2

            sparse_positions1 = from_feature_coords_to_image_coords(
                self._model, sparse_positions1
            )
            sparse_positions2 = from_feature_coords_to_image_coords(
                self._model, sparse_positions2
            )

            for i_batch in range(len(scenes)):
                Ks = [Ks1[i_batch].detach().numpy(), Ks2[i_batch].detach().numpy()]
                Ws = [gray_img1.shape[1], gray_img2.shape[1]]
                Hs = [gray_img1.shape[0], gray_img2.shape[0]]
                camera1, camera2 = self._get_cameras(Ks, Ws, Hs)

                frame_num = int(pair_names[1][i_batch][-9:-4])
                scene = scenes[i_batch]

                matches = SILK_MATCHER(
                    sparse_descriptors1[i_batch], sparse_descriptors2[i_batch]
                )
                pts1, pts2 = self._get_points_from_matches(
                    sparse_positions1[i_batch], sparse_positions2[i_batch], matches
                )
                pts1 = pts1.detach().cpu().numpy()
                pts2 = pts2.detach().cpu().numpy()
                pts1_3d = unproject_points(
                    pts1, img1_depth[i_batch], camera1, depth_scale=1.0
                )
                pts2_3d = unproject_points(
                    pts2, img2_depth[i_batch], camera2, depth_scale=1.0
                )
                output = find_relative_pose(
                    pts1_3d,
                    pts2_3d,
                    ransac_iterations=self._ransac_iters,
                    inlier_threshold=self._inlier_threshold,
                    num_matches=self._num_matches,
                )
                if output != None:
                    (R, t), inliers, _, _ = output
                    r = Rot3(R.squeeze())
                    estimated_pose = (Pose3(r, t), float(inliers), int(frame_num))
                    estimated_poses[scene].append(estimated_pose)
        return estimated_poses
