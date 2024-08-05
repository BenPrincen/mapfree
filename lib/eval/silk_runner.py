import os
from collections import defaultdict

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
from lib.utils.data import data_to_model_device


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

        self._model.eval()

    def run_one(
        self, img1, img2, img1_depth, img2_depth, camera1, camera2, depth_scale
    ) -> tuple:
        # convert to grayscale first
        grayscale = transforms.Compose(
            [
                transforms.Grayscale(
                    num_output_channels=1
                ),  # Convert image to grayscale
            ]
        )
        grayscale_img1 = img1 if img1.shape[0] == 1 else grayscale(img1)
        grayscale_img2 = img2 if img2.shape[0] == 1 else grayscale(img2)
        grayscale_img1 = grayscale_img1.unsqueeze(0)
        grayscale_img2 = grayscale_img2.unsqueeze(0)

        with torch.no_grad():
            sparse_positions_1, sparse_descriptors_1 = self._model(grayscale_img1)
            sparse_positions_2, sparse_descriptors_2 = self._model(grayscale_img2)

        sparse_positions_1 = from_feature_coords_to_image_coords(
            self._model, sparse_positions_1
        )
        sparse_positions_2 = from_feature_coords_to_image_coords(
            self._model, sparse_positions_2
        )

        matches = SILK_MATCHER(sparse_descriptors_1[0], sparse_descriptors_2[0])

        pts1 = sparse_positions_1[0][matches[:, 0]].detach().cpu().numpy()
        pts2 = sparse_positions_2[0][matches[:, 1]].detach().cpu().numpy()
        pts1 = pts1[:, :-1]
        pts2 = pts2[:, :-1]

        img1_depth = img1_depth.detach().cpu().numpy()
        img2_depth = img2_depth.detach().cpu().numpy()
        pts1 = pts1[:, [1, 0]]
        pts2 = pts2[:, [1, 0]]
        pts1_3d = unproject_points(pts1, img1_depth, camera1, depth_scale)
        pts2_3d = unproject_points(pts2, img2_depth, camera2, depth_scale)

        output = find_relative_pose(
            pts1_3d,
            pts2_3d,
            ransac_iterations=self._config.SIFT.RANSAC_ITERATIONS,
            inlier_threshold=self._config.SIFT.INLIER_THRESHOLD,
            num_matches=self._config.SIFT.NUM_MATCHES,
        )
        if output == None:
            return None
        (R, t), inliers, pt1, pt2 = output
        return R, t, inliers

    def run(self, data_loader: DataLoader) -> dict:
        estimated_poses = defaultdict(list)
        for data in tqdm(data_loader):
            data = data_to_model_device(data, self._model)
            img1 = data["image0"].squeeze()
            img2 = data["image1"].squeeze()
            img1_depth = data["depth0"].squeeze()
            img2_depth = data["depth1"].squeeze()
            camera1 = Camera.from_K(
                data["K_color0"].detach().cpu().numpy().squeeze(),
                img1.shape[1],
                img1.shape[0],
            )
            camera2 = Camera.from_K(
                data["K_color1"].detach().cpu().numpy().squeeze(),
                img2.shape[1],
                img2.shape[0],
            )
            frame_num = data["pair_names"][1][0][-9:-4]
            scene = data["scene_id"][0]

            output = self.run_one(
                img1, img2, img1_depth, img2_depth, camera1, camera2, depth_scale=1.0
            )
            if output != None:
                R, t, inliers = output
                r = Rot3(R.squeeze())
                estimated_pose = (Pose3(r, t), float(inliers), int(frame_num))
                estimated_poses[scene].append(estimated_pose)
        return estimated_poses
