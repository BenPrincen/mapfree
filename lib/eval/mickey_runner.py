import logging
import os
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode as CN

from lib.models.builder import build_model
from lib.pose3 import Pose3
from lib.rot3 import Rot3
from lib.utils.data import data_to_model_device


class MicKeyRunner:
    def __init__(
        self,
        learning_config: str,
        checkpoint: str = "",
    ) -> None:
        self._config = CN()
        self._config.set_new_allowed(True)

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        if os.path.exists(learning_config):
            self._config.merge_from_file(learning_config)
        else:
            logging.warning(
                f"Learning config file not found at {learning_config}, Skipping."
            )

        if os.path.exists(checkpoint):
            self._model = build_model(self._config, checkpoint)
        else:
            logging.warning(f"Checkpoint ")

    # Note, this currently does not work well with a batch size greater than 1
    def _runModel(self, dataloader: DataLoader) -> dict:
        estimated_poses = defaultdict(list)

        for data in tqdm(dataloader):
            data = data_to_model_device(data, self._model)
            with torch.no_grad():
                # data needed to run model
                # image0 and image1 (tensors)
                #
                R_batched, t_batched = self._model(data)

            for i_batch in range(len(data["scene_id"])):
                # refer_fname = data["pair_names"][0][i_batch]
                # query_fname = data["pair_names"][1][i_batch]
                R = R_batched[i_batch].unsqueeze(0).detach().cpu().numpy()
                t = t_batched[i_batch].reshape(-1).detach().cpu().numpy()
                inliers = data["inliers"][i_batch].item()
                scene = data["scene_id"][i_batch]
                frame_num = data["pair_names"][1][i_batch][-9:-4]
                if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                    continue

                r = Rot3(R.squeeze())
                estimated_pose = (Pose3(r, t), inliers, int(frame_num))
                estimated_poses[scene].append(estimated_pose)

        return estimated_poses

    def run_one(
        self, img1, img2, camera1, camera2
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]]:
        # Model expects batches, so make a batch of size 1
        data = {}
        data["image0"] = torch.from_numpy(img1).unsqueeze(0)
        data["image1"] = torch.from_numpy(img2).unsqueeze(0)
        data["K_color0"] = torch.from_numpy(camera1.K).unsqueeze(0)
        data["K_color1"] = torch.from_numpy(camera2.K).unsqueeze(0)

        data = data_to_model_device(data, self._model)
        with torch.no_grad():
            R, t = self._model(data)
        # Move inliers to cpu, and return the 2D point correspondences
        # Format of mickey inliers list seems to be:
        # pt0, pt1, score, depth0, depth1
        inliers = data["inliers_list"][0].cpu().numpy()
        inliers_score = float(data["inliers"].cpu())
        return (
            R.squeeze().cpu().numpy(),
            t.squeeze().cpu().numpy(),
            inliers_score,
            inliers[:, 0:2],
            inliers[:, 2:4],
        )

    # assuming validation dataset right now
    def run(self, dataloader: DataLoader) -> dict:
        """Run MicKey on the entire mapfree validation dataset

        Parameters:
        dataloader -- The torch Dataloader containing the mapfree validation dataset

        Returns:
        Dictionary with keys being the scene id (string) and the values being a tuple.
        The first element in the tuple is a Pose3 representing the estimated pose.
        The second element is a float containing the number of inliers.
        The third element is a string containing the name of the query image.
        """

        estimated_poses = self._runModel(dataloader)
        return estimated_poses
