from transforms3d.quaternions import mat2quat

from collections import defaultdict
from lib.dataset.mapfree import MapFreeDataset
from lib.utils.data import data_to_model_device
from lib.models.builder import build_model
from lib.pose3 import Pose3
from lib.rot3 import Rot3
import logging
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
from yacs.config import CfgNode as CN


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
                query_img = data["pair_names"][1][i_batch]
                if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                    continue

                r = Rot3(R)
                estimated_pose = (Pose3(r, t), inliers, query_img)
                estimated_poses[scene].append(estimated_pose)

        return estimated_poses

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
