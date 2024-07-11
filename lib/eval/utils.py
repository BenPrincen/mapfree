from transforms3d.quaternions import mat2quat

from collections import defaultdict
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


class MickeyEvalSession:
    def __init__(
        self,
        config: CN,
        learning_config: str,
        checkpoint: str = "",
    ) -> None:
        self._config = config
        self._checkpoint = checkpoint

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        if os.path.exists(learning_config):
            self._config.merge_from_file(learning_config)
        else:
            logging.warning(
                f"Learning config file not found at {learning_config}, Skipping."
            )
        self._model = build_model(config, checkpoint)

    def _data_to_cpu(self, data: dict):
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.detach().cpu()
            if k == "inliers_list":
                data[k] = [i.detach().cpu() for i in data[k]]

    def _print_memory_usage(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f"total memory: {t}")
        print(f"memory reserved: {r}")
        print(f"memory allocated: {a}")
        print(f"free memory: {f}\n")

    def _runModel(self, dataloader: DataLoader) -> dict:
        estimated_poses = defaultdict(list)

        for data in tqdm(dataloader):
            data = data_to_model_device(data, self.model)
            with torch.no_grad():
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

    def runOnPair(
        self,
        query_img: str,
        ref_img: str,
        intrinsics_path: str,
        resize=None,
    ) -> None:

        # gather data and run on model
        K, W, H = load_K(Path("../data/val/s00462") / "intrinsics.txt")
        data = {}
        im0 = read_color_image(ref_img, resize).to(self.device)
        im1 = read_color_image(query_img, resize).to(self.device)
        data["image0"] = im0
        data["image1"] = im1

        frame_nums = [int(query_img[-9:-4]), int(ref_img[-9:-4])]
        for i in frame_nums:
            data["K_color" + str(i)] = (
                torch.from_numpy(K[i]).unsqueeze(0).to(self.device)
            )

        self.model(data)
        self.data = data

        gt_poses_path = os.path.join(
            query_img, "..", "..", "poses.txt"
        )  # assume img in seq folder
        estimated_poses_path = os.path.join(query_img, "..", "..", "poses_device.txt")

        estimated_poses_path = os.path.abspath(estimated_poses_path)
        gt_poses_path = os.path.abspath(gt_poses_path)

        if os.path.exists(gt_poses_path):
            print("ground truth poses exist")
        if os.path.exists(estimated_poses_path):
            print("estimated poses exist")

        with open(gt_poses_path, "r") as gt_poses_file:
            gt_poses = load_poses(gt_poses_file, load_confidence=False)
        with open(estimated_poses_path, "r") as estimated_poses_file:
            estimated_poses = load_poses(estimated_poses_file, load_confidence=False)

        metric_manager = MetricManager()
        gt_poses = subsample_poses(gt_poses, subsample=5)
        failures = 0
        results = defaultdict(list)

        # # calculate metrics
        # for frame_num, (q_gt, t_gt, _) in gt_poses.items():
        #     if frame_num not in estimated_poses:
        #         failures += 1
        #         continue

        #     q_est, t_est = estimated_poses[frame_num]
        #     inputs = Inputs(
        #         q_gt=q_gt,
        #         t_gt=t_gt,
        #         q_est=q_est,
        #         t_est=t_est,
        #         confidence=confidence,
        #         K=K[frame_num],
        #         W=W,
        #         H=H,
        #     )
        #     metric_manager(inputs, results)

        return results, failures

    def runOnSequence(self, dataset_split: str):
        pass

    # assuming validation dataset right now
    def runOnDataset(self, dataloader: DataLoader) -> dict:
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


class MapFreeResult:
    def __init__(self) -> None:
        pass
