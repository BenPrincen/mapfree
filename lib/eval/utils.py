from benchmark.metrics import MetricManager, Inputs
from benchmark.utils import (
    load_poses,
    subsample_poses,
    load_K,
    precision_recall,
    convert_world2cam_to_cam2world,
)
from lib.misc.utils import (
    read_intrinsics,
    read_color_image,
    compute_scene_metrics,
    Pose,
)
from lib.datasets.datamodules import DataModule
from lib.datasets.utils import correct_intrinsic_scale
from lib.models.builder import build_model
from lib.utils.data import data_to_model_device
from transforms3d.quaternions import mat2quat

from collections import defaultdict
import logging
import numpy as np
import os
from pathlib import Path
import torch
from typing import Tuple
from tqdm import tqdm
from yacs.config import CfgNode as CN


class MickeyEvalSession:
    def __init__(
        self,
        config: CN,
        learning_config: str,
        mapfree_config: str = "../config/datasets/mapfree.yaml",
        checkpoint: str = "",
    ) -> None:
        self.config = config
        self.checkpoint = checkpoint
        self.output_path = "runs/"
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.data = None
        if os.path.exists(learning_config):
            self.config.merge_from_file(learning_config)
        else:
            logging.warning(
                f"Learning config file not found at {learning_config}, Skipping."
            )
        if os.path.exists(mapfree_config):
            self.config.merge_from_file(mapfree_config)
        else:
            logging.warning(
                f"Dataset config file not found at {mapfree_config}, Skipping."
            )
        self.model = build_model(config, checkpoint)

    def calculateMatrics(self, results: dict, dataset_path: Path) -> dict:
        scenes = tuple(f.name for f in dataset_path.iterdir() if f.is_dir())
        for scene in scenes:
            pose = results[scene]
            K, W, H = load_K(dataset_path / scene / "intrinsics.txt")
            with (dataset_path / scene / "poses.txt").open(
                "r", encoding="utf-8"
            ) as gt_poses_file:
                gt_poses = load_poses(gt_poses_file, load_confidence=False)

            q, t = convert_world2cam_to_cam2world(pose.q, pose.t)

            poses = {}
            poses[int(pose.image_name[-9:-4])] = ()

    def data_to_cpu(self, data: dict):
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.detach().cpu()
            if k == "inliers_list":
                data[k] = [i.detach().cpu() for i in data[k]]

    def print_memory_usage(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f"total memory: {t}")
        print(f"memory reserved: {r}")
        print(f"memory allocated: {a}")
        print(f"free memory: {f}\n")

    def runModel(self, dataloader: DataModule) -> Tuple[dict, dict]:
        estimated_poses = defaultdict(list)
        pair_data = defaultdict(list)

        for data in tqdm(dataloader):
            data = data_to_model_device(data, self.model)
            with torch.no_grad():
                R_batched, t_batched = self.model(data)

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

                estimated_pose = Pose(
                    image_name=query_img,
                    q=mat2quat(R).reshape(-1),
                    t=t.reshape(-1),
                    inliers=inliers,
                )
                self.data_to_cpu(data)

                estimated_poses[scene].append(estimated_pose)
                pair_data[scene].append(data["inliers_list"])

        return estimated_poses, pair_data

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

    def runOnDataset(self, dataset_path: str, dataset_split: str):

        # assert os.path.exists(Path(dataset_path) / dataset_split)

        self.config.DATASET.DATA_ROOT = dataset_path
        self.dataloader = None

        if dataset_split == "test":
            self.config.TRAINING.BATCH_SIZE = 8
            self.config.TRAINING.NUM_WORKERS = 8
            self.dataloader = DataModule(
                self.config, drop_last_val=False
            ).test_dataloader()
        elif dataset_split == "val":
            self.config.TRAINING.BATCH_SIZE = 16
            self.config.TRAINING.NUM_WORKERS = 8
            self.dataloader = DataModule(
                self.config, drop_last_val=False
            ).val_dataloader()
        else:
            raise NotImplemented(f"Invalid split: {dataset_split}")

        estimated_poses, pair_data = self.runModel(self.dataloader)
        return estimated_poses, pair_data


class MapFreeResult:
    def __init__(self) -> None:
        pass
