from lib.gen_utils import load_poses, load_K
from lib.dataset.mapfree import MapFreeDataset
from lib.eval.mickey_runner import MicKeyRunner
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN
from collections import defaultdict
from lib.eval.mickey_runner import MicKeyRunner
from lib.dataset.mapfree import MapFreeDataset
from lib.pose3 import Pose3
from lib.eval.utils import MetricManager, Inputs
import os
from pathlib import Path
from torch.utils.data import DataLoader
import unittest
from yacs.config import CfgNode as cfg
import torch
import numpy as np


class Eval:
    def __init__(
        self,
        estimated_poses: list,
        ground_truth_poses: list,
        Ks: list,
        Ws: list,
        Hs: list,
    ):
        """estimated_poses element format: (quaternion), (translation), confidence (optional)
        ground_truth_poses element format: (quaternion), (translation)
        Ks: lists of intrinsics
        Ws: list of widths
        Hs: list of heights
        """

        results = defaultdict(list)
        metricmanager = MetricManager()
        for i in range(len(estimated_poses)):
            q_est, t_est, confidence = estimated_poses[i]
            q_gt, t_gt = ground_truth_poses[i]
            inputs = Inputs(
                q_gt=q_gt,
                t_gt=t_gt,
                q_est=q_est,
                t_est=t_est,
                confidence=confidence,
                K=Ks[i],
                W=Ws[i],
                H=Hs[i],
            )
            metricmanager(inputs, results)
        self._results = results

    @property
    def results(self):
        return self._results

    @classmethod
    def fromMapFree(
        cls,
        estimated_poses: dict,
        ground_truth_poses: dict,
        Ks: dict,
        Ws: float,
        Hs: float,
    ):
        """Inputs should be in mapfree format
        estimated_poses:
            keys: scene id's
            values: list of poses where an example of a list elem is [Pose3, confidence, query_img]
        ground_truth_poses:
            keys: scene id's
            values: example of a list elem is [(quaternion), (translation)] without confidence
        """

        def preprocessPosesIntrinsics(
            estimated_poses: dict, gt_poses: dict, Ks: dict
        ) -> dict:
            new_estimated_poses = defaultdict(list)
            new_gt_poses = defaultdict(list)
            new_Ks = defaultdict(list)
            for k, v in estimated_poses.items():
                for est_info in v:
                    pose3, conf, frame_num = est_info
                    q, t = pose3.rotation.getQuat().squeeze(), pose3.translation
                    new_estimated_poses[k].append((q, t, conf))
                    q, t, _ = gt_poses[k][frame_num]
                    new_gt_poses[k].append((q, t))
                    new_Ks[k].append(Ks[k][frame_num])
                    # print(f'pose3: {pose3}, conf: {conf}, frame_num: {frame_num}')
                # est_info = [
                #     [pose3.rotation.getQuat(), pose3.translation, conf] for pose3, conf, _ in v
                # ]
                # pose3, conf, _ = v
                # new_estimated_poses[k] = est_info
            return new_estimated_poses, new_gt_poses, new_Ks

        """est poses, gt poses, and ks have lists stored with each key

            merge lists 
        """

        preprocessed_estimated_poses, preprocessed_gt_poses, preprocessed_ks = (
            preprocessPosesIntrinsics(estimated_poses, ground_truth_poses, Ks)
        )
        list_est_poses = []
        list_gt_poses = []
        list_ks = []
        list_ws = []
        list_hs = []
        scenes = preprocessed_estimated_poses.keys()
        for scene in scenes:
            print(f"K: {preprocessed_ks[scene]}")
            l = len(preprocessed_estimated_poses[scene])
            list_est_poses += preprocessed_estimated_poses[scene]
            list_gt_poses += preprocessed_gt_poses[scene]
            list_ks += preprocessed_ks[scene]
            list_ws += [Ws[scene]] * l
            list_hs += [Hs[scene]] * l

        return cls(list_est_poses, list_gt_poses, list_ks, list_ws, list_hs)
