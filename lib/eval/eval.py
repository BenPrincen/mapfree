from collections import defaultdict

from transforms3d.quaternions import rotate_vector

from lib.dataset.mapfree import MapFreeDataset
from lib.eval.metrics_manager import Inputs, MetricManager


class Eval:
    def __init__(
        self,
        estimated_poses: list,
        ground_truth_poses: list,
        Ks: list,
        Ws: list,
        Hs: list,
        confidence=False,
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
    def fromMapFree(cls, estimated_poses: dict, dataset: MapFreeDataset):
        """Inputs should be in mapfree format
        estimated_poses:
            keys: scene id's
            values: list of poses where an example of a list elem is [Pose3, confidence, query_img]
        ground_truth_poses:
            keys: scene id's
            values: example of a list elem is [(quaternion), (translation)] without confidence
        """

        def preprocessPosesIntrinsics(estimated_poses: dict) -> dict:
            new_estimated_poses = defaultdict(list)
            for k, v in estimated_poses.items():
                for est_info in v:
                    pose3, conf, frame_num = est_info
                    q, t = pose3.rotation.get_quat().squeeze(), pose3.translation
                    new_estimated_poses[k].append((q, t, conf))
            return new_estimated_poses

        def collectGTFromDataset(
            dataset: MapFreeDataset, estimated_poses: dict
        ) -> tuple:
            """
            1. loop through dataset
            2. extract scene and frame number
            3. check if scene and frame number exist in estimated poses, if so then extract that
            """
            gt_poses = defaultdict(list)
            Ks = defaultdict(list)
            Ws = defaultdict(list)
            Hs = defaultdict(list)
            for data in dataset:
                frame_num = int(data["pair_names"][1][-9:-4])
                scene = scene = data["scene_id"]

                if scene in estimated_poses:
                    for pose in estimated_poses[scene]:
                        _, _, pose_frame_num = pose
                        if frame_num == pose_frame_num:
                            # info to extract: quat, trans, K, W, H
                            q = data["abs_q_1"]
                            c = data["abs_c_1"]
                            t = rotate_vector(-c, q)  # get translation
                            gt_poses[scene].append((q, t))
                            Ks[scene].append(data["K_color1"])
                            Ws[scene].append(data["W"])
                            Hs[scene].append(data["H"])
            return gt_poses, Ks, Ws, Hs

        """est poses, gt poses, and ks have lists stored with each key

            merge lists 
        """

        preprocessed_estimated_poses = preprocessPosesIntrinsics(estimated_poses)
        gt_poses, Ks, Ws, Hs = collectGTFromDataset(dataset, estimated_poses)
        list_est_poses = []
        list_gt_poses = []
        list_ks = []
        list_ws = []
        list_hs = []
        scenes = preprocessed_estimated_poses.keys()
        for scene in scenes:
            l = len(preprocessed_estimated_poses[scene])
            list_est_poses += preprocessed_estimated_poses[scene]
            list_gt_poses += gt_poses[scene]
            list_ks += Ks[scene]
            list_ws += Ws[scene]
            list_hs += Hs[scene]

        return cls(
            list_est_poses, list_gt_poses, list_ks, list_ws, list_hs, confidence=True
        )
