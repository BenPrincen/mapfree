# Code from Mickey repository https://github.com/nianticlabs/mickey

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, Grayscale
from transforms3d.quaternions import qinverse, qmult, quat2mat, rotate_vector

from lib.dataset.sampler import RandomConcatSampler
from lib.dataset.utils import (
    correct_intrinsic_scale,
    read_color_image,
    read_depth_image,
)


class DataModuleTraining(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.seed = cfg.DATASET.SEED

        datasets = {"MapFree": MapFreeDataset}

        assert (
            cfg.DATASET.DATA_SOURCE in datasets.keys()
        ), "invalid DATA_SOURCE, this dataset is not implemented"
        self.dataset_type = datasets[cfg.DATASET.DATA_SOURCE]

    def get_sampler(self, dataset, reset_epoch=False, seed=66):
        if self.cfg.TRAINING.SAMPLER == "scene_balance":
            sampler = RandomConcatSampler(
                dataset,
                self.cfg.TRAINING.N_SAMPLES_SCENE,
                self.cfg.TRAINING.SAMPLE_WITH_REPLACEMENT,
                shuffle=True,
                reset_on_iter=reset_epoch,
                seed=seed,
            )
        else:
            sampler = None
        return sampler

    def train_dataloader(self):
        transforms = (
            ColorJitter()
            if self.cfg.DATASET.AUGMENTATION_TYPE == "colorjitter"
            else None
        )
        transforms = (
            Grayscale(num_output_channels=3)
            if self.cfg.DATASET.BLACK_WHITE
            else transforms
        )

        dataset = self.dataset_type(self.cfg, "train", transforms=transforms)
        sampler = self.get_sampler(dataset, seed=self.seed)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.TRAINING.BATCH_SIZE,
            num_workers=self.cfg.TRAINING.NUM_WORKERS,
            sampler=sampler,
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset_type(self.cfg, "val")
        sampler = self.get_sampler(dataset, reset_epoch=True)
        # sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.TRAINING.BATCH_SIZE,
            num_workers=self.cfg.TRAINING.NUM_WORKERS,
            sampler=sampler,
            drop_last=True,
        )
        return dataloader

    def test_dataloader(self):
        dataset = self.dataset_type(self.cfg, "test")
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.TRAINING.BATCH_SIZE,
            num_workers=self.cfg.TRAINING.NUM_WORKERS,
            shuffle=False,
        )
        return dataloader


class MapFreeScene(data.Dataset):

    def __init__(
        self,
        scene_root,
        scene_depth_root,
        resize,
        sample_factor=1,
        overlap_limits=None,
        transforms=None,
        test_scene=False,
    ):
        super().__init__()

        self.scene_root = Path(scene_root)
        if scene_depth_root is not None:
            self.scene_depth_root = Path(scene_depth_root)
        else:
            self.scene_depth_root = None
        self.resize = resize
        self.sample_factor = sample_factor
        self.transforms = transforms
        self.test_scene = test_scene

        # load absolute poses
        self.poses = self.read_poses(self.scene_root)

        # read intrinsics
        self.K, self.K_ori, self.W, self.H = self.read_intrinsics(
            self.scene_root, resize
        )

        # load pairs
        self.pairs = self.load_pairs(
            self.scene_root, overlap_limits, self.sample_factor
        )

    @staticmethod
    def read_intrinsics(scene_root: Path, resize=None):
        Ks = {}
        K_ori = {}
        with (scene_root / "intrinsics.txt").open("r") as f:
            for line in f.readlines():
                if "#" in line:
                    continue

                line = line.strip().split(" ")
                img_name = line[0]
                fx, fy, cx, cy, W, H = map(float, line[1:])

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                K_ori[img_name] = K
                if resize is not None:
                    K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
                Ks[img_name] = K
        return Ks, K_ori, W, H

    @staticmethod
    def read_poses(scene_root: Path):
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        poses = {}
        with (scene_root / "poses.txt").open("r") as f:
            for line in f.readlines():
                if "#" in line:
                    continue

                line = line.strip().split(" ")
                img_name = line[0]
                qt = np.array(list(map(float, line[1:])))
                poses[img_name] = (qt[:4], qt[4:])
        return poses

    def load_pairs(
        self, scene_root: Path, overlap_limits: tuple = None, sample_factor: int = 1
    ):
        """
        For training scenes, filter pairs of frames based on overlap (pre-computed in overlaps.npz)
        For test/val scenes, pairs are formed between keyframe and every other sample_factor query frames.
        If sample_factor == 1, all query frames are used. Note: sample_factor applicable only to test/val
        Returns:
        pairs: nd.array [Npairs, 4], where each column represents seaA, imA, seqB, imB, respectively
        """
        pairs = self.load_pairs_overlap(scene_root, overlap_limits, sample_factor)

        return pairs

    def load_pairs_overlap(
        self, scene_root: Path, overlap_limits: tuple = None, sample_factor: int = 1
    ):
        overlaps_path = scene_root / "overlaps.npz"

        if overlaps_path.exists():
            f = np.load(overlaps_path, allow_pickle=True)
            idxs, overlaps = f["idxs"], f["overlaps"]
            if overlap_limits is not None:
                min_overlap, max_overlap = overlap_limits
                mask = (overlaps > min_overlap) * (overlaps < max_overlap)
                idxs = idxs[mask]
                return idxs.copy()
        else:
            idxs = np.zeros((len(self.poses) - 1, 4), dtype=np.uint16)
            idxs[:, 2] = 1
            idxs[:, 3] = np.array(
                [int(fn[-9:-4]) for fn in self.poses.keys() if "seq0" not in fn],
                dtype=np.uint16,
            )
            return idxs[::sample_factor]

    def get_pair_path(self, pair):
        seqA, imgA, seqB, imgB = pair
        return (f"seq{seqA}/frame_{imgA:05}.jpg", f"seq{seqB}/frame_{imgB:05}.jpg")

    def get_mickey_depth_pair_path(self, pair):
        seqA, imgA, seqB, imgB = pair
        return (
            f"seq{seqA}/frame_{imgA:05}.mickey.png",
            f"seq{seqB}/frame_{imgB:05}.mickey.png",
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # image paths (relative to scene_root)
        im1_path, im2_path = self.get_pair_path(self.pairs[index])
        # load color images
        image1 = read_color_image(
            self.scene_root / im1_path, self.resize, augment_fn=self.transforms
        )
        image2 = read_color_image(
            self.scene_root / im2_path, self.resize, augment_fn=self.transforms
        )
        if self.scene_depth_root is not None:
            depth1_path, depth2_path = self.get_mickey_depth_pair_path(
                self.pairs[index]
            )
            depth1 = read_depth_image(self.scene_depth_root / depth1_path)
            depth2 = read_depth_image(self.scene_depth_root / depth2_path)

        # get absolute pose of im0 and im1
        if self.test_scene:
            t1, t2, c1, c2 = np.zeros([3]), np.zeros([3]), np.zeros([3]), np.zeros([3])
            q1, q2 = np.zeros([4]), np.zeros([4])
            T = np.zeros([4, 4])
        else:
            # quaternion and translation vector that transforms World-to-Cam
            q1, t1 = self.poses[im1_path]
            # quaternion and translation vector that transforms World-to-Cam
            q2, t2 = self.poses[im2_path]
            c1 = rotate_vector(
                -t1, qinverse(q1)
            )  # center of camera 1 in world coordinates)
            c2 = rotate_vector(
                -t2, qinverse(q2)
            )  # center of camera 2 in world coordinates)

            # get 4 x 4 relative pose transformation matrix (from im1 to im2)
            # for val set, q1,t1 is the identity pose, so the relative pose matches the absolute pose
            q12 = qmult(q2, qinverse(q1))
            t12 = t2 - rotate_vector(t1, q12)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = quat2mat(q12)
            T[:3, -1] = t12

        T = torch.from_numpy(T)

        data = {
            "image0": image1,  # (3, h, w)
            "image1": image2,
            "T_0to1": T,  # (4, 4)  # relative pose
            "abs_q_0": q1,
            "abs_t_0": t1,
            "abs_c_0": c1,
            "abs_q_1": q2,
            "abs_t_1": t2,
            "abs_c_1": c2,
            "K_color0": self.K[im1_path],  # (3, 3)
            "Kori_color0": self.K_ori[im1_path],  # (3, 3)
            "K_color1": self.K[im2_path],  # (3, 3)
            "Kori_color1": self.K_ori[im2_path],  # (3, 3)
            "W": self.W,
            "H": self.H,
            "dataset_name": "Mapfree",
            "scene_id": self.scene_root.stem,
            "scene_root": str(self.scene_root),
            "pair_id": index * self.sample_factor,
            "pair_names": (im1_path, im2_path),
        }
        if self.scene_depth_root is not None:
            data["depth0"] = depth1  # (1, h, w)
            data["depth1"] = depth2

        return data


class MapFreeDataset(data.ConcatDataset):

    def __init__(self, cfg, mode, transforms=None):
        assert mode in ["train", "val", "test"], "Invalid dataset mode"

        data_root = Path(cfg.DATASET.DATA_ROOT) / mode
        if hasattr(cfg.DATASET, "DEPTH_ROOT"):
            depth_root = Path(cfg.DATASET.DEPTH_ROOT) / mode
        else:
            depth_root = None
        resize = (cfg.DATASET.WIDTH, cfg.DATASET.HEIGHT)

        if mode == "test":
            test_scene = True
        else:
            test_scene = False

        overlap_limits = (cfg.DATASET.MIN_OVERLAP_SCORE, cfg.DATASET.MAX_OVERLAP_SCORE)
        sample_factor = {"train": 1, "val": 5, "test": 5}[mode]

        scenes = cfg.DATASET.SCENES
        if scenes is None:
            # Locate all scenes of the current dataset
            scenes = [s.name for s in data_root.iterdir() if s.is_dir()]

        if cfg.DEBUG:
            if mode == "train":
                scenes = scenes[:30]
            elif mode == "val":
                scenes = scenes[:10]

        # Init dataset objects for each scene
        data_srcs = [
            MapFreeScene(
                data_root / scene,
                depth_root / scene if depth_root is not None else None,
                resize,
                sample_factor,
                overlap_limits,
                transforms,
                test_scene,
            )
            for scene in scenes
        ]
        super().__init__(data_srcs)
