import unittest

import numpy as np

from lib.find_pose import find_relative_pose, kabsch
from lib.pose3 import Pose3
from lib.rot3 import Rot3


class TestKabsch(unittest.TestCase):
    def setUp(self):
        self.pts = 10 * np.random.rand(3, 3)

    def test_translation(self):
        pose = Pose3(T=[0, 1, 1])
        tx_pts = np.array([pose * p for p in self.pts])
        R, t, err = kabsch(self.pts, tx_pts)
        est_pose = Pose3(Rot3(R), t)
        self.assertTrue(est_pose.almost_equal(pose))

    def test_rotation(self):
        pose = Pose3(R=Rot3.Rx(60), T=[0, 0, 0])
        tx_pts = np.array([pose * p for p in self.pts])
        R, t, err = kabsch(self.pts, tx_pts)
        est_pose = Pose3(Rot3(R), t)
        self.assertTrue(est_pose.almost_equal(pose))

    def test_both(self):
        pose = Pose3(R=Rot3.Ry(60), T=[1, 0.5, 0.6])
        tx_pts = np.array([pose * p for p in self.pts])
        R, t, err = kabsch(self.pts, tx_pts)
        est_pose = Pose3(Rot3(R), t)
        self.assertTrue(est_pose.almost_equal(pose))


class TestFindRelativePose(unittest.TestCase):
    def setUp(self):
        self.pts = 10 * np.random.rand(100, 3)

    def test_no_outliers(self):
        # Create a random set of 100 3D points
        # Transform points - translation only test
        pose = Pose3(T=[0, 1, 1])
        pts2 = np.array([pose * p for p in self.pts])
        (rot, translation), num_inliers, inc_pts1, inc_pts2 = find_relative_pose(
            self.pts, pts2
        )
        est_pose = Pose3(Rot3(rot), translation)
        self.assertEqual(num_inliers, 100)
        self.assertTrue(est_pose.almost_equal(pose))
        np.testing.assert_equal(inc_pts1, self.pts)
        np.testing.assert_equal(inc_pts2, pts2)
        # Test with rotation
        pose = Pose3(R=Rot3.Rx(45), T=[0, 0, 0])
        pts2 = np.array([pose * p for p in self.pts])
        (rot, translation), num_inliers, inc_pts1, inc_pts2 = find_relative_pose(
            self.pts, pts2
        )
        est_pose = Pose3(Rot3(rot), translation)
        self.assertEqual(num_inliers, 100)
        self.assertTrue(est_pose.almost_equal(pose))

    def test_with_outliers(self):
        # Test with 50 % outliers
        pose = Pose3(T=[0, 1, 1])
        pts2 = np.array([pose * p for p in self.pts])
        # Set some of the points to zeros
        pts2[pts2.shape[0] // 2 :, :] = np.zeros(3)
        (rot, translation), num_inliers, inc_pts1, inc_pts2 = find_relative_pose(
            self.pts, pts2
        )
        self.assertEqual(inc_pts1.shape, (50, 3))
        self.assertEqual(inc_pts2.shape, (50, 3))
        est_pose = Pose3(Rot3(rot), translation)
        self.assertEqual(num_inliers, 50)
        self.assertTrue(est_pose.almost_equal(pose))
