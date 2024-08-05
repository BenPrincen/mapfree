import unittest

import numpy as np

from lib.camera import Camera
from lib.unproject_points import unproject_points


class TestUnprojectPoints(unittest.TestCase):
    def setUp(self):
        self._camera = Camera(590, 590, 270, 360, 540, 720)

    def test_uniform_depth(self):
        pts_y = np.array([360, 0, 719, 0, 719])
        pts_x = np.array([240, 0, 539, 539, 0])
        pts = np.stack([pts_x, pts_y], axis=1).squeeze()
        depth_image = np.int16(np.ones((720, 540)) * 3000)
        pts_3d = unproject_points(pts, depth_image, self._camera)
        self.assertEqual(pts_3d.shape, (5, 3))
        # Project back to get points - should be equal
        for pt3, pt2 in zip(pts_3d, pts):
            proj_pt = self._camera.project(pt3)
            np.testing.assert_almost_equal(proj_pt, pt2)
