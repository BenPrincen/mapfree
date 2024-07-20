import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from lib.rot3 import Rot3


class TestRot3(unittest.TestCase):
    def test_properties(self):
        r = Rot3()
        self.assertTrue(isinstance(r, Rot3))
        np.testing.assert_equal(r.matrix, np.eye(3))

    def test_quaternion(self):
        # Rotation by 90 about z-axis
        q = np.array([1, 0, 0, 1])
        r = Rot3.from_quaternion(q)
        expected = np.zeros((3, 3))
        expected[0, 1] = -1
        expected[1, 0] = 1
        expected[2, 2] = 1
        np.testing.assert_almost_equal(r.matrix, expected)

        # Rotation by 90 about x-axis
        r_gt = Rotation.from_euler("x", 90, degrees=True)
        r_t = Rot3.Rx(90)
        # print(r_gt.as_quat())
        # print(r_t.get_quat())
        np.testing.assert_almost_equal(r_gt.as_quat()[[3, 0, 1, 2]], r_t.get_quat())

    def test_from_r(self):
        q = np.array([1, 0, 0, 1])
        rq = Rot3.from_quaternion(q)
        rz = Rot3.Rz(90)
        self.assertTrue(rz.almost_equal(rq))


if __name__ == "__main__":
    unittest.main()
