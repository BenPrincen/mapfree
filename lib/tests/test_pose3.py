import numpy as np
import unittest

from lib.pose3 import Pose3
from lib.rot3 import Rot3


class TestPose3(unittest.TestCase):
    def test_properties(self):
        p = Pose3()
        self.assertTrue(isinstance(p, Pose3))
        np.testing.assert_equal(p.rotation.matrix, np.eye(3))
        np.testing.assert_equal(p.translation, np.zeros(3))

    def test_mul_inv(self):
        p0 = Pose3()
        p1 = Pose3()
        pp = p1 * p0
        np.testing.assert_equal(pp.rotation.matrix, np.eye(3))
        np.testing.assert_equal(pp.translation, np.zeros(3))
        v_in = np.array([1, 2, 3])
        v_out = p0 * v_in
        np.testing.assert_array_equal(v_out, v_in)
        p0 = Pose3(T=np.array([1, 2, 3]))
        pp = p1 * p0
        self.assertTrue(pp.almost_equal(p0))
        r = Rot3.Rx(30)
        p = Pose3(R=Rot3(R=r.matrix), T=np.array([1, 2, 3]))
        p_inv = p.inverse()
        p_pi = p * p_inv
        p_pi.almost_equal(Pose3())

    def test_almost_equal(self):
        p0 = Pose3()
        p1 = Pose3()
        self.assertTrue(p0.almost_equal(p1))


if __name__ == "__main__":
    unittest.main()
