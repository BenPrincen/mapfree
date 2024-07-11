""" 3D Rigid transformation """

import numpy as np

from .rot3 import Rot3


class Pose3(object):

    def __init__(self, R=Rot3(), T=np.zeros(3)):
        """Initialize from a Rot3 and a Translation"""
        self._rotation = R
        self._translation = T
        m = np.eye(4)
        m[:3, :3] = R.matrix
        m[:3, 3] = T
        self._matrix = m

    @property
    def matrix(self):
        return self._matrix

    @property
    def rotation(self):
        return self._rotation

    @property
    def translation(self):
        return self._translation

    def inverse(self):
        m_inv = np.linalg.inv(self._matrix)
        return Pose3(R=Rot3(R=m_inv[:3, :3]), T=m_inv[:3, 3])

    def __mul__(self, m):
        """Overload * operator so it works for multiplying 2 Pose3's or
        multiplying a Pose3 times a 3D point
        """
        if isinstance(m, Pose3):
            m = np.dot(self.matrix, m.matrix)
            return Pose3(R=Rot3(m[:3, :3]), T=m[:3, 3])
        elif isinstance(m, np.ndarray) and m.shape == (3,):
            return np.dot(self.matrix, np.append(m, 1.0))[:3]
        else:
            raise ValueError("Input must be Pose3, or numpy array (3,)")

    def __str__(self):
        return "r: " + str(self._rotation) + "\n" + "t: " + str(self._translation)

    def almost_equal(self, other, tol=1e-7):
        return np.allclose(self._matrix, other._matrix, atol=tol)
