""" 3D rotation """

import math
import numpy as np
from scipy.spatial.transform import Rotation


class Rot3(object):
    def __init__(self, R=np.eye(3)):
        """Initialize from a 3x3 matrix (should be unitary!)"""
        self._rotation = R

    @property
    def matrix(self):
        return self._rotation

    def inverse(self):
        return Rot3(R=self._rotation.transpose())

    @classmethod
    def Rx(cls, angle):
        """Rot3 representing rotation about x axis
        Input:
            angle - angle in degrees
        """
        c = math.cos(np.radians(angle))
        s = math.sin(np.radians(angle))
        r = np.eye(3)
        r[1, 1] = c
        r[2, 2] = c
        r[1, 2] = -s
        r[2, 1] = s
        return cls(R=r)

    @classmethod
    def Ry(cls, angle):
        """Rot3 representing rotation about y axis
        Input:
            angle - angle in degrees
        """
        c = math.cos(np.radians(angle))
        s = math.sin(np.radians(angle))
        r = np.eye(3)
        r[0, 0] = c
        r[2, 2] = c
        r[0, 2] = -s
        r[2, 0] = s
        return cls(R=r)

    @classmethod
    def Rz(cls, angle):
        """Rot3 representing rotation about z axis
        Input:
            angle - angle in degrees
        """
        c = math.cos(np.radians(angle))
        s = math.sin(np.radians(angle))
        r = np.eye(3)
        r[0, 0] = c
        r[1, 1] = c
        r[0, 1] = -s
        r[1, 0] = s
        return cls(R=r)

    @classmethod
    def from_quaternion(cls, quat):
        """Note: quat is in wxyz format"""
        # Change to xyzw format for scipy
        qxyzw = np.roll(quat, -1)
        r = Rotation.from_quat(qxyzw)
        r = r.as_matrix()
        return cls(R=r)

    def __mul__(self, m):
        """Multiply works for either a Rot3, or a vector"""
        if isinstance(m, Rot3):
            m = np.dot(self.matrix, m.matrix)
            return Rot3(R=m)
        elif isinstance(m, np.ndarray) and m.shape == (3,):
            return np.dot(self.matrix, m)
        else:
            raise ValueError("Input must be Rot33, or numpy array (3,)")

    def __str__(self):
        return str(self.matrix)

    def almost_equal(self, other, tol=1e-7):
        return np.allclose(self.matrix, other.matrix, atol=tol)
