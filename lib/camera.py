''' Simple linear camera '''
import math
import numpy as np

class Camera(object):
    def __init__(self, fx, fy, px, py, width, height):
        ''' Construct simple linear camera.
            Inputs:
                fx, fy (float) - focal length
                px, py (float) - principal point
                width, height (float) - size of image
        '''
        self._ff = (fx, fy)
        self._pp = (px, py)
        self._K = np.eye((3))
        self._K[0, 0] = fx
        self._K[1, 1] = fy
        self._K[0, 2] = px
        self._K[1, 2] = py
        self._width = width
        self._height = height

    @classmethod
    def from_yfov(cls, yfov, width, height):
        px = width / 2.0
        py = height / 2.0
        fy = height / math.tan(yfov / 2.0) / 2.0
        fx = fy
        return cls(fx, fy, px, py, width, height)
       
    @property
    def ff(self):
        return self._ff

    @property
    def pp(self):
        return self._pp

    @property
    def K(self):
        return self._K

    @property
    def resolution(self):
        return (self._width, self._height)

    def project(self, pt3):
        ''' Project from 3D point to 2D image point.
            Input:
                pt3 (np.array (3,)) 
            Returns:
                np.array (2,) point in image coordinates
        '''
        pt2 = np.dot(self._K, pt3)
        return pt2[:2] / pt2[2]

    def unproject(self, pt2):
        ''' Unproject 2D image point to 3D on the Z=1 plane
            Input:
                pt2 (np.array (2,))
            Returns:
                np.array (3,) unprojected point on the plane Z=1
        '''
        pt2 = pt2 - np.array(self.pp)
        X = pt2 / np.array(self.ff)
        return np.hstack((X, np.array(1)))

    def __str__(self):
        return "ff: " + str(self._ff) + " pp: " + str(self._pp) + " size: ({},{})".format(self._width, self._height)
