import numpy as np
import unittest

from lib.camera import Camera 

class TestCamera(unittest.TestCase):
    def test_properties(self):
        camera = Camera(640, 480, 320, 240, 640, 480)
        self.assertTrue(isinstance(camera, Camera))
        self.assertEqual(camera.ff, (640, 480))
        self.assertEqual(camera.pp, (320, 240))
        self.assertEqual(camera.resolution, (640, 480))

    def test_from_yfov(self):
        camera = Camera.from_yfov(np.pi/2, 640, 480)
        self.assertTrue(isinstance(camera, Camera))
        self.assertAlmostEqual(camera.ff[0], 240)
        self.assertAlmostEqual(camera.ff[1], 240)
        self.assertEqual(camera.pp, (320, 240))
        self.assertEqual(camera.resolution, (640, 480))

    def test_project(self):
        camera = Camera(640, 480, 320, 240, 640, 480)
        # Point along principal axis
        pt3 = np.array([0, 0, 10])
        pt2 = camera.project(pt3)
        self.assertEqual(pt2.shape, (2,))
        np.testing.assert_almost_equal(pt2, np.array(camera.pp))
        pt3 = np.array([1, 1, 10])
        pt2 = camera.project(pt3)
        np.testing.assert_almost_equal(pt2, np.array([384, 288]))
        pt3 = np.array([-1, -1, 10])
        pt2 = camera.project(pt3)
        np.testing.assert_almost_equal(pt2, np.array([256, 192]))
        
    def test_unproject(self):
        camera = Camera(640, 480, 320, 240, 640, 480)
        # Unproject principal point
        pt2 = np.array([320, 240])
        pt3 = camera.unproject(pt2)
        self.assertEqual(pt3.shape, (3,))
        np.testing.assert_almost_equal(pt3, np.array([0, 0, 1]))
        pt2 = np.array([256, 192], dtype=float)
        pt3 = camera.unproject(pt2)
        np.testing.assert_almost_equal(pt3, np.array([-0.1, -0.1, 1]))
        pt2 = np.array([384, 192], dtype=float)
        pt3 = camera.unproject(pt2)
        np.testing.assert_almost_equal(pt3, np.array([0.1, -0.1, 1]))
    
if __name__ == '__main__':
    unittest.main()