import numpy as np
from lib.camera import Camera


def unproject_points(pts, depth_image, camera, depth_scale=1000):
    """Given a set of points, depth_image and a camera,
    unproject the points into 3D. Default assumption is that
    depth is in millimeters
    """
    unprojected = np.array([camera.unproject(pt) for pt in pts])
    depth = np.array([depth_image[int(pt[0]), int(pt[1])] for pt in pts]) / depth_scale
    return unprojected * depth[:, None]
