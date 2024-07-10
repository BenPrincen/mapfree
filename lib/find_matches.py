import cv2
import numpy as np


def find_keypoints(imgs):
    """Find keypoints for all images in the list, return a list of
    (keypoint, descriptor) tuples
    """
    sift = cv2.SIFT_create()
    return [sift.detectAndCompute(img, None) for img in imgs]


def find_matches(kp1, des1, kp2, des2, min_match_count=10):
    """Find matches between descriptors"""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src_pts, dst_pts
