import numpy as np


# from: https://hunterheidenreich.com/posts/kabsch_algorithm/
# Translation vector from the implementation was different
# than we expect, so code is changed a bit
def kabsch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # Translation vector - NOTE: this is different than source
    t = centroid_Q - np.matmul(centroid_P, R.T)
    # RMSD - NOTE: this is different than source
    rmsd = np.sqrt(np.sum(np.square(np.matmul(R, P) + t - Q)) / P.shape[0])

    return R, t, rmsd


def find_relative_pose(
    pts1, pts2, ransac_iterations=100, inlier_threshold=0.15, num_matches=3
):
    """Find the relative pose between 3D pts1 and pts2.
    Use Kabsch algorithm in a RANSAC loop. Returns the best pose,
    number of inliers, and the inlier points
    """
    if len(pts1) != len(pts2):
        raise AttributeError("Length of points don't match")

    if len(pts1) < num_matches:
        # print("Min number of matches not met")
        return None

    max_inliers = 0
    best_pose = None
    for it in range(0, ransac_iterations):
        samples = np.random.choice(range(0, len(pts1)), num_matches, replace=False)
        s1, s2 = pts1[samples], pts2[samples]
        # Solve
        R, t, _ = kabsch(s1, s2)
        # Count inliers
        tx_pts1 = np.matmul(pts1, R.T) + t
        # Find distances between points
        distances = np.sqrt(np.sum((pts2 - tx_pts1) ** 2, axis=-1))
        # Count inliers below threshold
        inliers = distances < inlier_threshold
        inliers_idx = np.nonzero(distances < inlier_threshold)
        if inliers.sum() > max_inliers:
            best_pose = R, t
            max_inliers = float(inliers.sum())
            # Save the inlier points
            inc_pts1 = pts1[inliers_idx]
            inc_pts2 = pts2[inliers_idx]
    if best_pose is not None:
        return best_pose, max_inliers, inc_pts1, inc_pts2
    else:
        return None
