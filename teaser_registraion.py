import os
import copy
import time
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from timeit import default_timer as timer

import teaserpp_python


NOISE_BOUND = 0.05

def read_feature_file(file_path):
    data = np.load(file_path)
    keypts = o3d.geometry.PointCloud()
    keypts.points = o3d.utility.Vector3dVector(data['keypts'])
    
    desc = o3d.registration.Feature()
    desc.data = data["features"].T
    
    return keypts, desc


def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    """
    Use kdtree to find mutually closest keypoints 

    ref_key: reference keypoints (source)
    test_key: test keypoints (target)
    ref: reference feature (source feature)
    test: test feature (target feature)
    """
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]

    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test.data.T)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)

    # find mutually closest points
    ref_match_idx = np.nonzero(
        np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    )[0]
    
    ref_matched_keypoints = ref_keypoints[ref_match_idx]
    test_matched_keypoints = test_keypoints[ref_NN_idx[ref_match_idx]]

    return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints)


def compose_mat4_from_teaserpp_solution(solution):
    """
    Compose a 4-by-4 matrix from teaserpp solution
    """
    s = solution.scale
    rotR = solution.rotation
    t = solution.translation
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = rotR
    M = T.dot(R)

    if s == 1:
        M = T.dot(R)
    else:
        S = np.eye(4)
        S[0:3, 0:3] = np.diag([s, s, s])
        M = T.dot(R).dot(S)

    return M


def execute_teaser_global_registration(source, target):
    """
    Use TEASER++ to perform global registration
    """
    # Prepare TEASER++ Solver
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    print("TEASER++ Parameters are:", solver_params)
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # Solve with TEASER++
    teaserpp_solver.solve(source, target)
    est_solution = teaserpp_solver.getSolution()
    est_mat = compose_mat4_from_teaserpp_solution(est_solution)
    max_clique = teaserpp_solver.getTranslationInliersMap()
    print("Max clique size:", len(max_clique))
    final_inliers = teaserpp_solver.getTranslationInliers()
    return est_mat, max_clique, final_inliers


source_file = "data/FeaturesV7/0.05/1620203813753530200.npz"
target_file = "data/FeaturesV7/0.05/1620203820217838500.npz"

source_keypts, source_desc = read_feature_file(source_file)
target_ketpts, target_desc = read_feature_file(target_file)

print(len(source_keypts.points), len(target_ketpts.points))

source_matched_keypts, target_matched_keypts = find_mutually_nn_keypoints(source_keypts, target_ketpts, source_desc, target_desc)

source_matched_keypts = np.squeeze(source_matched_keypts)
target_matched_keypts = np.squeeze(target_matched_keypts)

print("source_matched_keypts:", source_matched_keypts.shape)
print("target_matched_keypts:", target_matched_keypts.shape)

est_mat, max_clique, time = execute_teaser_global_registration(source_matched_keypts, target_matched_keypts)

print(est_mat, time)


