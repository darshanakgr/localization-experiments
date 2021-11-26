import open3d
import numpy as np


def get_features(file_path):
    data = np.load(file_path)
    scores = data["scores"]
    features = open3d.pipelines.registration.Feature()
    features.data = data["features"].T
    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"])
    return keypts, features, scores


def get_features_arr(file_path):
    data = np.load(file_path)
    features = data["features"]
    return features


def get_open3d_features(features_arr):
    features = open3d.pipelines.registration.Feature()
    features.data = features_arr.T
    return features


def read_point_cloud(file_path):
    return open3d.io.read_point_cloud(file_path)


def registration(src_keypts, tgt_keypts, src_desc, tgt_desc, voxel_size, max_iteration=4000000, max_validation=500):
    distance_threshold = voxel_size * 1.5
    iterations = len(src_keypts.points) * len(tgt_keypts.points)
    print(f"Iterations: {iterations}\t", end="")
    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            # open3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.radians(30)),
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, max_validation)
    )
    return result
