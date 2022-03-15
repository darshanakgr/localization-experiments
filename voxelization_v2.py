import os

import cv2
import numpy as np
import open3d


def make_pcd(points, color=None):
    c = [1, 0, 0] if color is None else color
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector([c for i in range(len(points))])
    return pcd


def get_features(feature_file):
    data = np.load(feature_file)
    scores = data["scores"]
    features = data["features"]
    # features = open3d.registration.Feature()
    # features.data = data["features"].T
    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"])
    return keypts, features, scores


def registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    result = open3d.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc,
        distance_threshold,
        open3d.TransformationEstimationPointToPoint(False), 3,
        [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        open3d.RANSACConvergenceCriteria(40000000, 600))
    return result


def print_registration_result(src_keypts, tgt_keypts, result_ransac):
    print(f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]", end="\t")
    print(f"No of matches: {len(result_ransac.correspondence_set)}", end="\t")
    print(f"Inlier RMSE: {result_ransac.inlier_rmse}", end="\n")


def get_matching_indices(src_pts, tgt_pts, search_voxel_size):
    match_inds = []
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
    match = bf_matcher.match(src_pts, tgt_pts)
    for match_val in match:
        if match_val.distance < search_voxel_size:
            match_inds.append([match_val.queryIdx, match_val.trainIdx])
    distances = [m.distance for m in match]
    print(max(distances), min(distances))
    return np.array(match_inds)


def main():
    src_file_names = os.listdir("data/DatasetV2/Secondary/03/")
    for src_file_name in src_file_names:
        tgt_feature_file = "data/FeaturesV1/0.1/lidar_1637299401488642900.npz"
        src_feature_file = os.path.join("data/FeaturesV1/0.1/", src_file_name.replace("pcd", "npz"))

        src_keypts, src_features, src_scores = get_features(src_feature_file)
        tgt_keypts, tgt_features, tgt_scores = get_features(tgt_feature_file)

        src_keypts.paint_uniform_color([1, 0.706, 0])
        tgt_keypts.paint_uniform_color([0, 0.651, 0.929])

        tgt_pts = np.asarray(tgt_keypts.points).astype(np.float32)

        matches = get_matching_indices(src_features, tgt_features, 0.3)

        tgt_heatmap_pcd = open3d.geometry.PointCloud()
        tgt_heatmap_pcd.points = open3d.utility.Vector3dVector(tgt_pts)
        colors = [[1, 0.706, 0] if i in matches[:, 1] else [0, 0.651, 0.929] for i in range(len(tgt_pts))]
        tgt_heatmap_pcd.colors = open3d.utility.Vector3dVector(colors)

        # open3d.visualization.draw_geometries([src_keypts])
        open3d.visualization.draw_geometries([tgt_heatmap_pcd])


if __name__ == '__main__':
    main()
