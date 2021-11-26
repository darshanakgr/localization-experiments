import copy
import math
import os
import numpy as np
import open3d
from open3d import PointCloud
import utils.pcd as pcd


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_features_arr(file_path):
    data = np.load(file_path)
    features = data["features"]
    return features


def read_features(file_path):
    features = get_features_arr(file_path)
    reg_features = open3d.registration.Feature()
    reg_features.data = features.T
    return reg_features


def remove_redundant_points(pc, features):
    open3d.geometry.estimate_normals(pc, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    pcd_tree = open3d.geometry.KDTreeFlann(pc)
    points = np.asarray(pc.points)
    normals = np.asarray(pc.normals)
    outliers = []
    for i in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pc.points[i], 30)
        v = np.std([angle_between(normals[i], normals[j]) for j in idx])
        if v < np.radians(10):
            outliers.append(i)
    points = np.delete(points, outliers, axis=0)
    features = np.delete(features, outliers, axis=0)
    reg_features = open3d.registration.Feature()
    reg_features.data = features.T
    return pcd.make_point_cloud(points), reg_features


voxel_size = 0.1

for tgt_file_name in os.listdir("data/DatasetV2/Secondary/00/"):
    src_file = f"data/DatasetV2/Secondary/00/{tgt_file_name}"
    tgt_file = "data/DatasetV2/Primary/01/lidar_1636437066490883500.pcd"

    src_pcd: PointCloud = open3d.read_point_cloud(src_file)
    tgt_pcd: PointCloud = open3d.read_point_cloud(tgt_file)

    src_pcd = open3d.voxel_down_sample(src_pcd, voxel_size)
    tgt_pcd = open3d.voxel_down_sample(tgt_pcd, voxel_size)

    open3d.geometry.estimate_normals(src_pcd,
                                     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    open3d.geometry.estimate_normals(tgt_pcd,
                                     search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))

    src_features = get_features_arr(f"data/Features/{voxel_size}/{tgt_file_name.replace('pcd', 'npz')}")

    tgt_features = read_features(f"data/Features/{voxel_size}/lidar_1636437066490883500.npz")
    # tgt_features = get_features_arr(f"data/Features/{voxel_size}/{tgt_file_name.replace('pcd', 'npz')}")

    src_pcd, src_features = remove_redundant_points(src_pcd, src_features)
    # tgt_pcd, tgt_features = remove_redundant_points(tgt_pcd, tgt_features)

    src_pcd.paint_uniform_color([1, 0.706, 0])
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])

    # open3d.draw_geometries([src_pcd, tgt_pcd])

    result_ransac = pcd.execute_global_registration(src_pcd, tgt_pcd, src_features, tgt_features, voxel_size)
    to_print = f"Keypts: [{len(src_pcd.points)}, {len(tgt_pcd.points)}]\t"
    to_print += f"No of matches: {len(result_ransac.correspondence_set)}"
    print(to_print)

    # Plot point clouds after registration
    pcd.draw_registration_result(src_pcd, tgt_pcd, result_ransac.transformation)
