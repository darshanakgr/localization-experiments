import os

import numpy as np
import open3d

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


def remove_common_points(point_cloud, threshold=10):
    open3d.geometry.estimate_normals(
        point_cloud, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
    )

    pcd_tree = open3d.geometry.KDTreeFlann(point_cloud)
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    outliers = []
    for i in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[i], 30)
        v = np.std([angle_between(normals[i], normals[j]) for j in idx])
        if v < np.radians(threshold):
            outliers.append(i)

    points = np.delete(points, outliers, axis=0)

    return pcd.make_point_cloud(points)


voxel_size = 0.03

# https://github.com/isl-org/Open3D/blob/master/cpp/open3d/pipelines/registration/Registration.cpp

data_dir = "data/DatasetV2/Secondary/00/"

for src_file_name in os.listdir(data_dir):
    src_file = f"{data_dir}/{src_file_name}"

    src_pcd = open3d.read_point_cloud(src_file)
    # cl, ind = open3d.geometry.statistical_outlier_removal(src_pcd, nb_neighbors=20, std_ratio=2.0)
    # src_pcd = open3d.geometry.select_down_sample(src_pcd, ind)

    src_pcd = open3d.voxel_down_sample(src_pcd, voxel_size)
    src_pcd.paint_uniform_color([1, 0.706, 0])

    tgt_pcd = remove_common_points(src_pcd, threshold=5)
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])

    # Plot point clouds after registration
    open3d.draw_geometries([src_pcd])
    open3d.draw_geometries([tgt_pcd])
