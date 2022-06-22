import copy
import math
import os
import numpy as np
import open3d


def convert_to_pcd(pc, output_file):
    header = [
        "VERSION .7\n",
        "FIELDS x y z\n",
        "SIZE 4 4 4\n",
        "TYPE F F F\n",
        "COUNT 1 1 1\n",
        "WIDTH 1024\n",
        "HEIGHT 768\n",
        "VIEWPOINT 0 0 0 1 0 0 0\n",
        "POINTS 786432\n",
        "DATA ascii\n"
    ]
    file = open(output_file, mode="w")
    file.writelines(header)
    for i in np.arange(pc.shape[0]):
        file.write("%.5f %.5f %.5f\n" % (pc[i, 0], pc[i, 1], pc[i, 2]))
        if i % 10 == 0:
            file.flush()

    file.close()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # open3d.estimate_normals(source_temp)
    # open3d.estimate_normals(target_temp)
    target_temp.paint_uniform_color([1, 0.706, 0])
    source_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])


# def execute_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
#     result = open3d.registration_ransac_based_on_feature_matching(
#         src_keypts, tgt_keypts, src_desc, tgt_desc,
#         distance_threshold,
#         open3d.TransformationEstimationPointToPoint(False), 3,
#         [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#          open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
#         open3d.RANSACConvergenceCriteria(4000000, 500))
#     return result

def execute_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    result = open3d.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold,
        open3d.TransformationEstimationPointToPoint(False),
        3, [
            open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        open3d.RANSACConvergenceCriteria(4000000, 1000)
    )
    return result


def read_pcd_file(file_path, voxel_size=0.03):
    pcd = open3d.read_point_cloud(file_path)
    pcd = open3d.voxel_down_sample(pcd, voxel_size)
    return pcd


def read_features_file(file_path, random_points=0):
    data = np.load(file_path)
    scores = data["scores"]
    
    if random_points > 0:
        indices = np.random.choice(data["features"].shape[0], random_points, replace=False)
        features = open3d.registration.Feature()
        features.data = data["features"][indices, :].T
        keypts = open3d.PointCloud()
        keypts.points = open3d.Vector3dVector(data["keypts"][indices, :])
        return keypts, features, scores
    else:
        features = open3d.registration.Feature()
        features.data = data["features"].T
        keypts = open3d.PointCloud()
        keypts.points = open3d.Vector3dVector(data["keypts"])
        return keypts, features, scores





def read_pcd_and_features(file_path, feature_dir, voxel_size, random_points=0):
    pcd = open3d.read_point_cloud(file_path)
    pcd = open3d.voxel_down_sample(pcd, voxel_size)
    feature_file = os.path.join(feature_dir, str(voxel_size), file_path.split("/")[-1].replace(".pcd", ".npz"))
    data = np.load(feature_file)
    scores = data["scores"]
    if random_points > 0:
        # indices = np.where(scores > 0.7)[0]
        indices = np.random.choice(data["features"].shape[0], random_points, replace=False)
        features = open3d.registration.Feature()
        features.data = data["features"][indices, :].T
        keypts = open3d.PointCloud()
        keypts.points = open3d.Vector3dVector(data["keypts"][indices, :])
        return pcd, keypts, features, scores
    else:
        features = open3d.registration.Feature()
        features.data = data["features"].T
        keypts = open3d.PointCloud()
        keypts.points = open3d.Vector3dVector(data["keypts"])
        return pcd, keypts, features, scores


def get_angles_from_transformation(t):
    rx = math.degrees(math.atan2(t[2][1], t[2][2]))
    ry = math.degrees(math.atan2(-t[2][0], math.sqrt(t[2][1] ** 2 + t[2][2] ** 2)))
    rz = math.degrees(math.atan2(t[1][0], t[0][0]))
    return np.array([rx, ry, rz])


def get_translation_from_transformation(t):
    return np.array([t[0][3], t[1][3], t[2][3]])


def make_point_cloud(pts):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    return pcd


def remove_color(pcd_file, voxel_size=0.03):
    pcd = read_pcd_file(pcd_file, voxel_size)
    pcd = make_point_cloud(np.asarray(pcd.points))
    return pcd


def rgb(r, g, b):
    return r / 255.0, g / 255.0, b / 255.0
