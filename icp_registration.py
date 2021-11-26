import numpy as np
import open3d
from open3d.cpu.pybind.geometry import PointCloud


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def separate_planes(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inliers, outlier_cloud


def get_features(file_path):
    data = np.load(file_path)
    scores = data["scores"]
    features = open3d.pipelines.registration.Feature()
    features.data = data["features"].T
    keypts = open3d.cpu.pybind.geometry.PointCloud()
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


def read_point_cloud(file_path) -> PointCloud:
    return open3d.io.read_point_cloud(file_path)


def registration(src_keypts, tgt_keypts, src_desc, tgt_desc, voxel_size, max_iteration=4000000, max_validation=500):
    distance_threshold = voxel_size * 1.5
    # distance_threshold = 0.05
    # iterations = len(src_keypts.points) * len(tgt_keypts.points)
    # print(f"Iterations: {iterations}\t", end="")
    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, max_validation)
    )
    return result


def remove_redundant_points(pcd, features):
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))

    pcd_tree = open3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    inliers = []
    outliers = []
    for i in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 10)
        v = np.std([angle_between(normals[i], normals[j]) for j in idx])
        if v > 0.5:
            inliers.append(i)
        else:
            outliers.append(i)

    features = np.delete(features, outliers, axis=0)
    features = get_open3d_features(features)
    return pcd.select_by_index(inliers), features


def main():
    VOXEL_SIZE = 0.1

    # for tgt_file_name in os.listdir("data/DatasetV2/Secondary/00/"):
    #     src_pcd = read_point_cloud("data/DatasetV2/Primary/01/lidar_1636437066490883500.pcd")
    #     src_pcd = src_pcd.voxel_down_sample(VOXEL_SIZE)
    #     src_pcd.paint_uniform_color([1, 0.706, 0])
    #
    #     src_features = get_features_arr(f"data/Features/{VOXEL_SIZE}/lidar_1636437066490883500.npz")
    #
    #     src_keypts = copy.deepcopy(src_pcd)
    #
    #     for i in range(2):
    #         idx, src_keypts = separate_planes(src_keypts)
    #         src_features = np.delete(src_features, idx, axis=0)
    #
    #     src_features = get_open3d_features(src_features)
    #
    #     tgt_pcd = read_point_cloud(f"data/DatasetV2/Secondary/00/{tgt_file_name}")
    #     tgt_pcd = tgt_pcd.voxel_down_sample(VOXEL_SIZE)
    #     tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
    #
    #     tgt_keypts = copy.deepcopy(tgt_pcd)
    #     tgt_features = get_features_arr(f"data/Features/{VOXEL_SIZE}/{tgt_file_name.replace('pcd', 'npz')}")
    #
    #     # for i in range(3):
    #     #     idx, tgt_keypts = separate_planes(tgt_keypts)
    #     #     tgt_features = np.delete(tgt_features, idx, axis=0)
    #
    #     tgt_features = get_open3d_features(tgt_features)
    #
    #     open3d.visualization.draw_geometries([src_keypts, tgt_keypts])
    #
    #     result_ransac = registration(src_keypts, tgt_pcd, src_features, tgt_features, VOXEL_SIZE)
    #     src_pcd.transform(result_ransac.transformation)
    #     print(result_ransac)
    #
    #     open3d.visualization.draw_geometries([src_pcd, tgt_pcd])

    src_pcd: PointCloud = open3d.io.read_point_cloud("data/DatasetV2/Primary/01/lidar_1636437066490883500.pcd")
    tgt_pcd: PointCloud = open3d.io.read_point_cloud("data/DatasetV2/Secondary/00/lidar_1636435283264391400.pcd")

    src_pcd = src_pcd.voxel_down_sample(VOXEL_SIZE)
    tgt_pcd = tgt_pcd.voxel_down_sample(VOXEL_SIZE)

    src_pcd.paint_uniform_color([1, 0.706, 0])
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])

    # src_keypts, src_features, src_scores = get_features(f"data/Features/{VOXEL_SIZE}/lidar_1636437066490883500.npz")
    # tgt_keypts, tgt_features, tgt_scores = get_features(f"data/Features/{VOXEL_SIZE}/lidar_1636435283264391400.npz")
    #
    # result_ransac = registration(src_keypts, tgt_keypts, src_features, tgt_features, VOXEL_SIZE)
    # src_pcd.transform(result_ransac.transformation)
    # print(result_ransac)
    #
    # open3d.visualization.draw_geometries([src_pcd, tgt_pcd])
    #
    # # src_pcd, tgt_pcd, source_fpfh, target_fpfh = prepare_dataset(src_pcd, tgt_pcd, VOXEL_SIZE)
    # # result_ransac = registration(src_pcd, tgt_pcd, source_fpfh, target_fpfh, VOXEL_SIZE)
    # # src_pcd.transform(result_ransac.transformation)
    # # open3d.visualization.draw_geometries([src_pcd, tgt_pcd])
    #
    # # with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    # #     labels = np.array(src_pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))
    # #
    # # max_label = labels.max()
    # # print(f"point cloud has {max_label + 1} clusters")
    # # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # # colors[labels < 0] = 0
    # # src_pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
    # # open3d.visualization.draw_geometries([src_pcd])
    #
    # src_features = get_features_arr(f"data/Features/{VOXEL_SIZE}/lidar_1636437066490883500.npz")
    # tgt_features = get_features_arr(f"data/Features/{VOXEL_SIZE}/lidar_1636435283264391400.npz")
    #
    # src_keypts = copy.deepcopy(src_pcd)
    # tgt_keypts = copy.deepcopy(tgt_pcd)
    #
    # for i in range(4):
    #     idx, src_keypts = separate_planes(src_keypts)
    #     src_features = np.delete(src_features, idx, axis=0)
    #     print(f":: Feature Vectors {src_features.shape[0]} {tgt_features.shape[0]}")
    #
    # for i in range(3):
    #     idx, tgt_keypts = separate_planes(tgt_keypts)
    #     tgt_features = np.delete(tgt_features, idx, axis=0)
    #     print(f":: Feature Vectors {src_features.shape[0]} {tgt_features.shape[0]}")
    #
    # open3d.visualization.draw_geometries([src_keypts, tgt_keypts])
    #
    # src_features = get_open3d_features(src_features)
    # tgt_features = get_open3d_features(tgt_features)
    #
    # print(f":: Feature Vectors {src_features.num()} {tgt_features.num()}")
    # result_ransac = registration(src_keypts, tgt_pcd, src_features, tgt_features, VOXEL_SIZE, 1000000, 500)
    # src_pcd.transform(result_ransac.transformation)
    # print(result_ransac)
    #
    # open3d.visualization.draw_geometries([src_pcd, tgt_pcd])

    src_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    tgt_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    # open3d.visualization.draw_geometries([src_pcd], point_show_normal=True)

    # pcd_tree = open3d.geometry.KDTreeFlann(src_pcd)
    # points = np.asarray(src_pcd.points)
    # normals = np.asarray(src_pcd.normals)
    # inliers = []
    # for i in range(points.shape[0]):
    #     [k, idx, _] = pcd_tree.search_knn_vector_3d(src_pcd.points[i], 10)
    #     v = np.std([angle_between(normals[i], normals[j]) for j in idx])
    #     if v > 0.5:
    #         inliers.append(i)
    #         src_pcd.colors[i] = [1, 0, 0]
    #     else:
    #         src_pcd.colors[i] = [0, 1, 0]
    #
    # src_pcd.colors[1500] = [1, 0, 0]
    # np.asarray(src_pcd.colors)[idx[1:], :] = [0, 0, 1]

    src_features = get_features_arr(f"data/Features/{VOXEL_SIZE}/lidar_1636437066490883500.npz")
    tgt_features = get_features_arr(f"data/Features/{VOXEL_SIZE}/lidar_1636435283264391400.npz")

    src_pcd, src_features = remove_redundant_points(src_pcd, src_features)
    tgt_pcd, tgt_features = remove_redundant_points(tgt_pcd, tgt_features)

    open3d.visualization.draw_geometries([src_pcd])
    open3d.visualization.draw_geometries([tgt_pcd])

    print(f":: Feature Vectors {src_features.num()} {tgt_features.num()}")
    result_ransac = registration(src_pcd, tgt_pcd, src_features, tgt_features, VOXEL_SIZE)
    src_pcd.transform(result_ransac.transformation)
    print(result_ransac)

    open3d.visualization.draw_geometries([src_pcd, tgt_pcd])


if __name__ == '__main__':
    main()
