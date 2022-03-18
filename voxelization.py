import open3d
import numpy as np
import os
import matplotlib.pyplot as plt


def rgb(r, g, b):
    return r / 255.0, g / 255.0, b / 255.0


def create_line_set(points, lines):
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def make_pcd(points, color=None):
    c = [1, 0, 0] if color is None else color
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector([c for i in range(len(points))])
    return pcd


def filter_indices(points, p, cell_size):
    px_min = p[0] - cell_size
    px_max = p[0] + cell_size
    pz_min = p[2] - cell_size
    pz_max = p[2] + cell_size
    xf = np.logical_and(points[:, 0] > px_min, points[:, 0] < px_max)
    zf = np.logical_and(points[:, 2] > pz_min, points[:, 2] < pz_max)
    return np.logical_and(xf, zf)


def get_features(feature_file):
    data = np.load(feature_file)
    scores = data["scores"]
    # features = open3d.registration.Feature()
    # features.data = data["features"].T
    features = data["features"].T
    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"])
    return keypts, features, scores


def get_cell_features(feature_file, p, cell_size):
    data = np.load(feature_file)
    scores = data["scores"]

    f = filter_indices(data["keypts"], p, cell_size)
    f = np.where(f)[0]

    # f = np.random.choice(f, 500, replace=False)

    # features = open3d.registration.Feature()
    # features.data = data["features"][f].T
    features = data["features"][f].T

    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"][f])

    return keypts, features, scores


def get_grid(pcd, cell_size):
    pcd_points = np.asarray(pcd.points)

    x_min, y_min, z_min = np.min(pcd_points, axis=0)
    x_max, y_max, z_max = np.max(pcd_points, axis=0)
    y_val = np.mean([y_min, y_max])

    points = []
    x_n = int((x_max - x_min) // cell_size)
    z_n = int((z_max - z_min) // cell_size)
    for i in range(z_n):
        z0 = float(z_min + cell_size * (i + 1))
        for j in range(x_n):
            x0 = float(x_min + cell_size * (j + 1))
            points.append([x0, y_val, z0])

    return points


def registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    result = open3d.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc,
        distance_threshold,
        open3d.TransformationEstimationPointToPoint(False), 3,
        [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold),
         open3d.CorrespondenceCheckerBasedOnNormal(normal_angle_threshold=np.deg2rad(30.0))],
        open3d.RANSACConvergenceCriteria(40000000, 600))
    return result


def fast_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, voxel_size):
    distance_threshold = 0.05
    result = open3d.registration.registration_fast_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc,
        open3d.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
    )
    return result


def print_registration_result(src_keypts, tgt_keypts, result_ransac, end="\n"):
    to_print = f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]\t"
    to_print += f"No of matches: {len(result_ransac.correspondence_set)}\t"
    to_print += f"Inlier RMSE: {result_ransac.inlier_rmse}"
    print(to_print, end=end)


def create_camera(camera_width, color):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    points = points * camera_width
    points = points - camera_width / 2
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def main():
    cell_size = 2

    pcd_file = "data/DatasetV2/Primary/06/lidar_1637299401488642900.pcd"
    # pcd_file = "data/DatasetV2/Primary/10/lidar_1638353640807327100.pcd"
    feature_file = "data/FeaturesV1/0.1/lidar_1637299401488642900.npz"

    pcd = open3d.read_point_cloud(pcd_file)
    pcd = open3d.voxel_down_sample(pcd, 0.03)
    pcd.paint_uniform_color(rgb(149, 165, 166))

    points = np.asarray(pcd.points)
    filter = np.logical_and(points[:, 1] > 1, points[:, 1] < 1.05)
    x = points[filter][:, 0]
    y = points[filter][:, 2]

    grid_points = get_grid(pcd, cell_size)
    grid = make_pcd(grid_points)

    for src_file_name in os.listdir("data/DatasetV2/Secondary/03/"):
        # src_file = f"data/DatasetV2/Secondary/03/{tgt_file_name}"
        src_feature_file = os.path.join("data/FeaturesV1/0.1/", src_file_name.replace("pcd", "npz"))
        highest_matches, highest_p = 0, None
        t = None

        src_keypts, src_features, src_scores = get_features(src_feature_file)
        src_feature_vector = src_features.min(axis=1)
        tgt_feature_vectors = []

        for p in grid_points:
            tgt_keypts, tgt_features, tgt_scores = get_cell_features(feature_file, p, cell_size)
            tgt_feature_vectors.append(tgt_features.min(axis=1))

        tgt_feature_vectors = np.array(tgt_feature_vectors)
        distances = tgt_feature_vectors - src_feature_vector.reshape(1, -1)
        distances = np.sum(np.square(distances), axis=1)
        print(np.argmin(distances))

        #     result_ransac = registration(src_keypts, tgt_keypts, src_features, tgt_features, 0.05)
        #     print_registration_result(src_keypts, tgt_keypts, result_ransac)
        #
        #     if highest_matches < len(result_ransac.correspondence_set):
        #         highest_matches = len(result_ransac.correspondence_set)
        #         highest_p = p
        #         t = result_ransac.transformation
        #
        #     # src_keypts.transform(result_ransac.transformation)
        #     # open3d.visualization.draw_geometries([src_keypts, tgt_keypts, grid])
        #
        # if highest_p is not None:
        #     src_keypts, src_features, src_scores = get_features(src_feature_file)
        #     tgt_keypts, tgt_features, tgt_scores = get_cell_features(feature_file, highest_p, cell_size)
        #
        #     src_keypts.paint_uniform_color([1, 0.706, 0])
        #     tgt_keypts.paint_uniform_color([0, 0.651, 0.929])
        #
        #     src_camera = create_camera(camera_width=0.25, color=[1, 0, 0])
        #
        #     result = open3d.registration.registration_icp(
        #         src_keypts, pcd, 0.05, t, open3d.registration.TransformationEstimationPointToPoint()
        #     )
        #
        #     cp = np.matmul(result.transformation, [[0], [0], [0], [1]])
        #
        #     plt.scatter(x, y, c="green")
        #     plt.scatter([cp[0]], cp[2], c="red")
        #     plt.xlim(-6, 6)
        #     plt.ylim(-6, 6)
        #     plt.gca().set_aspect('equal', adjustable='box')
        #     plt.draw()
        #     plt.show()
        #
        #     src_keypts.transform(result.transformation)
        #     # src_keypts.transform(result.transformation)
        #     src_camera.transform(result.transformation)
        #     open3d.visualization.draw_geometries([src_keypts, pcd, src_camera, grid])


if __name__ == '__main__':
    main()
