import open3d
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from utils.pcd import get_angles_from_transformation, get_translation_from_transformation


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
    features = open3d.registration.Feature()
    features.data = data["features"].T
    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"])
    return keypts, features, scores


def get_cell_features(feature_file, p, cell_size):
    data = np.load(feature_file)
    scores = data["scores"]

    f = filter_indices(data["keypts"], p, cell_size)
    f = np.where(f)[0]

    # f = np.random.choice(f, 500, replace=False)

    features = open3d.registration.Feature()
    features.data = data["features"][f].T

    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"][f])

    return keypts, features, scores


def get_limits(pcd):
    pcd_points = np.asarray(pcd.points)

    x_min, y_min, z_min = np.min(pcd_points, axis=0)
    x_max, y_max, z_max = np.max(pcd_points, axis=0)

    return x_min, x_max, y_min, y_max, z_min, z_max


def get_grid(pcd, cell_size):
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(pcd)
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
    print(f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]", end="\t")
    print(f"No of matches: {len(result_ransac.correspondence_set)}", end="\t")
    print(f"Inlier RMSE: {result_ransac.inlier_rmse:.4f}", end=end)


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


def check_limits(tx, ty, tz, x_min, x_max, y_min, y_max, z_min, z_max):
    return (x_min <= tx <= x_max) and (y_min <= ty <= y_max) and (z_min <= tz <= z_max)


def main():
    cell_size = 3
    logging = False

    logs = pd.DataFrame(columns=[
        "src_id", "tgt_pcd",
        "cx", "cy", "cz",
        "src_keypts", "tgt_keypts",
        "matches", "rmse",
        "rx", "ry", "rz",
        "tx", "ty", "tz"
    ])

    tgt_pcd_file = "data/DatasetV2/Primary/07/lidar_1637299421190934300.pcd"
    tgt_file_name = tgt_pcd_file.split('/')[-1]
    tgt_feature_file = f"data/FeaturesV8/0.05/{tgt_file_name.replace('pcd', 'npz')}"

    pcd = open3d.read_point_cloud(tgt_pcd_file)
    pcd = open3d.voxel_down_sample(pcd, 0.025)
    pcd.paint_uniform_color(rgb(149, 165, 166))

    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(pcd)
    y_min = 0
    y_max = 1.8

    grid_points = get_grid(pcd, cell_size)
    grid = make_pcd(grid_points)

    for src_file_name in os.listdir("data/DatasetV2/Secondary/04/"):
        src_feature_file = os.path.join("data/FeaturesV8/0.05/", src_file_name.replace("pcd", "npz"))
        best_score, best_p = 0, None
        t = None

        for p in grid_points:
            src_keypts, src_features, src_scores = get_features(src_feature_file)
            tgt_keypts, tgt_features, tgt_scores = get_cell_features(tgt_feature_file, p, cell_size)

            if len(tgt_keypts.points) < 2000:
                continue

            src_keypts.paint_uniform_color([1, 0.706, 0])
            tgt_keypts.paint_uniform_color([0, 0.651, 0.929])

            result_ransac = registration(src_keypts, tgt_keypts, src_features, tgt_features, 0.05)
            tx, ty, tz = get_translation_from_transformation(result_ransac.transformation)
            rx, ry, rz = get_angles_from_transformation(result_ransac.transformation)

            if logging:
                print_registration_result(src_keypts, tgt_keypts, result_ransac, end="\t")

                print(f"TX: {tx:.2f} TY: {ty:.2f} TZ: {tz:.2f}", end="\t")
                print(f"RX: {rx:.2f} RY: {ry:.2f} RZ: {rz:.2f}", end="\n")

                logs = logs.append({
                    "src_id": src_file_name.split(".")[0], "tgt_pcd": tgt_file_name.split(".")[0],
                    "cx": p[0], "cy": p[1], "cz": p[2],
                    "src_keypts": len(src_keypts.points), "tgt_keypts": len(tgt_keypts.points),
                    "matches": len(result_ransac.correspondence_set), "rmse": result_ransac.inlier_rmse,
                    "rx": rx, "ry": ry, "rz": rz,
                    "tx": tx, "ty": ty, "tz": tz
                }, ignore_index=True)

            else:
                print_registration_result(src_keypts, tgt_keypts, result_ransac, end="\n")

            if best_score < len(result_ransac.correspondence_set) and check_limits(tx, ty, tz, x_min, x_max, y_min, y_max, z_min, z_max):
                best_score = len(result_ransac.correspondence_set)
                best_p = p
                t = result_ransac.transformation

            # src_keypts.transform(result_ransac.transformation)
            # open3d.visualization.draw_geometries([src_keypts, tgt_keypts, grid])

        if best_p is not None:
            src_keypts, src_features, src_scores = get_features(src_feature_file)
            tgt_keypts, tgt_features, tgt_scores = get_cell_features(tgt_feature_file, best_p, cell_size)

            src_keypts.paint_uniform_color([1, 0.706, 0])
            tgt_keypts.paint_uniform_color([0, 0.651, 0.929])

            src_camera = create_camera(camera_width=0.25, color=[1, 0, 0])

            result = open3d.registration.registration_icp(
                src_keypts, pcd, 0.05, t, open3d.registration.TransformationEstimationPointToPoint()
            )

            # cp = np.matmul(result.transformation, [[0], [0], [0], [1]])

            # plt.scatter(x, y, c="green")
            # plt.scatter([cp[0]], cp[2], c="red")
            # plt.xlim(-6, 6)
            # plt.ylim(-6, 6)
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.draw()
            # plt.show()

            src_keypts.transform(result.transformation)
            src_camera.transform(result.transformation)
            open3d.visualization.draw_geometries([src_keypts, pcd, src_camera, grid])

    if logging:
        logs.to_csv(f"data/Logs/{tgt_file_name.replace('pcd', 'csv')}", index=False)


if __name__ == '__main__':
    main()
