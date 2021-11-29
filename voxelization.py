import open3d
import numpy as np
import os


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
         open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        open3d.RANSACConvergenceCriteria(4000000, 600))
    return result


def main():
    cell_size = 3

    pcd_file = "D:/Projects/Research/localization_experiments/data/DatasetV2/Primary/07/lidar_1637299421190934300.pcd"
    feature_file = "D:/Projects/Research/localization_experiments/data/Features/0.1/lidar_1637299421190934300.npz"

    pcd = open3d.read_point_cloud(pcd_file)
    pcd = open3d.voxel_down_sample(pcd, 0.1)
    pcd.paint_uniform_color(rgb(149, 165, 166))

    grid_points = get_grid(pcd, cell_size)
    grid = make_pcd(grid_points)

    for tgt_file_name in os.listdir("data/DatasetV2/Secondary/04/"):
        print(f"Matching -> {'lidar_1637299421190934300'} & {tgt_file_name}")
        # src_file = f"data/DatasetV2/Secondary/03/{tgt_file_name}"
        highest_matches, highest_p = 0, None
        for p in grid_points:
            src_feature_file = os.path.join("data/Features/0.1/", tgt_file_name.replace("pcd", "npz"))
            src_keypts, src_features, src_scores = get_features(src_feature_file)
            src_keypts.paint_uniform_color([1, 0.706, 0])

            tgt_keypts, tgt_features, tgt_scores = get_cell_features(feature_file, p, cell_size)

            if len(tgt_keypts.points) < 2000:
                continue

            tgt_keypts.paint_uniform_color([0, 0.651, 0.929])

            result_ransac = registration(src_keypts, tgt_keypts, src_features, tgt_features, 0.05)
            to_print = f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]\t"
            to_print += f"No of matches: {len(result_ransac.correspondence_set)}\t"
            to_print += f"Inlier RMSE: {result_ransac.inlier_rmse}"
            print(to_print)

            if highest_matches < len(result_ransac.correspondence_set):
                highest_matches = len(result_ransac.correspondence_set)
                highest_p = p

            # src_keypts.transform(result_ransac.transformation)
            #
            # open3d.visualization.draw_geometries([src_keypts, tgt_keypts, grid])
        if highest_p is not None:
            src_feature_file = os.path.join("data/Features/0.1/", tgt_file_name.replace("pcd", "npz"))
            src_keypts, src_features, src_scores = get_features(src_feature_file)
            src_keypts.paint_uniform_color([1, 0.706, 0])

            tgt_keypts, tgt_features, tgt_scores = get_cell_features(feature_file, highest_p, cell_size)
            tgt_keypts.paint_uniform_color([0, 0.651, 0.929])

            result_ransac = registration(src_keypts, tgt_keypts, src_features, tgt_features, 0.05)
            to_print = f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]\t"
            to_print += f"No of matches: {len(result_ransac.correspondence_set)}\t"
            to_print += f"Inlier RMSE: {result_ransac.inlier_rmse}"
            print(to_print)

            src_keypts.transform(result_ransac.transformation)
            open3d.visualization.draw_geometries([src_keypts, tgt_keypts, grid])


if __name__ == '__main__':
    main()
