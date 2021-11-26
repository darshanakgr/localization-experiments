import os

import open3d

from utils import pcdv2 as pcd


def main():
    voxel_size = 0.1

    for tgt_file_name in os.listdir("data/DatasetV2/Secondary/00/"):
        tgt_file = "data/DatasetV2/Primary/01/lidar_1636437069389021900.pcd"
        src_file = f"data/DatasetV2/Secondary/00/{tgt_file_name}"

        src_pcd = pcd.read_point_cloud(src_file)
        tgt_pcd = pcd.read_point_cloud(tgt_file)

        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size)
        src_pcd = src_pcd.voxel_down_sample(voxel_size)
        src_pcd.paint_uniform_color([1, 0.706, 0])
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])

        src_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
        tgt_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))

        # Load the descriptors and estimate the transformation parameters using RANSAC
        src_keypts, src_features, src_scores = pcd.get_features(
            f"data/Features/{voxel_size}/{tgt_file_name.replace('pcd', 'npz')}"
        )
        tgt_keypts, tgt_features, tgt_scores = pcd.get_features(
            f"data/Features/{voxel_size}/lidar_1636437069389021900.npz"
        )

        # Matching with RANSAC
        result_ransac = pcd.registration(src_keypts, tgt_keypts, src_features, tgt_features, voxel_size)
        to_print = f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]\t"
        to_print += f"No of matches: {len(result_ransac.correspondence_set)}"
        print(to_print)

        src_pcd.transform(result_ransac.transformation)

        # Plot point clouds after registration
        open3d.visualization.draw_geometries([src_pcd, tgt_pcd])


if __name__ == '__main__':
    main()
