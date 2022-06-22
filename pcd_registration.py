from utils import pcd
import open3d
import os
import numpy as np
import matplotlib.pyplot as plt


def main():

    # voxel_size = 0.1
    # secondary_folder = "data/DatasetV2/Secondary/05/"
    # tgt_file = "data/DatasetV2/Primary/10/lidar_1638353640807327100.pcd"

    # for tgt_file_name in os.listdir(secondary_folder):
    #     src_file = os.path.join(secondary_folder, tgt_file_name)

    #     # Load the descriptors and estimate the transformation parameters using RANSAC
    #     src_pcd, src_keypts, src_features, src_scores = pcd.read_pcd_and_features(src_file, "data/FeaturesV6", voxel_size)
    #     tgt_pcd, tgt_keypts, tgt_features, tgt_scores = pcd.read_pcd_and_features(tgt_file, "data/FeaturesV6", voxel_size)

    #     # open3d.geometry.estimate_normals(
    #     #     src_pcd, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30),
    #     # )
    #     # open3d.geometry.estimate_normals(
    #     #     tgt_pcd, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
    #     # )

    #     # Matching with RANSAC
    #     result_ransac = pcd.execute_global_registration(src_keypts, tgt_keypts, src_features, tgt_features, voxel_size)
    #     to_print = f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]\t"
    #     to_print += f"No of matches: {len(result_ransac.correspondence_set)}"
    #     print(to_print)

    #     # Plot point clouds after registration
    #     pcd.draw_registration_result(src_keypts, tgt_keypts, result_ransac.transformation)

    
    src_keypts, src_features, src_scores = pcd.read_features_file("data/samples2/frame-000200.npz")
    tgt_keypts, tgt_features, tgt_scores = pcd.read_features_file("data/samples3/frame-000150.npz")
    
    result_ransac = pcd.execute_global_registration(src_keypts, tgt_keypts, src_features, tgt_features, 0.05)
    to_print = f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]\t"
    to_print += f"No of matches: {len(result_ransac.correspondence_set)}"
    print(to_print)
    
    src_keypts.paint_uniform_color([1, 0.706, 0])
    tgt_keypts.paint_uniform_color([0, 0.651, 0.929])
    
    # open3d.visualization.draw_geometries([src_keypts, tgt_keypts])
    
    src_keypts.transform(result_ransac.transformation)
    
    open3d.visualization.draw_geometries([src_keypts, tgt_keypts])

if __name__ == '__main__':
    main()
