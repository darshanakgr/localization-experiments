import numpy as np
import open3d


def main():
    # pcd_file = "sample.pcd"
    # pcd = open3d.io.read_point_cloud(pcd_file)
    # pcd = open3d.voxel_down_sample(pcd, 0.1)
    # open3d.visualization.draw_geometries([pcd])

    src_pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud("dataset/1647331687234192900.pcd")
    tgt_pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud("dataset/1647331703080928000.pcd")

    src_pose = np.load("dataset/1647331687234192900.npy")
    tgt_pose = np.load("dataset/1647331703080928000.npy")

    src_pcd.paint_uniform_color([1, 0.706, 0])
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])

    src_inv = np.diag(np.ones(4))
    src_inv[0:3, 0:3] = np.linalg.inv(src_pose[0:3, 0:3])

    tgt_inv = np.diag(np.ones(4))
    tgt_inv[0:3, 0:3] = np.linalg.inv(tgt_pose[0:3, 0:3])

    # src_inv = np.linalg.inv(src_pose)
    # tgt_inv = np.linalg.inv(tgt_pose)

    # print(src_pose)
    # print(src_inv)

    src_pcd = src_pcd.transform(src_pose)
    tgt_pcd = tgt_pcd.transform(tgt_pose)

    open3d.visualization.draw_geometries([src_pcd, tgt_pcd])


if __name__ == '__main__':
    main()
