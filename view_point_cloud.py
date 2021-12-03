import open3d


def main():
    pcd_file = "data/DatasetV2/Primary/06/lidar_1637299401488642900.pcd"
    pcd = open3d.read_point_cloud(pcd_file)
    pcd = open3d.voxel_down_sample(pcd, 0.1)
    open3d.draw_geometries([pcd])


if __name__ == '__main__':
    main()