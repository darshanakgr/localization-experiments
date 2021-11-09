import open3d


def main():
    pcd_file = "data/DatasetV1/lidar_1636428659366014800.pcd"
    pcd = open3d.read_point_cloud(pcd_file)
    open3d.draw_geometries([pcd])


if __name__ == '__main__':
    main()