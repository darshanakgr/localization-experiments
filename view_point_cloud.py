import open3d


def main():
    pcd_file = "sample.pcd"
    pcd = open3d.io.read_point_cloud(pcd_file)
    # pcd = open3d.voxel_down_sample(pcd, 0.1)
    open3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()