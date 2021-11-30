import open3d
import numpy as np
import matplotlib.pyplot as plt


pcd_file = "D:/Projects/Research/localization_experiments/data/DatasetV2/Primary/06/lidar_1637299401488642900.pcd"
feature_file = "D:/Projects/Research/localization_experiments/data/Features/0.1/lidar_1637299401488642900.npz"

pcd = open3d.read_point_cloud(pcd_file)
pcd = open3d.voxel_down_sample(pcd, 0.1)

points = np.asarray(pcd.points)

filter = np.logical_and(points[:, 1] > 1, points[:, 1] < 1.2)

plt.scatter(points[filter][:, 0], points[filter][:, 2])
plt.show()
