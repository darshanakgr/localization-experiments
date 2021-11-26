import os
import glob
import shutil
from datetime import time
from time import time_ns

import open3d

from utils import pcd


def main():
    # dataset_dir = "data/DatasetV1"
    # for i in os.listdir(dataset_dir):
    #     for j in os.listdir(os.path.join(dataset_dir, i)):
    #         sub_dir = os.path.join(dataset_dir, i, j)
    #         timestamps = [f.split("_")[-1].split(".")[0] for f in os.listdir(sub_dir)]
    #         for t in timestamps:
    #             files = glob.glob(f"data/DatasetV1/*_{t}.*")
    #             for f in files:
    #                 file_name = f.split("\\")[-1]
    #                 shutil.copy2(f, os.path.join(sub_dir, file_name))

    x = pcd.remove_color("data/IPadLIDAR/meeting_room_1.pcd")
    print(len(x.points))
    open3d.write_point_cloud(f"data/DatasetV2/Primary/08/lidar_{time_ns()}.pcd", x)

    x = open3d.read_point_cloud("data/DatasetV2/Primary/08/lidar_1637905144483014800.pcd")
    open3d.draw_geometries([x])


if __name__ == '__main__':
    main()
