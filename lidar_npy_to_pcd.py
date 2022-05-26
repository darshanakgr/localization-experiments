import os

import shutil
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import pptk
import tqdm

from utils import pcd


def main():
    # directory = "data/DatasetV1/Secondary/05"
    # out_dir = "data/DatasetV2/Secondary/05"
    # npy_files = glob.glob(os.path.join(directory, "lidar_*.npy"))
    # for file in tqdm.tqdm(npy_files):
    #     npy_pcd = np.load(file)
    #     file_name = file.split("\\")[-1]
    #     pcd.convert_to_pcd(npy_pcd, os.path.join(out_dir, file_name.replace(".npy", ".pcd")))

    dataset_dir = "data/DatasetV1"
    out_dir = "data/DatasetV2"
    for i in os.listdir(dataset_dir):
        if i != "Sequence":
            continue
        for j in os.listdir(os.path.join(dataset_dir, i)):
            if os.path.exists(os.path.join(out_dir, i, j)): continue    
        
            npy_files = glob.glob(os.path.join(dataset_dir, i, j, "lidar_*.npy"))
            for file in tqdm.tqdm(npy_files):
                npy_pcd = np.load(file)
                file_name = file.split("\\")[-1]
                if not os.path.exists(os.path.join(out_dir, i, j)):
                    os.makedirs(os.path.join(out_dir, i, j))
                pcd.convert_to_pcd(npy_pcd, os.path.join(out_dir, i, j, file_name.replace(".npy", ".pcd")))


if __name__ == '__main__':
    main()
