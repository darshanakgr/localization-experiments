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
    directory = "data/DatasetV1"
    npy_files = glob.glob(os.path.join(directory, "lidar_*.npy"))
    for file in tqdm.tqdm(npy_files):
        npy_pcd = np.load(file)
        file_name = file.split("\\")[-1]
        pcd.convert_to_pcd(npy_pcd, os.path.join(directory, file_name.replace(".npy", ".pcd")))


if __name__ == '__main__':
    main()
