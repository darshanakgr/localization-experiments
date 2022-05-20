from tkinter import N
import numpy as np
import pandas as pd
import argparse
import os

import utils.pointcloud as pcu


parser = argparse.ArgumentParser(description='Calculate the transformation error between two point clouds.')
parser.add_argument('--set', type=int, default=1, help='The set of the two point clouds.')
parser.add_argument('--src', type=float, default=0.1, help='The source voxel size.')
parser.add_argument('--tgt', type=float, default=0.1, help='The target voxel size.')

args = parser.parse_args()

df = pd.DataFrame(columns=["src_id", "tgt_id", "src_keypts", "tgt_keypts", "matches", "is_match", "rx", "ry", "rz"])

src_dl = args.src
tgt_dl = args.tgt

transformations = np.load("data/GroundTruthTransforms/ground_truth_transformations.npz")

for i in np.arange(transformations["source_ids"].shape[0]):
    source_id = transformations["source_ids"][i]
    target_id = transformations["target_ids"][i]
    is_match = transformations["correct_matches"][i]
    transformation = transformations["transformations"][i]
    path = transformations["paths"][i]

    # src_file = f"{path}/{source_id}/x{src_dl}/lidar_{source_id}.pcd"
    # tgt_file = f"{path}/{target_id}/x{tgt_dl}/lidar_{target_id}.pcd"

    # src_file = f"data/LiDAR/DatasetV4/{source_id}.pcd"
    # tgt_file = f"data/LiDAR/DatasetV4/{target_id}.pcd"

    src_features_file = f"data/Features/{src_dl}/{source_id}.npz"
    tgt_features_file = f"data/Features/{tgt_dl}/{target_id}.npz"

    # Load the descriptors and estimate the transformation parameters using RANSAC
    # src_pcd = pcu.read_pcd_file(src_file, voxel_size=src_dl)
    # tgt_pcd = pcu.read_pcd_file(tgt_file, voxel_size=tgt_dl)
    
    try:
        src_keypts, src_features, src_scores = pcu.read_features_file(src_features_file)
        tgt_keypts, tgt_features, tgt_scores = pcu.read_features_file(tgt_features_file)
    except FileNotFoundError:
        print("File not found:", (src_features_file, tgt_features_file))
        continue

    result_ransac = pcu.execute_global_registration(src_keypts, tgt_keypts, src_features, tgt_features, 0.05)

    gt = pcu.get_angles_from_transformation(transformation)
    ransac = pcu.get_angles_from_transformation(result_ransac.transformation)
    error = gt - ransac

    to_print = f"Samples: [{source_id}, {target_id}]\t"
    to_print += f"Keypts: [{len(src_keypts.points)}, {len(tgt_keypts.points)}]\t"
    to_print += f"No of matches: {len(result_ransac.correspondence_set)}\t"
    to_print += f"RX: {error[0]:.2f}, RY: {error[1]:.2f}, RZ: {error[2]:.2f}"
    print(to_print)

    df = df.append({
        "src_id": source_id,
        "tgt_id": target_id,
        "src_keypts": len(src_keypts.points),
        "tgt_keypts": len(tgt_keypts.points),
        "matches": len(result_ransac.correspondence_set),
        "is_match": is_match,
        "rx": error[0],
        "ry": error[1],
        "rz": error[2]
    }, ignore_index=True)

if not os.path.exists(f"data/Results/TransformationError/set_{args.set}"):
    os.mkdir(f"data/Results/TransformationError/set_{args.set}")

df.to_csv(f"data/Results/TransformationError/set_{args.set}/ransac_{src_dl}_{tgt_dl}.csv", index=False)
