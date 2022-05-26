from ast import arg
from scipy.spatial.transform import Rotation as R

import open3d
import numpy as np
import pandas as pd
import os
import tqdm
import copy
import math
import argparse


def preprocess_pcd(pcd, voxel_size, down_sample=True):
    if down_sample:
        pcd = open3d.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    open3d.geometry.estimate_normals(pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = open3d.registration.compute_fpfh_feature(pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def execute_global_registration(source_down, target_down, source_feat, target_feat, n_ransac, threshold):
    result = open3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_feat, target_feat, threshold,
        open3d.registration.TransformationEstimationPointToPoint(False), n_ransac, 
        [open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), open3d.registration.CorrespondenceCheckerBasedOnDistance(threshold)],
        open3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, voxel_size, t):
    distance_threshold = voxel_size * 1
    result = open3d.registration.registration_icp(
        source, target, distance_threshold, t,
        open3d.registration.TransformationEstimationPointToPlane(),
        open3d.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    return result


def inv_transform(T):
    T_inv = np.identity(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -np.dot(T[:3, :3].T, T[:3, 3])
    return T_inv


def make_pcd(points, color=None):
    c = [1, 0, 0] if color is None else color
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector([c for i in range(len(points))])
    return pcd


def get_limits(pcd):
    pcd_points = np.asarray(pcd.points)

    x_min, y_min, z_min = np.min(pcd_points, axis=0)
    x_max, y_max, z_max = np.max(pcd_points, axis=0)

    return x_min, x_max, y_min, y_max, z_min, z_max


def get_grid(pcd, cell_size):
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(pcd)
    y_val = np.mean([y_min, y_max])

    points = []
    x_n = int((x_max - x_min) // cell_size)
    z_n = int((z_max - z_min) // cell_size)
    for i in range(z_n):
        z0 = float(z_min + cell_size * (i + 1))
        for j in range(x_n):
            x0 = float(x_min + cell_size * (j + 1))
            points.append([x0, y_val, z0])

    return points


def get_features(feature_file):
    data = np.load(feature_file)
    scores = data["scores"]
    features = open3d.registration.Feature()
    features.data = data["features"].T
    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"])
    return keypts, features, scores


def filter_indices(points, p, cell_size):
    px_min = p[0] - cell_size
    px_max = p[0] + cell_size
    pz_min = p[2] - cell_size
    pz_max = p[2] + cell_size
    xf = np.logical_and(points[:, 0] > px_min, points[:, 0] < px_max)
    zf = np.logical_and(points[:, 2] > pz_min, points[:, 2] < pz_max)
    return np.logical_and(xf, zf)


def get_cell_features(feature_file, p, cell_size):
    data = np.load(feature_file)
    scores = data["scores"]

    f = filter_indices(data["keypts"], p, cell_size)
    f = np.where(f)[0]

    features = open3d.registration.Feature()
    features.data = data["features"][f].T

    keypts = open3d.geometry.PointCloud()
    keypts.points = open3d.utility.Vector3dVector(data["keypts"][f])

    return keypts, features, scores


def rgb(r, g, b):
    return r / 255.0, g / 255.0, b / 255.0


def rotation(transformation):
    rm = transformation[:3, :3].tolist()
    return R.from_matrix(rm).as_euler('xzy', degrees=True)


def translation(t):
    return np.array([t[0][3], t[1][3], t[2][3]])


def inv_transform(T):
    T_inv = np.identity(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -np.dot(T[:3, :3].T, T[:3, 3])
    return T_inv

def calc_error(T1, T2):
    e1 = np.sum(np.abs(rotation(T1) - rotation(T2)))
    e2 = np.sum(np.abs(translation(T1) - translation(T2)))
    return e1, e2

def check(T1, T2, max_r=5, max_t=1):
    er, et = calc_error(T1, T2)
    return er < max_r and et < max_t

def validate(T1, T2, T3, t1, t2, max_dist, max_rot):
    c1 = check(T3, np.dot(T2, inv_transform(t2)), max_t=max_dist, max_r=max_rot)
    c2 = check(T3, np.dot(np.dot(T1, inv_transform(t1)), inv_transform(t2)), max_t=max_dist, max_r=max_rot)
    c3 = check(T2, np.dot(T1, inv_transform(t1)), max_t=max_dist, max_r=max_rot)

    print(f"Check 1: {c1}, Check 2: {c2}, Check 3: {c3}")
    
    # If two checks are true, the combination is wrong
    if (c1 + c2 + c3) == 2:
        raise Exception("Invalid combination")

    # If two checks are true, the combination is wrong
    if (c1 + c2 + c3) == 0:
        raise Exception("Invalid transformations")

    # If all the checks are valid, there is no need of correction
    if (c1 + c2 + c3) == 3:
        print(":: No need of correction.")
        return T1, T2, T3
    
    # If two checks are wrong, only one transformation needs correction
    if c1:
        print(":: Correcting Previous Transformation")
        T1 = np.dot(T2, t1)
    elif c2:
        print(":: Correcting Current Transformation")
        T2 = np.dot(T1, inv_transform(t1))
    else:
        print(":: Correcting Future Transformation")
        T3 = np.dot(T2, inv_transform(t2))

    return T1, T2, T3


parser = argparse.ArgumentParser(description="Registration pipeline based on FCGF and ICP")

parser.add_argument("--dataset_dir", type=str, default="data/DatasetV2/SequenceCombined", help="Dataset directory")
parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size")
parser.add_argument("--seq_id", type=int, default=6, help="Sequence ID")
parser.add_argument("--skip", type=int, default=5, help="Number of frames to skip between each registrations")
parser.add_argument("--location", type=str, default="kitchen", help="Primary location")
parser.add_argument("--n", type=int, default=10, help="Number of steps in the registration")

args = parser.parse_args()

sequence_dir = "{}/{:02d}".format(args.dataset_dir, args.seq_id)

print(":: Sequence directory: {}".format(sequence_dir))

sequence_files = os.listdir(sequence_dir)
sequence_files = sorted(sequence_files, key=lambda x: int(x.split("_")[2].split(".")[0]))

print(":: Performing FAST + ICP Registration")

cell_size = 3
local_t = []
global_t = []
local_pcds = []
global_pcds = []
transformation = np.identity(4)

for i in tqdm.trange(args.n):
    src_feature_file = os.path.join("data/FeaturesV8/0.05/", sequence_files[args.skip * i].replace("pcd", "npz"))
    tgt_feature_file = os.path.join("data/FeaturesV8/0.05/", sequence_files[args.skip * (i + 1)].replace("pcd", "npz"))

    src_keypts, src_features, src_scores = get_features(src_feature_file)
    tgt_keypts, tgt_features, tgt_scores = get_features(tgt_feature_file)
    
    source, source_fpfh = preprocess_pcd(src_keypts, args.voxel_size, down_sample=False)
    target, target_fpfh = preprocess_pcd(tgt_keypts, args.voxel_size, down_sample=False)

    source.paint_uniform_color(np.random.random(3).tolist())

    ransac_result = execute_global_registration(source, target, source_fpfh, target_fpfh, 4, args.voxel_size * 1.5)
    icp_result = refine_registration(source, target, 0.05, ransac_result.transformation)

    local_t.append(icp_result.transformation)
    
    transformation = np.dot(transformation, inv_transform(local_t[i - 1]) if i > 0 else np.identity(4))
    
    source.transform(transformation) 
    local_pcds.append(source)
  
print(":: Performing Global Registration with FCGF")

if args.location == "office":
    primary_pcd_file = "data/DatasetV2/Primary/07/lidar_1637299421190934300.pcd"
elif args.location == "kitchen":
    primary_pcd_file = "data/DatasetV2/Primary/06/lidar_1637299401488642900.pcd"

tgt_feature_file = f"data/FeaturesV8/0.05/{primary_pcd_file.split('/')[-1].replace('pcd', 'npz')}"

primary_pcd = open3d.read_point_cloud(primary_pcd_file)
primary_pcd = open3d.voxel_down_sample(primary_pcd, 0.025)
primary_pcd.paint_uniform_color(rgb(149, 165, 166))

grid_points = get_grid(primary_pcd, cell_size)
grid = make_pcd(grid_points)

for i in tqdm.trange(args.n):
    src_feature_file = os.path.join("data/FeaturesV8/0.05/", sequence_files[args.skip * i].replace("pcd", "npz"))
    best_score = 0
    best_p = None
    t = None

    for p in grid_points:
        src_keypts, src_features, src_scores = get_features(src_feature_file)
        tgt_keypts, tgt_features, tgt_scores = get_cell_features(tgt_feature_file, p, cell_size)

        if len(tgt_keypts.points) < 2000:
            continue

        result_ransac = execute_global_registration(src_keypts, tgt_keypts, src_features, tgt_features, 3, 0.05)
        
        if best_score < len(result_ransac.correspondence_set):
            best_score = len(result_ransac.correspondence_set)
            best_p = p
            t = result_ransac.transformation
            
    if best_p is not None:
        src_keypts, src_features, src_scores = get_features(src_feature_file)

        src_keypts.paint_uniform_color(np.random.random(3).tolist())

        global_t.append(t)

        src_keypts.transform(t)
        global_pcds.append(src_keypts)
        
global_pcds.append(primary_pcd)


open3d.visualization.draw_geometries(local_pcds)
open3d.visualization.draw_geometries(global_pcds)


print(":: Performing correction over global registration")
global_tc = copy.deepcopy(global_t)

for i in range(1, args.n - 1):
    print("====================={}-{}-{}=====================".format(i - 1, i, i + 1))
    try:
        global_tc[i - 1], global_tc[i], global_tc[i + 1] = validate(global_tc[i - 1], global_tc[i], global_tc[i + 1], local_t[i - 1], local_t[i], 1, 10)
    except Exception as e:
        print(e)
        

del local_pcds, global_pcds

print(":: Visualizing corrected global registration")

global_pcds = []
        
for i in range(1, args.n - 1):
    src_feature_file = os.path.join("data/FeaturesV8/0.05/", sequence_files[args.skip * i].replace("pcd", "npz"))
    
    src_keypts, src_features, src_scores = get_features(src_feature_file)
    
    src_keypts.transform(global_tc[i])
    src_keypts.paint_uniform_color(np.random.random(3).tolist())
    global_pcds.append(src_keypts)

global_pcds.append(primary_pcd)
    
open3d.visualization.draw_geometries(global_pcds)

trajectory = [global_tc[i][:3, 3].tolist() for i in range(args.n)]
lines = [[i, i + 1] for i in range(len(trajectory) - 1)]

colors = [[1, 0, 0] for i in range(len(lines))]
line_set = open3d.geometry.LineSet()
line_set.points = open3d.utility.Vector3dVector(trajectory)
line_set.lines = open3d.utility.Vector2iVector(lines)
line_set.colors = open3d.utility.Vector3dVector(colors)

open3d.visualization.draw_geometries([primary_pcd, line_set])