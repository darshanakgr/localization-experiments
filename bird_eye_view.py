import os

import numpy as np

# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
import open3d
import cv2
import matplotlib.pyplot as plt


def scale_to_255(a, vmin, vmax, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - vmin) / float(vmax - vmin)) * 255).astype(dtype)


# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_birdseye(points, res=0.05):
    """ Creates an 2D birds eye view representation of the point cloud data.
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 2]
    y_points = points[:, 0]
    z_points = points[:, 1]

    fwd_range = (np.min(x_points), np.max(x_points))
    side_range = (np.min(y_points), np.max(y_points))
    height_range = (np.min(z_points), np.max(z_points))

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                vmin=height_range[0],
                                vmax=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im


def bev(pcd):
    points = np.asarray(pcd.points)

    filter = np.logical_and(points[:, 1] > 0.3, points[:, 1] < 0.7)

    x = points[filter][:, 0]
    y = points[filter][:, 2]

    plt.scatter(x, y)
    plt.show()


def find_eq_plane(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d


def check_point_on_plane(p, eq):
    return eq[3] - np.dot(p, eq[:3].T)


def main():
    pcd_file = "data/DatasetV2/Primary/06/lidar_1637299401488642900.pcd"
    # pcd_file = "data/DatasetV2/Secondary/03/lidar_1636709742297907800.ply"
    pcd = open3d.read_point_cloud(pcd_file)
    pcd = open3d.voxel_down_sample(pcd, 0.01)
    open3d.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    img = point_cloud_2_birdseye(points, 0.01)
    cv2.imwrite("data/BEV/primary_1.png", img)
    plt.imshow(img)
    plt.show()
    # bev(pcd)

    # pcd_file = "data/DatasetV2/Secondary/03/lidar_1636709742297907800.ply"
    # pcd = open3d.read_point_cloud(pcd_file)
    # pcd = open3d.voxel_down_sample(pcd, 0.03)
    #
    #
    # cl, ind = open3d.geometry.statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0)
    # pcd = open3d.geometry.select_down_sample(pcd, ind)
    #
    # points = np.asarray(pcd.points)

    # colors = [[0, 0, 1] if p[1] > 0 else [0, 1, 0] for p in points]
    # pcd.colors = open3d.utility.Vector3dVector(colors)

    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]
    # xmax, xmin = np.max(x), np.min(x)
    # ymax, ymin = np.max(y), np.min(y)
    # zmax, zmin = np.max(z), np.min(z)
    #
    # p = [[xmin, 0, 0], [xmax, 0, 0], [0, ymin, 0], [0, ymax, 0], [0, 0, zmin], [0, 0, zmax]]
    # l = [[0, 1], [2, 3], [4, 5]]
    # c = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #
    # line_set = open3d.geometry.LineSet()
    # line_set.points = open3d.utility.Vector3dVector(p)
    # line_set.lines = open3d.utility.Vector2iVector(l)
    # line_set.colors = open3d.utility.Vector3dVector(c)
    #
    # open3d.draw_geometries([pcd, line_set])

    # combinations = np.random.choice(np.arange(points.shape[0]), (3000, 3), replace=False)
    # highest_ratio = 0
    # highest_eq = None
    # for i, j, k in combinations:
    #     p1, p2, p3 = points[[i, j, k]]
    #     a, b, c, d = find_eq_plane(p1, p2, p3)
    #     eq = np.array([a, b, c, d])
    #     diff = check_point_on_plane(points, eq)
    #     ratio = np.count_nonzero(diff < 0.001) / points.shape[0]
    #     if ratio > highest_ratio:
    #         highest_ratio = ratio
    #         highest_eq = eq
    #
    # print(f'Equation: {highest_eq[0]}x + {highest_eq[1]}y + {highest_eq[2]}z = {highest_eq[3]} -> {highest_ratio}')
    #
    # # pcd.paint_uniform_color([1, 0.706, 0])
    # colors = [[0, 0, 1] if check_point_on_plane(p, highest_eq) < 0.001 else [0, 1, 0] for p in points]
    # pcd.colors = open3d.utility.Vector3dVector(colors)
    #
    # open3d.draw_geometries([pcd, line_set])
    # p1 = np.array([1, 2, 3])
    # p2 = np.array([4, 6, 9])
    # p3 = np.array([12, 11, 9])
    #
    # a, b, c, d = find_eq_plane(p1, p2, p3)
    #
    # eq = np.array([a, b, c, d])
    #
    # print(f'Equation: {a}x + {b}y + {c}z = {d}')
    # print(f'{p1} on {eq}: {check_point_on_plane(p1, eq)}')


# data_dir = "D:/Projects/Research/localization_experiments/data/DatasetV2/Secondary/04/"
# for i in os.listdir(data_dir):
#
#     pcd_file = os.path.join(data_dir, i)
#
#     pcd = open3d.read_point_cloud(pcd_file)
#     pcd = open3d.voxel_down_sample(pcd, 0.1)
#     open3d.draw_geometries([pcd])
#     points = np.asarray(pcd.points)
#
#     img = point_cloud_2_birdseye(points, 0.01)
#
#     plt.imshow(img)
#     plt.show()
# cv2.imwrite("data/BEV/lidar_1637299421190934300.png", img)

if __name__ == '__main__':
    main()
