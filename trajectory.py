import open3d
import numpy as np
import math as m


def rotate_transformation_matrix(T, rx, ry, rz):
    RX = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])

    RY = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])

    RZ = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return np.dot(np.dot(np.dot(T, RZ), RY), RX)


def get_default_transformation_matrix():
    t = np.identity(4)
    t[1, 3] = 1
    t[0, 3] = 0
    t = rotate_transformation_matrix(t, np.radians(90), np.radians(90), np.radians(-90))
    return t


def get_current_transformation_matrix(vis):
    control = vis.get_view_control()
    cam = control.convert_to_pinhole_camera_parameters()
    return np.copy(np.asarray(cam.extrinsic))


def change_camera_extrinsic_params(vis, transformation):
    control = vis.get_view_control()
    cam = control.convert_to_pinhole_camera_parameters()
    cam.extrinsic = transformation
    control.convert_from_pinhole_camera_parameters(cam)


def move(vis, axis=0, direction=1, step=0.05):
    t = get_current_transformation_matrix(vis)
    t[axis, 3] = t[axis, 3] + direction * step
    change_camera_extrinsic_params(vis, t)


def rotate(vis, axis=0, direction=1, step=5):
    t = get_current_transformation_matrix(vis)
    if axis == 0:
        rx = np.radians(direction * step)
        ry, rz = 0, 0
    elif axis == 1:
        rz = np.radians(direction * step)
        rx, ry = 0, 0
    else:
        ry = np.radians(direction * step)
        rx, rz = 0, 0

    t = rotate_transformation_matrix(t, rx, ry, rz)
    change_camera_extrinsic_params(vis, t)


def move_through_pcd(pcd):
    """
        Axis info
        x - 0
        y - 2
        z - 1
    """

    def reset_view(vis):
        change_camera_extrinsic_params(vis, get_default_transformation_matrix())
        return False

    def move_x_forward(vis):
        move(vis, 0, 1, 0.05)
        return False

    def move_x_backward(vis):
        move(vis, 0, -1, 0.05)
        return False

    def move_y_forward(vis):
        move(vis, 2, 1, 0.05)
        return False

    def move_y_backward(vis):
        move(vis, 2, -1, 0.05)
        return False

    def rotate_y_forward(vis):
        rotate(vis, 2, 1, 5)
        return False

    def rotate_y_backward(vis):
        rotate(vis, 2, -1, 5)
        return False

    def rotate_z_forward(vis):
        rotate(vis, 1, 1, 5)
        return False

    def rotate_z_backward(vis):
        rotate(vis, 1, -1, 5)
        return False

    def capture_pcd(vis):
        vis.capture_depth_point_cloud("sample.pcd")
        return False

    key_to_callback = dict()
    key_to_callback[ord("R")] = reset_view

    key_to_callback[ord("W")] = move_y_backward
    key_to_callback[ord("S")] = move_y_forward
    key_to_callback[ord("A")] = move_x_forward
    key_to_callback[ord("D")] = move_x_backward

    key_to_callback[ord("J")] = rotate_y_forward
    key_to_callback[ord("L")] = rotate_y_backward
    key_to_callback[ord("I")] = rotate_z_forward
    key_to_callback[ord("K")] = rotate_z_backward

    key_to_callback[ord("C")] = capture_pcd
    open3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud("data/DatasetV2/Primary/06/lidar_1637299401488642900.pcd")
pcd.paint_uniform_color([1, 0.706, 0])
move_through_pcd(pcd)
