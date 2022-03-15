# import json
# import numpy as np
# import open3d as o3d
#
# CUBOID_EXTENT_METERS = 200
#
# METERS_BELOW_START = 5
# METERS_ABOVE_START = 30
#
#
# def main():
#     ## Point Cloud
#     points = np.array([
#         ## These points lie inside the cuboid
#         [-2770.94365061042, 722.0595600050154, -20.004812609192445],
#         [-2755.94365061042, 710.0595600050154, -20.004812609192445],
#         [-2755.94365061042, 710.0595600050154, -15.004812609192445],
#
#         ## These points lie outside the cuboid
#         [-2755.94365061042 + CUBOID_EXTENT_METERS, 710.0595600050154, -15.004812609192445],
#         [-2755.94365061042, 710.0595600050154 + CUBOID_EXTENT_METERS, -15.004812609192445],
#     ]).reshape([-1, 3])
#
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#
#     ## Start point here corresponds to an ego vehicle position start in a point cloud
#     start_position = {'x': -2755.94365061042, 'y': 722.0595600050154, 'z': -20.004812609192445}
#     cuboid_points = getCuboidPoints(start_position)
#
#     points = o3d.utility.Vector3dVector(cuboid_points)
#     oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
#     point_cloud_crop = point_cloud.crop(oriented_bounding_box)
#
#     # View original point cloud with the cuboid, all 5 points present
#     o3d.visualization.draw_geometries([point_cloud, oriented_bounding_box])
#
#     # View cropped point cloud with the cuboid, only 3 points present
#     o3d.visualization.draw_geometries([point_cloud_crop, oriented_bounding_box])
#
#
# def getCuboidPoints(start_position):
#     return np.array([
#         # Vertices Polygon1
#         [start_position['x'] + (CUBOID_EXTENT_METERS / 2), start_position['y'] + (CUBOID_EXTENT_METERS / 2),
#          start_position['z'] + METERS_ABOVE_START],  # face-topright
#         [start_position['x'] - (CUBOID_EXTENT_METERS / 2), start_position['y'] + (CUBOID_EXTENT_METERS / 2),
#          start_position['z'] + METERS_ABOVE_START],  # face-topleft
#         [start_position['x'] - (CUBOID_EXTENT_METERS / 2), start_position['y'] - (CUBOID_EXTENT_METERS / 2),
#          start_position['z'] + METERS_ABOVE_START],  # rear-topleft
#         [start_position['x'] + (CUBOID_EXTENT_METERS / 2), start_position['y'] - (CUBOID_EXTENT_METERS / 2),
#          start_position['z'] + METERS_ABOVE_START],  # rear-topright
#
#         # Vertices Polygon 2
#         [start_position['x'] + (CUBOID_EXTENT_METERS / 2), start_position['y'] + (CUBOID_EXTENT_METERS / 2),
#          start_position['z'] - METERS_BELOW_START],
#         [start_position['x'] - (CUBOID_EXTENT_METERS / 2), start_position['y'] + (CUBOID_EXTENT_METERS / 2),
#          start_position['z'] - METERS_BELOW_START],
#         [start_position['x'] - (CUBOID_EXTENT_METERS / 2), start_position['y'] - (CUBOID_EXTENT_METERS / 2),
#          start_position['z'] - METERS_BELOW_START],
#         [start_position['x'] + (CUBOID_EXTENT_METERS / 2), start_position['y'] - (CUBOID_EXTENT_METERS / 2),
#          start_position['z'] - METERS_BELOW_START],
#     ]).astype("float64")
#
#
# if __name__ == '__main__':
#     main()


# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/visualization/customized_visualization.py

import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_rotation(pcd):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)


def custom_draw_geometry_load_option(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("test_data/renderoption.json")
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_key_callback(pcd):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "test_data/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


def draw_geometry_with_camera_trajectory(pcd):
    draw_geometry_with_camera_trajectory.index = -1
    draw_geometry_with_camera_trajectory.trajectory = o3d.io.read_pinhole_camera_trajectory("test_data/camera_trajectory.json")
    draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()

    if not os.path.exists("test_data/image/"):
        os.makedirs("test_data/image/")
    if not os.path.exists("test_data/depth/"):
        os.makedirs("test_data/depth/")

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave("test_data/depth/{:05d}.png".format(glb.index), \
                       np.asarray(depth), dpi=1)
            plt.imsave("test_data/image/{:05d}.png".format(glb.index), \
                       np.asarray(image), dpi=1)
            # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            draw_geometry_with_camera_trajectory.vis. \
                register_animation_callback(None)
        return False

    vis = draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=1920, height=1080)
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("test_data/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("test_data/fragment.ply")

    print("6. Customized visualization playing a camera trajectory")
    draw_geometry_with_camera_trajectory(pcd)
