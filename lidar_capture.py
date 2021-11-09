from time import time_ns, time

import cv2
import numpy as np
import pyrealsense2 as rs

USB_SUPPORT = 3

if USB_SUPPORT > 2:
    rgb_width, rgb_height = 1280, 720
    depth_width, depth_height = 1024, 768
else:
    rgb_width, rgb_height = 640, 480
    depth_width, depth_height = 320, 240

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, 30)
config.enable_stream(rs.stream.accel)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()

cv2.namedWindow("LiDAR Viewer", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("LiDAR Viewer", w, h)
out_dir = "data/DatasetV1"

record_start_time = time()
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames: rs.composite_frame = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Convert depth frame to a point cloud
        points = pc.calculate(depth_frame)
        pc.map_to(depth_frame)

        # Point cloud data to arrays
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
        # View depth map
        cv2.imshow("LiDAR Viewer", depth_colormap)

        # To get key presses
        key = cv2.waitKey(1)

        if key == ord("s"):
            # Saving images
            timestamp = time_ns()
            np.save("%s/lidar_%d.npy" % (out_dir, timestamp), vertices)
            np.save("%s/depth_map_%d.npy" % (out_dir, timestamp), depth_image)
            cv2.imwrite("%s/rgb_image_%d.png" % (out_dir, timestamp), color_image)
            cv2.imwrite("%s/depth_map_%d.png" % (out_dir, timestamp), depth_colormap)

        if key in (27, ord("q")) or cv2.getWindowProperty("LiDAR Viewer", cv2.WND_PROP_AUTOSIZE) < 0:
            print("Duration:", (time() - record_start_time) / 60)
            break

finally:
    # Stop streaming
    pipeline.stop()
