import pyrealsense2 as rs
import numpy as np
import cv2
import os

from time import time_ns

out_dir = "data/DatasetV3/sample_data"

if not os.path.exists(out_dir): os.makedirs(out_dir)

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
# depth_intrinsics = depth_profile.get_intrinsics()

# print(depth_intrinsics.width, depth_intrinsics.height, depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)

# clipping_distance_in_meters = 1 #1 meter
# clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()

count = 0
time_t = time_ns()

try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image

        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Convert depth frame to a point cloud
        points = pc.calculate(aligned_depth_frame)
        pc.map_to(aligned_color_frame)

        # Point cloud data to arrays
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow("Aligned RGB & Depth", cv2.WINDOW_AUTOSIZE)
        
        cv2.imshow("Aligned RGB & Depth", images)
        
        if time_ns() - time_t > 1000000000:
            time_t = time_ns()
            
            cv2.imwrite(f"{out_dir}/frame-{count:06d}.color.png", color_image)
            cv2.imwrite(f"{out_dir}/frame-{count:06d}.depth.png", depth_image)
            np.save(f"{out_dir}/frame-{count:06d}.pcd.npy", vertices)
            
            count += 1

        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        
        if key == ord("s"):
            # Saving images
            cv2.imwrite(f"{out_dir}/frame-{count:06d}.color.png", color_image)
            cv2.imwrite(f"{out_dir}/frame-{count:06d}.depth.png", depth_image)
            
            count += 1
        
finally:
    pipeline.stop()
