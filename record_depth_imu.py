import os.path
from time import time_ns, time
from threading import Thread
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs


class IMUStream(Thread):
    def __init__(self, stream, suffix, out_dir):
        super(IMUStream, self).__init__()
        self.__suffix = suffix
        self.__out_dir = out_dir

        self.__pipeline = rs.pipeline()
        self.__config = rs.config()

        self.__config.enable_stream(stream, rs.format.motion_xyz32f, 200)

        # Start streaming
        self.__pipeline.start(self.__config)

        self.__is_running = True

    @staticmethod
    def extract_motion_data(data):
        return np.array([time_ns(), data.x, data.y, data.z])

    def terminate(self):
        self.__is_running = False

    def run(self):
        try:
            data = np.array([]).reshape(0, 4)
            while self.__is_running:
                # Wait for a coherent pair of frames: depth and color
                frames: rs.composite_frame = self.__pipeline.wait_for_frames()

                motion_frame = frames[0].as_motion_frame()
                if not motion_frame:
                    continue
                motion = IMUStream.extract_motion_data(motion_frame.get_motion_data())
                data = np.concatenate([data, [motion]], axis=0)

            df = pd.DataFrame(data, columns=["t", "x", "y", "z"])
            df.to_csv(f"{self.__out_dir}/{self.__suffix}.csv")
        finally:
            # Stop streaming
            self.__pipeline.stop()


class LiDARCamera(Thread):
    def __init__(self, out_dir, sequence):
        super(LiDARCamera, self).__init__()
        rgb_width, rgb_height = 1280, 720
        depth_width, depth_height = 1024, 768

        self.__out_dir = os.path.join(out_dir, f"{sequence:02d}")

        if not os.path.exists(self.__out_dir):
            os.mkdir(self.__out_dir)

        self.__pipeline = rs.pipeline()
        self.__config = rs.config()
        self.__config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)
        self.__config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, 30)

        # Start streaming
        self.__pipeline.start(self.__config)
        self.__accel_stream = IMUStream(stream=rs.stream.accel, suffix="accelerometer", out_dir=self.__out_dir)
        self.__gyro_stream = IMUStream(stream=rs.stream.gyro, suffix="gyroscope", out_dir=self.__out_dir)

    def get_width_and_height(self):
        # Get stream profile and camera intrinsics
        profile = self.__pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        return w, h

    def run(self):
        w, h = self.get_width_and_height()

        self.__accel_stream.start()
        self.__gyro_stream.start()

        # Processing blocks
        pc = rs.pointcloud()

        cv2.namedWindow("LiDAR Viewer", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("LiDAR Viewer", w, h)

        record_start_time = time()
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames: rs.composite_frame = self.__pipeline.wait_for_frames()

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
                    np.save("%s/lidar_%s.npy" % (self.__out_dir, timestamp), vertices)
                    np.save("%s/depth_map_%s.npy" % (self.__out_dir, timestamp), depth_image)
                    cv2.imwrite("%s/rgb_image_%s.png" % (self.__out_dir, timestamp), color_image)
                    cv2.imwrite("%s/depth_map_%s.png" % (self.__out_dir, timestamp), depth_colormap)

                if key in (27, ord("q")) or cv2.getWindowProperty("LiDAR Viewer", cv2.WND_PROP_AUTOSIZE) < 0:
                    self.__gyro_stream.terminate()
                    self.__accel_stream.terminate()
                    print("Duration:", (time() - record_start_time) / 60)
                    break

        finally:
            # Stop streaming
            self.__pipeline.stop()


if __name__ == '__main__':
    LiDARCamera(out_dir="data/DatasetV1/Sequence", sequence=5).start()
