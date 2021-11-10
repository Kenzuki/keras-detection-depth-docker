import pyrealsense2 as rs
import numpy as np
from PIL import Image


class DepthCamera:
    def __init__(self):
        self.__pipeline = rs.pipeline()
        self.__config = rs.config()

        self.__pipeline_wrapper = rs.pipeline_wrapper(self.__pipeline)
        self.__pipeline_profile = self.__config.resolve(self.__pipeline_wrapper)
        self.__device = self.__pipeline_profile.get_device()
        self.__device_product_line = str(self.__device.get_info(rs.camera_info.product_line))

        self.__is_rgb_camera()

    def __del__(self):
        self.__pipeline.stop()

    def __is_rgb_camera(self):
        found_rgb = False

        for s in self.__device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break

        if not found_rgb:
            raise Exception("Depth camera with Color sensor is required.")

    def start(self):
        self.__config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.__device_product_line == 'L500':
            self.__config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.__config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.__pipeline.start(self.__config)

    def get_frame(self):
        try:
            frames: rs.composite_frame = self.__pipeline.wait_for_frames()
            depth_frame: rs.depth_frame = frames.get_depth_frame()
            color_frame: rs.video_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            color_image = np.asanyarray(color_frame.get_data())
            color_image = color_image[:, :, ::-1]
            rgb_frame = Image.fromarray(color_image)

            return rgb_frame, depth_frame

        except Exception as e:
            print(e)
            return None
