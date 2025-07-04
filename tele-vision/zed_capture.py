# teleop_vision_project/zed_capture.py

import pyzed.sl as sl
import numpy as np
import logging
from dataclasses import dataclass
from typing import Union # 1. 导入 Union

@dataclass
class ZEDFramePacket:
    """
    一个数据类，用于封装从ZED相机单次抓取的所有相关数据。
    """
    left_image: np.ndarray
    right_image: np.ndarray
    timestamp_ns: int # 硬件纳秒级时间戳

class ZEDCapture:
    def __init__(self, resolution=sl.RESOLUTION.HD720, fps=30):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        self.runtime_params = sl.RuntimeParameters()
        self.left_image_mat = sl.Mat()
        self.right_image_mat = sl.Mat()

    def open(self):
        """打开并初始化ZED相机。"""
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logging.error(f"ZED Camera Error: {repr(err)}")
            self.zed.close()
            raise ConnectionError("Failed to open ZED camera")
        logging.info("ZED camera initialized successfully.")
        logging.info(f"  - Resolution: {self.zed.get_camera_information().camera_configuration.resolution}")
        logging.info(f"  - FPS: {self.zed.get_camera_information().camera_configuration.fps}")

    # 2. 将 ZEDFramePacket | None 替换为 Union[ZEDFramePacket, None]
    def grab_frame_packet(self) -> Union[ZEDFramePacket, None]:
        """
        抓取一帧同步的左右图像，并附带硬件时间戳。
        这是实现同步的核心函数。
        """
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            
            self.zed.retrieve_image(self.left_image_mat, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.right_image_mat, sl.VIEW.RIGHT)
            
            left_image_data = self.left_image_mat.get_data().copy()
            right_image_data = self.right_image_mat.get_data().copy()

            return ZEDFramePacket(
                left_image=left_image_data,
                right_image=right_image_data,
                timestamp_ns=timestamp
            )
        return None

    def close(self):
        """关闭相机。"""
        if self.zed.is_opened():
            self.zed.close()
            logging.info("ZED camera closed.")