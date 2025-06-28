# data_acquisition/zed_capture.py (最终正确版)

import pyzed.sl as sl
import threading

class ZedCapture:
    """
    负责从ZED相机采集同步的、经过校正的左右眼图像。
    采用直接拉取模式，调用grab_synced_frames()即可获得一帧数据。
    """
    def __init__(self, resolution: sl.RESOLUTION = sl.RESOLUTION.HD720, fps: int = 30):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise IOError(f"ZED camera initialization failed. Error code: {err}")

        self.runtime_params = sl.RuntimeParameters()
        self.left_image_mat = sl.Mat()
        self.right_image_mat = sl.Mat()
        # 使用线程锁来确保grab操作的线程安全
        self.lock = threading.Lock()
        print("ZedCapture (Pull Mode) initialized successfully.")

    def grab_synced_frames(self):
        """
        执行一次抓取，并返回包含图像和时间戳的字典。
        这是一个阻塞操作，设计为在独立的线程执行器中被调用。
        """
        with self.lock:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                self.zed.retrieve_image(self.left_image_mat, sl.VIEW.LEFT)
                self.zed.retrieve_image(self.right_image_mat, sl.VIEW.RIGHT)
                
                data_packet = {
                    "timestamp_ns": timestamp.get_nanoseconds(),
                    "left_image": self.left_image_mat.get_data().copy(),
                    "right_image": self.right_image_mat.get_data().copy()
                }
                return data_packet
        return None

    def close(self):
        """关闭相机."""
        print("Closing ZED camera.")
        self.zed.close()