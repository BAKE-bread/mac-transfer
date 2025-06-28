# teleop_vision_project/2_depth_calculation/depth_server.py

import threading
import queue
import time
from .depth_estimator import MockDepthEstimator

class DepthServer:
    """
    模拟深度计算服务器。
    它从一个输入队列获取左右图像对，调用深度估计算法，
    然后将左图、深度图和时间戳打包放入一个输出队列。
    """
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.estimator = MockDepthEstimator()
        self.is_running = False
        self.processing_thread = None
        print("DepthServer initialized.")

    def _processing_loop(self):
        """
        处理线程循环，不断从输入队列获取数据、计算深度并放入输出队列。
        """
        print("Depth processing thread started.")
        self.is_running = True
        while self.is_running:
            try:
                # 从采集队列获取数据
                input_packet = self.input_queue.get(timeout=1)
                
                timestamp_ns = input_packet['timestamp_ns']
                left_image = input_packet['left_image']
                right_image = input_packet['right_image']
                
                # --- 核心：调用深度估计算法 ---
                # 在真实场景中，这里可能是个耗时的操作
                start_time = time.perf_counter()
                depth_map = self.estimator.calculate_depth(left_image, right_image)
                end_time = time.perf_counter()
                # print(f"Depth estimation took {(end_time - start_time) * 1000:.2f} ms")

                # 创建新的数据包，包含左图和深度图
                output_packet = {
                    "timestamp_ns": timestamp_ns,
                    "left_image": left_image,
                    "depth_map": depth_map # 16-bit 浮点数格式
                }

                try:
                    self.output_queue.put(output_packet, block=False)
                except queue.Full:
                    # print("Warning: Output queue for streaming is full. Dropping a frame.")
                    pass

            except queue.Empty:
                # 如果输入队列为空，继续等待
                continue
            
        print("Depth processing thread stopped.")

    def start(self):
        """
        启动深度处理线程。
        """
        if self.processing_thread is None:
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            print("Depth processing thread starting...")

    def stop(self):
        """
        停止深度处理线程。
        """
        self.is_running = False
        if self.processing_thread is not None:
            self.processing_thread.join()
        print("DepthServer stopped.")

