# teleop_vision_project/2_depth_calculation/depth_estimator.py

import numpy as np
import time

class MockDepthEstimator:
    """
    一个模拟的深度估计算法类。
    在实际应用中，您应该用您自己的、效果好的深度估计算法替换这里的逻辑。
    """
    def __init__(self):
        print("MockDepthEstimator initialized.")
        self.is_first_run = True

    def calculate_depth(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """
        接收左右图像，返回一个模拟的16位浮点数深度图。

        Args:
            left_image: 左眼图像 (H, W, C)
            right_image: 右眼图像 (H, W, C)

        Returns:
            一个与左眼图像对齐的深度图 (H, W)，数据类型为 np.float16。
        """
        if self.is_first_run:
            print("Running MOCK depth estimation. This should be replaced with a real algorithm.")
            self.is_first_run = False
            
        # 模拟计算延迟
        time.sleep(0.02) # 模拟20ms的计算时间

        height, width, _ = left_image.shape
        
        # --- 模拟深度图生成 ---
        # 创建一个从左到右的水平梯度作为深度图的简单模拟
        # 现实世界中，近处物体视差大，远处物体视差小。
        # 这里我们简单模拟：图像左侧（x小）深度值小（近），右侧（x大）深度值大（远）。
        gradient = np.linspace(0.5, 10.0, width) # 模拟0.5米到10米的深度
        depth_map = np.tile(gradient, (height, 1))

        # 增加一些随机噪声来让它看起来更真实一点
        noise = np.random.rand(height, width) * 0.1
        depth_map += noise
        
        # 返回16位浮点数格式的深度图
        return depth_map.astype(np.float16)

