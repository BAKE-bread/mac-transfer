# teleop_vision_project/4_teleoperation/vision_teleop.py

import sys
import os
import threading
import queue
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import ssl
import logging

# --- START PATH FIX ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
# --- END PATH FIX ---

# 关键修改：导入项目已有的、能工作的组件
from teleop_vision_project.data_acquisition.zed_capture import ZedCapture
from teleop_vision_project.remote_control.vuer_controller import UnifiedTeleopServer
from teleop_vision_project.remote_control.dynamixel.active_cam import DynamixelAgent 

# 设置日志
logging.basicConfig(level=logging.INFO)

# 全局变量
HEAD_POSE = np.eye(4)
HEAD_POSE_LOCK = threading.Lock()

def vuer_camera_move_callback(event, scene):
    """从Vuer更新全局头部姿态的回调函数"""
    global HEAD_POSE
    if event.etype == "CAMERA_MOVE":
        with HEAD_POSE_LOCK:
            # Vuer的矩阵是列主序，需要转置
            HEAD_POSE = np.array(event.value["camera"]["matrix"]).reshape(4, 4).T

def main():
    # 1. 初始化 ZED 相机
    capture = ZedCapture()
    capture_queue = capture.start()

    # 2. 初始化统一的服务器 (Vuer 和 WebRTC)
    server = UnifiedTeleopServer(
        capture_queue=capture_queue,
        project_root=PROJECT_ROOT, # 确保 PROJECT_ROOT 被正确定义
        camera_move_cb=vuer_camera_move_callback
    )
    
    # 3. 初始化 Dynamixel 舵机 (使用正确的方式)
    logging.info("Initializing Dynamixel servos using DynamixelAgent...")
    port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U241-if00-port0"
    
    try:
        # 使用和 test_dynamixel.py 一样的方式来创建 agent
        agent = DynamixelAgent(port=port)
        # 启用扭矩
        agent._robot.set_torque_mode(True)
        # 移动到中心位置 (发送弧度)
        agent._robot.command_joint_state([0, 0])
        time.sleep(1.0)
        logging.info("Dynamixel agent connected and servos centered.")
    except Exception as e:
        logging.error(f"Could not connect to Dynamixel servos: {e}")
        logging.error("Please ensure the port is correct and the config in active_cam.py is set.")
        # 清理
        if 'capture' in locals():
            capture.stop()
        return

    logging.info("\n" + "="*50 + "\nSYSTEM IS RUNNING\n" + "="*50)

    try:
        # 4. 开始主遥操作循环
        while True:
            with HEAD_POSE_LOCK:
                head_pose_copy = HEAD_POSE.copy()
            
            # 从姿态矩阵中提取欧拉角
            r_matrix = head_pose_copy[:3, :3]
            r = R.from_matrix(r_matrix)
            # 使用 'yxz' 顺序，通常yaw对应y轴旋转，pitch对应x轴
            yaw, pitch, roll = r.as_euler('yxz', degrees=False) # 获取弧度值

            # 直接使用弧度值进行控制 (DynamixelRobot 需要弧度)
            # 根据需要调整符号和缩放
            yaw_goal_rad = -yaw  # 水平方向，可能需要反向
            pitch_goal_rad = pitch # 垂直方向

            # 添加安全限制 (以弧度为单位)
            yaw_limit = np.deg2rad(45) # 水平限制 +/- 45度
            pitch_limit = np.deg2rad(45) # 垂直限制 +/- 45度
            
            yaw_goal_rad = np.clip(yaw_goal_rad, -yaw_limit, yaw_limit)
            pitch_goal_rad = np.clip(pitch_goal_rad, -pitch_limit, pitch_limit)
            
            # 发送指令 [水平, 垂直]
            command = [yaw_goal_rad, pitch_goal_rad]
            agent._robot.command_joint_state(command)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # 确保所有资源都被正确关闭
        if 'agent' in locals() and agent:
            logging.info("Returning servos to center and disabling torque.")
            agent._robot.command_joint_state([0, 0])
            time.sleep(0.5)
            agent._robot.set_torque_mode(False)
        if 'capture' in locals():
            capture.stop()
        logging.info("System shutdown complete.")

if __name__ == "__main__":
    # 你需要在这里定义 PROJECT_ROOT 或者从其他地方导入它
    # from pathlib import Path
    # PROJECT_ROOT = Path(__file__).resolve().parent
    main()