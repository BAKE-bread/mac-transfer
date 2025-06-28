# main.py (最终重构版)

import logging
import asyncio
import time
from threading import Thread, Event
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# 关键修改：导入新的WebRTC服务器和我们仍然需要的组件
from teleop_vision_project.data_acquisition.zed_capture import ZedCapture
from teleop_vision_project.remote_control.dynamixel.active_cam import DynamixelAgent 
from teleop_vision_project.webrtc_streaming.streaming_server import WebRTCServer # 路径可能需要调整

logging.basicConfig(level=logging.INFO)

# 舵机相关的函数保持不变
def initialize_agent():
    try:
        logging.info("Attempting to initialize Dynamixel Agent in a separate thread...")
        port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U241-if00-port0"
        agent = DynamixelAgent(port=port)
        agent._robot.set_torque_mode(True)
        agent._robot.command_joint_state([0, 0])
        logging.info("Dynamixel Agent connected and centered successfully.")
        return agent
    except Exception as e:
        logging.error(f"Failed to initialize Dynamixel Agent: {e}", exc_info=True)
        return None

def agent_control_loop(agent, stop_event):
    logging.info("Agent control loop started.")
    while not stop_event.is_set():
        try:
            command = [0, 0.2 * np.sin(time.time())]
            agent._robot.command_joint_state(command)
        except Exception as e:
            logging.error(f"Error in agent control loop: {e}")
            time.sleep(1)
        time.sleep(0.05)
    
    logging.info("Agent control loop stopping. Returning servos to center.")
    agent._robot.command_joint_state([0, 0])
    time.sleep(0.5)
    agent._robot.set_torque_mode(False)
    logging.info("Agent torque disabled.")

async def main_async():
    loop = asyncio.get_event_loop()

    # 1. 初始化ZED相机 (这是一个简单的非线程对象)
    logging.info("Initializing ZED camera...")
    capture = ZedCapture()
    logging.info("ZED camera (Pull Mode) started.")

    # 2. 在后台线程中初始化舵机
    agent = await loop.run_in_executor(None, initialize_agent)

    # 3. 如果舵机初始化成功，则启动它的控制循环线程
    agent_thread = None
    stop_event = Event()
    if agent:
        agent_thread = Thread(target=agent_control_loop, args=(agent, stop_event), daemon=True)
        agent_thread.start()
    
    # 4. 初始化WebRTC服务器，直接传入capture对象
    # --- 核心修复：使用正确的关键字参数 ---
    webrtc_server = WebRTCServer(zed_capture=capture) 
    
    logging.info("\n" + "="*50 + "\nSYSTEM IS RUNNING\n" + "="*50)
    
    try:
        await webrtc_server.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Shutdown signal received.")
    finally:
        # 优雅地清理所有资源
        if agent_thread and agent_thread.is_alive():
            logging.info("Stopping agent control loop...")
            stop_event.set()
            agent_thread.join()
        
        logging.info("Closing ZED camera...")
        capture.close()
        logging.info("System shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")

