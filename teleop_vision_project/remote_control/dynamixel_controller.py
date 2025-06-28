# teleop_vision_project/remote_control/dynamixel_controller.py

import os
import time
from dynamixel_sdk import PortHandler, PacketHandler
import logging

class DynamixelController:
    """
    一个自洽的Dynamixel舵机控制器类。
    负责处理与舵机的底层通信。
    """
    # 控制表地址 (根据舵机型号可能不同, X系列通常是这些)
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    LEN_GOAL_POSITION = 4  # 4字节

    def __init__(self, port, dxl_ids, baudrate=1000000):
        self.port = port
        self.dxl_ids = dxl_ids  # e.g., [1, 2] for yaw and pitch
        self.baudrate = baudrate
        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(2.0)
        self.is_connected = False
        logging.info(f"DynamixelController initialized for port {port} with DXL IDs {dxl_ids}.")

    def connect(self):
        """
        打开端口，设置波特率，ping舵机，并为每个舵机启用扭矩。
        """
        if not self.portHandler.openPort():
            logging.error(f"Failed to open port {self.portHandler.port_name}")
            return False
        if not self.portHandler.setBaudRate(self.baudrate):
            logging.error(f"Failed to change the baudrate for {self.portHandler.port_name}")
            return False
        
        time.sleep(0.5)
        
        # Ping该端口上的每一个舵机
        for dxl_id in self.dxl_ids:
            if not self._ping(dxl_id):
                logging.error(f"Could not find DXL ID {dxl_id} on port {self.port}. Aborting.")
                self.portHandler.closePort()
                return False
        
        # 为每一个舵机使能扭矩
        for dxl_id in self.dxl_ids:
            self._enable_torque(dxl_id)

        self.is_connected = True
        logging.info(f"Successfully connected to all servos {self.dxl_ids} on port {self.port}.")
        return True

    def _ping(self, dxl_id):
        """
        Ping指定的舵机ID，检查它是否存在并响应。
        """
        # port_index = dxl_id - 1
        # portHandler = self.portHandlers[port_index]
        
        # 尝试ping舵机
        dxl_model_number, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, dxl_id)
        
        if dxl_comm_result != 0:
            logging.error(f"Ping failed for DXL ID {dxl_id} on port {self.portHandler.port_name}: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return False
        elif dxl_error != 0:
            logging.error(f"Ping error from DXL ID {dxl_id} on port {self.portHandler.port_name}: {self.packetHandler.getRxPacketError(dxl_error)}")
            return False
        
        logging.info(f"[SUCCESS] Ping successful for DXL ID {dxl_id} (Model: {dxl_model_number}) on port {self.portHandler.port_name}")
        return True

    def _enable_torque(self, dxl_id):
        """
        为指定的舵机使能扭矩。
        """
        # port_index = dxl_id - 1
        # portHandler = self.portHandlers[port_index]
        
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 1)
            
        if dxl_comm_result != 0:
            logging.error(f"Torque enable command failed for DXL ID {dxl_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            logging.error(f"Torque enable error response from DXL ID {dxl_id}: {self.packetHandler.getRxPacketError(dxl_error)}")
        else:
            logging.info(f"Successfully enabled torque for DXL ID {dxl_id}")


    def set_goal_positions(self, goal_positions):
        """
        为两个舵机（每个端口一个）设置目标位置。
        """
        if not self.is_connected:
            logging.warning("Controller not connected. Cannot set goal positions.")
            return

        for dxl_id, pos in goal_positions.items():
            self.packetHandler.write4ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_GOAL_POSITION, pos)

    def close(self):
        """
        禁用扭矩并关闭所有端口。
        """
        if self.is_connected:
            for dxl_id in self.dxl_ids:
                self._disable_torque(dxl_id)
            time.sleep(0.1)

        self.portHandler.closePort()
        self.is_connected = False
        logging.info("Dynamixel port closed.")

    def _disable_torque(self, dxl_id):
        """
        为指定的舵机禁用扭矩。
        """
        self.packetHandler.write1ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
