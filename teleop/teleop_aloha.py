from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
import torch
import os
import json

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations
from PIL import Image

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

from piper_sdk import *

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=100,
)

class DualPiperRobot:
    def __init__(self):
        self.left_piper = C_PiperInterface("can_left")
        self.left_piper.ConnectPort()
        self.left_piper.EnableArm(7)
        self.enable_fun(self.left_piper)

        self.right_piper = C_PiperInterface("can_right")
        self.right_piper.ConnectPort()
        self.right_piper.EnableArm(7)
        self.enable_fun(self.right_piper)

    def enable_fun(self, piper: C_PiperInterface):
        '''
        使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("使能状态:",enable_flag)
            piper.EnableArm(7)
            piper.GripperCtrl(0,1000,0x01, 0)
            print("--------------------")
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)
            pass
        if(elapsed_time_flag):
            print("程序自动使能超时,退出程序")
            exit(0)

    def get_states(self):
        left_arm_qpos = self.left_piper.GetArmJointMsgs()
        left_gripper_qpos = self.left_piper.GetArmJointMsgs()
        left_endpose = self.left_piper.GetArmEndPoseMsgs()
        
        right_arm_qpos = self.right_piper.GetArmJointMsgs()
        right_gripper_qpos = self.right_piper.GetArmJointMsgs()
        right_endpose = self.right_piper.GetArmEndPoseMsgs()

        return left_arm_qpos, right_arm_qpos, left_endpose, right_endpose


    def joint_ctrl(self, piper: C_PiperInterface, pos_action):
        # `pos_action` is 8 dim, the last 2 dim is left/right finger
        # only use left finger for real robot
        factor = 57295.7795     # 1000 * 180 / 3.1415926
        joint_0 = round(pos_action[0]*factor)
        joint_1 = round(pos_action[1]*factor)
        joint_2 = round(pos_action[2]*factor)
        joint_3 = round(pos_action[3]*factor)
        joint_4 = round(pos_action[4]*factor)
        joint_5 = round(pos_action[5]*factor)
        joint_6 = round(pos_action[6]*1000*1000)
        piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        # time.sleep(0.005)

    def dual_joint_ctrl(self, left_pos_action, right_pos_action):
        self.joint_ctrl(self.left_piper, left_pos_action)
        self.joint_ctrl(self.right_piper, right_pos_action)

class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        # self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True)
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=False)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]
        left_p = left_wrist_mat[:3, 3]
        left_r = left_wrist_mat[:3, :3]
        right_p = right_wrist_mat[:3, 3]
        right_r = right_wrist_mat[:3, :3]

        wrist_to_head = np.array([0, 0, 1.6])

        left_pose = np.concatenate([left_p + wrist_to_head,
                                    rotations.quaternion_from_matrix(left_r)[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_p + wrist_to_head,
                                     rotations.quaternion_from_matrix(right_r)[[1, 2, 3, 0]]])
        
        def retarget_single_hand(retargeting, joint_pos):
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            return qpos
        # https://docs.vuer.ai/en/latest/tutorials/physics/mocap_hand_control.html
        # tip_indices = [4, 9, 14, 19, 24]
        # tip_indices = [4, 9]
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0, 1]]
        # right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[0, 1]]
        left_qpos = retarget_single_hand(self.left_retargeting, left_hand_mat)
        right_qpos = retarget_single_hand(self.right_retargeting, right_hand_mat)

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos

class Sim:
    def __init__(self, 
                 args, 
                 print_freq=False):
        self.args = args
        self.print_freq = print_freq

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 4
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 1.0, 1.2, 0.1, table_asset_options)

        # load cube asset
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 1
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        def load_asset(sim, asset_root, asset_path, convex_decomposition=False):
            asset_options = gymapi.AssetOptions()
            asset_options.density = 1
            if convex_decomposition:
                asset_options.vhacd_enabled = True  # enable convex decomposition
                asset_options.vhacd_params.resolution = 300000
                asset_options.vhacd_params.max_convex_hulls = 256
                asset_options.vhacd_params.max_num_vertices_per_ch = 256
            return self.gym.load_asset(sim, asset_root, asset_path, asset_options)
        
        asset_root = "../assets"
        if self.args.multi_asset:
            torus_asset = load_asset(self.sim, asset_root, "torus/torus.urdf", convex_decomposition=True)
            apple_asset = load_asset(self.sim, asset_root, "apple/apple.urdf")
            plate_asset = load_asset(self.sim, asset_root, "plate/plate.urdf")
            bottle1_asset = load_asset(self.sim, asset_root, "bottle1/bottle1.urdf")
            bottle2_asset = load_asset(self.sim, asset_root, "bottle2/bottle2.urdf")
            bottle3_asset = load_asset(self.sim, asset_root, "bottle3/bottle3.urdf")

        left_asset_path = "aloha_piper_ros/piper_description/urdf/piper_description.urdf"
        right_asset_path = "aloha_piper_ros/piper_description/urdf/piper_description.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.vhacd_enabled = True  # enable convex decomposition
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 2
        asset_options.vhacd_params.max_num_vertices_per_ch = 256
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.left_dof = self.gym.get_asset_dof_count(left_asset)
        self.right_dof = self.gym.get_asset_dof_count(right_asset)

        # set up the env grid
        self.num_envs = 1
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.5, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(
            self.env, 
            table_asset, 
            pose, 
            'table', 
            group=0, 
            filter=1, 
            segmentationId=1)
        color = gymapi.Vec3(1.0, 0.0, 0.0)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.2, 0, 1.3)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        cube_handle = self.gym.create_actor(
            self.env, 
            cube_asset, 
            pose, 
            'cube', 
            group=0,
            filter=0,
            segmentationId=1)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        def set_asset_rigid_shape_props(actor_handle):
            props = self.gym.get_actor_rigid_shape_properties(self.env, actor_handle)
            props[0].friction = 100.0
            props[0].rolling_friction = 100.0
            props[0].torsion_friction = 100.0
            self.gym.set_actor_rigid_shape_properties(self.env, actor_handle, props)
        set_asset_rigid_shape_props(cube_handle)
        # # torus
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(-0.2, 0, 1.25)
        # pose.r = gymapi.Quat(0, 0, 0, 1)
        # torus_handle = self.gym.create_actor(
        #     self.env, 
        #     torus_asset, 
        #     pose, 
        #     'torus', 
        #     group=0,
        #     filter=0,
        #     segmentationId=1)

        if self.args.multi_asset:
            # apple
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.2, 0.2, 1.3)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            apple_handle = self.gym.create_actor(
                self.env, 
                apple_asset, 
                pose, 
                'apple', 
                group=0,
                filter=0,
                segmentationId=1)
            set_asset_rigid_shape_props(apple_handle)

            # plate
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.4, 0.1, 1.3)
            pose.r = gymapi.Quat(0.7071, 0, 0, 0.7071)
            plate_handle = self.gym.create_actor(
                self.env, 
                plate_asset, 
                pose, 
                'plate', 
                group=0,
                filter=0,
                segmentationId=1)
            set_asset_rigid_shape_props(plate_handle)

            # bottle
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.3, -0.2, 1.4)
            # pose.r = gymapi.Quat(0.7071, 0, 0, 0.7071)
            pose.r = gymapi.Quat(0, 0, -0.7071, 0.7071)
            bottle_handle = self.gym.create_actor(
                self.env, 
                bottle1_asset, 
                pose, 
                'bottle', 
                group=0,
                filter=0,
                segmentationId=1)
            set_asset_rigid_shape_props(bottle_handle)

        # left_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.3, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(
            self.env, 
            left_asset, 
            pose, 
            'left_piper',
            group=0, 
            filter=1,
            segmentationId=1)
        set_asset_rigid_shape_props(self.left_handle)
        # https://forums.developer.nvidia.com/t/carb-gymdofproperties-documentation/196552
        left_dof_props = self.gym.get_actor_dof_properties(self.env, self.left_handle)
        left_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        left_dof_props["stiffness"].fill(400.0)
        left_dof_props["damping"].fill(40.0)
        self.gym.set_actor_dof_properties(self.env, self.left_handle, left_dof_props)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.left_dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)
        self.left_eef_idx = self.gym.find_actor_rigid_body_index(self.env, self.left_handle, "gripper_base", gymapi.DOMAIN_SIM)
        # right_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, -0.3, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.right_handle = self.gym.create_actor(
            self.env, 
            right_asset, 
            pose, 
            'right_piper', 
            group=0, 
            filter=1,
            segmentationId=1)
        set_asset_rigid_shape_props(self.right_handle)
        right_dof_props = self.gym.get_actor_dof_properties(self.env, self.right_handle)
        right_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        right_dof_props["stiffness"].fill(400.0)
        right_dof_props["damping"].fill(40.0)
        self.gym.set_actor_dof_properties(self.env, self.right_handle, right_dof_props)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.right_dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)
        self.right_eef_idx = self.gym.find_actor_rigid_body_index(self.env, self.right_handle, "gripper_base", gymapi.DOMAIN_SIM)


        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]

        if not self.args.headless:
            # create default viewer
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        if not self.args.headless:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([0, 0, 1.6])
        self.cam_width = 1280
        self.cam_height = 720
        # self.fovx = 1.1355  # phone camera fov, 65.06 degree
        self.fovx = 1.57    # 90 degree
        self.fovy = 2 * np.arctan(self.cam_height / self.cam_width * np.tan(self.fovx/2))
        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cam_width
        camera_props.height = self.cam_height
        camera_props.horizontal_fov = self.fovx / np.pi * 180
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cam_width
        camera_props.height = self.cam_height
        camera_props.horizontal_fov = self.fovx / np.pi * 180
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))
        
        left_link_dict = self.gym.get_asset_rigid_body_dict(left_asset)
        left_hand_index = left_link_dict["gripper_base"]
        left_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "left_piper")
        self.left_jacobian = gymtorch.wrap_tensor(left_jacobian)
        self.left_j_eef = self.left_jacobian[:, left_hand_index - 1, :, :6]

        right_link_dict = self.gym.get_asset_rigid_body_dict(right_asset)
        right_hand_index = right_link_dict["gripper_base"]
        right_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "right_piper")
        self.right_jacobian = gymtorch.wrap_tensor(right_jacobian)
        self.right_j_eef = self.right_jacobian[:, right_hand_index - 1, :, :6]

        self.damping = 0.05
        # print(f"{self.left_jacobian.shape=}")
        # print(f"{self.left_j_eef.shape=}")


    def control_ik(self, dpose, j_eef, damping, num_envs):
        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
        return u

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos, robot:DualPiperRobot=None):

        # 1. step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # 2. set robot
        left_pos_action = np.zeros(self.left_dof, dtype=np.float32)
        left_effort_action = np.zeros(self.left_dof, dtype=np.float32)
        right_pos_action = np.zeros(self.right_dof, dtype=np.float32)
        right_effort_action = np.zeros(self.right_dof, dtype=np.float32)
        
        def get_robot_states():
            if False:
            # if self.args.real_robot:
                left_arm_qpos, right_arm_qpos, left_endpose, right_endpose = robot.get_states()
                left_dof_pos = left_arm_qpos
                right_dof_pos = right_arm_qpos
                left_eef_pos = left_endpose[:3]
                right_eef_pos = right_endpose[:3]
                left_eef_rot = left_endpose[3:]
                right_eef_rot = right_endpose[3:]
            else:
                left_dof_states = self.gym.get_actor_dof_states(self.env, self.left_handle, gymapi.STATE_ALL)
                left_dof_pos = left_dof_states['pos']


                right_dof_states = self.gym.get_actor_dof_states(self.env, self.right_handle, gymapi.STATE_ALL)
                right_dof_pos = right_dof_states['pos']

                rb_states = self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL)
                left_eef_states = rb_states[self.left_eef_idx]
                left_eef_pos = np.array([left_eef_states['pose']['p']['x'], 
                                        left_eef_states['pose']['p']['y'],
                                        left_eef_states['pose']['p']['z']])
                left_eef_rot = np.array([left_eef_states['pose']['r']['x'],
                                        left_eef_states['pose']['r']['y'],
                                        left_eef_states['pose']['r']['z'],
                                        left_eef_states['pose']['r']['w']])


                right_eef_states = rb_states[self.right_eef_idx]
                right_eef_pos = np.array([right_eef_states['pose']['p']['x'], 
                                        right_eef_states['pose']['p']['y'],
                                        right_eef_states['pose']['p']['z']])
                right_eef_rot = np.array([right_eef_states['pose']['r']['x'],
                                        right_eef_states['pose']['r']['y'],
                                        right_eef_states['pose']['r']['z'],
                                        right_eef_states['pose']['r']['w']])
            
            return left_dof_pos, right_dof_pos, left_eef_pos, right_eef_pos, left_eef_rot, right_eef_rot
        
        left_dof_pos, right_dof_pos, left_eef_pos, right_eef_pos, left_eef_rot, right_eef_rot = get_robot_states()

        left_goal_pos = left_pose[:3]
        left_goal_rot = left_pose[3:]
        right_goal_pos = right_pose[:3]
        right_goal_rot = right_pose[3:]

        print(f"{left_eef_pos=}")
        print(f"{left_goal_pos=}")
        print(f"{left_eef_rot=}")
        print(f"{left_goal_rot=}")

        def orientation_error(desired, current):
            # [x,y,z,w] -> [w,x,y,z]
            desired = desired[[3, 0, 1, 2]]
            current = current[[3, 0, 1, 2]]
            cc = rotations.q_conj(current)
            q_r = rotations.concatenate_quaternions(desired, cc)
            sign_w = np.sign(q_r[..., 0])
            result = q_r[..., 1:4] * sign_w[..., np.newaxis]
            return result

        # compute position and orientation error
        left_pos_err = left_goal_pos - left_eef_pos
        left_rot_err = orientation_error(left_goal_rot, left_eef_rot)
        left_pos_err = torch.from_numpy(left_pos_err).float()
        left_rot_err = torch.from_numpy(left_rot_err).float()

        right_pos_err = right_goal_pos - right_eef_pos
        right_rot_err = orientation_error(right_goal_rot, right_eef_rot)
        right_pos_err = torch.from_numpy(right_pos_err).float()
        right_rot_err = torch.from_numpy(right_rot_err).float()

        left_dpose = torch.cat([left_pos_err, left_rot_err], -1).unsqueeze(-1)
        left_u = self.control_ik(left_dpose, self.left_j_eef, self.damping, self.num_envs)
        left_u = left_u.numpy()
        left_pos_action[:6] = left_dof_pos[:6] + left_u

        right_dpose = torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1)
        right_u = self.control_ik(right_dpose, self.right_j_eef, self.damping, self.num_envs)
        right_u = right_u.numpy()
        right_pos_action[:6] = right_dof_pos[:6] + right_u

        # # gripper
        # left_pos_action[7:] = left_qpos[:]

        # left_pos_action = np.zeros(self.left_dof, dtype=np.float32)
        # right_pos_action = np.zeros(self.right_dof, dtype=np.float32)

        # left_pos_action = np.array([0.90850016,  0.83095923,  1.59208039, -0.30394616, -0.6506236 ,
        # -1.22253479,  0.69958254], dtype=np.float32)
        # right_pos_action = np.array([-0.22222348, -0.21170436, -0.70714719,
        # 0.30132911, -1.79343334,  0.39191789,  1.25847742], dtype=np.float32)

        # # step 0
        # left_pos_action = np.array([-0.006135923, -0.983281732, 1.1535536050796509, 0.003067962, -0.286854416, 0, -0.01870966], dtype=np.float32)
        # right_pos_action = np.array([0.003067962, -0.955670059, 1.1688933372497559, 0.001533981, -0.300660253, 0.001533981, 0.037235539], dtype=np.float32)

        # # step 100
        # left_pos_action = np.array([-0.006135923, -0.944932163, 1.1780972480773926, 0.029145635664463043, -0.357417524, -0.052155349, 0.022798033], dtype=np.float32)
        # right_pos_action = np.array([-0.050621368, -0.977145791, 1.1765632629394531, -0.003067962, -0.348213643, 0.050621368, 0.044454295], dtype=np.float32)

        # # step 500
        # left_pos_action = np.array([-0.006135923, -0.855961323, 1.1873011589050293, 0.03374758, -0.385029197, -0.142660215, 0.022798033], dtype=np.float32)
        # right_pos_action = np.array([-0.021475732, -0.994019568, 1.1857671737670898, -0.046019424, -0.349747628, 0.084368944, 0.044454295], dtype=np.float32)

        # step 1500
        # left_pos_action = np.array([-0.003067962, -1.175029278, 1.290077925, 0.027611654, -0.294524312, -0.087436907, 0.019188657], dtype=np.float32)
        # right_pos_action = np.array([0.383495212, 0.579844773, 0.299126267, -0.041417483, -0.671883583, 0.116582543, 0.629171669], dtype=np.float32)

        # fold cloth
        left_pos_action = np.array([-0.17592246,  1.26782502, -0.6701163 , -0.1494047 ,  0.15217471,-0.63840667], dtype=np.float32)
        right_pos_action = np.array([0.01376813,  0.31507315,  0.79390346, -0.92123384,-0.39565071, -0.25571007,  0.13712819,  0.00515977], dtype=np.float32)

        self.gym.set_actor_dof_position_targets(self.env, self.left_handle, left_pos_action)
        self.gym.set_actor_dof_position_targets(self.env, self.right_handle, right_pos_action)

        # control real robot
        # print(f"{left_pos_action=}")
        # print(f"{right_pos_action=}")
        if self.args.real_robot:
            robot.dual_joint_ctrl(left_pos_action, right_pos_action)

        # 3. set camera
        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        p_left = self.cam_pos + curr_left_offset
        p_right = self.cam_pos + curr_right_offset
        p_left = gymapi.Vec3(p_left[0], p_left[1], p_left[2])
        p_right = gymapi.Vec3(p_right[0], p_right[1], p_right[2])

        q_left = rotations.quaternion_from_matrix(head_rmat)[[1, 2, 3, 0]]
        q_right = rotations.quaternion_from_matrix(head_rmat)[[1, 2, 3, 0]]
        q_left = gymapi.Quat(q_left[0], q_left[1], q_left[2], q_left[3])
        q_right = gymapi.Quat(q_right[0], q_right[1], q_right[2], q_right[3])

        left_cam_transform = gymapi.Transform(p_left, q_left)
        right_cam_transform = gymapi.Transform(p_right, q_right)

        self.gym.set_camera_transform(self.left_camera_handle, self.env, left_cam_transform)
        self.gym.set_camera_transform(self.right_camera_handle, self.env, right_cam_transform)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        if not self.args.headless:
            self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        return left_image, right_image

    def end(self):
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

def adjust_aloha_gripper_orientation(left_pose, right_pose):
    left_p = left_pose[:3]
    right_p = right_pose[:3]
    left_r = left_pose[3:][[3, 0, 1, 2]]    # [x,y,z,w] -> [w,x,y,z]
    right_r = right_pose[3:][[3, 0, 1, 2]]  # [x,y,z,w] -> [w,x,y,z]

    # rotate both gripper to hand default orientation (fingers point to right, palm towards eyes)
    q1 = rotations.quaternion_from_axis_angle([0, 0, 1, -np.pi/2])
    # for right gripper, rotate to thumb down
    q2 = rotations.quaternion_from_axis_angle([0, 1, 0, np.pi])
    q3 = rotations.concatenate_quaternions(q2, q1)

    left_r = rotations.concatenate_quaternions(left_r, q1)
    right_r = rotations.concatenate_quaternions(right_r, q3)

    left_r = left_r[[1, 2, 3, 0]]           # [w,x,y,z] -> [x,y,z,w]
    right_r = right_r[[1, 2, 3, 0]]         # [w,x,y,z] -> [x,y,z,w]

    left_pose = np.concatenate([left_p, left_r])
    right_pose = np.concatenate([right_p, right_r])
    return left_pose, right_pose

def remap_aloha_gripper_qpos(left_qpos, right_qpos):
    # urdf range (0, 0.04) -> (0.828, 1.0)
    skew = (1.0 - 0.828) / (0.04 - 0.0)
    offset = 0.828
    return left_qpos * skew + offset, right_qpos * skew + offset

class DummyActionLoader:
    def __init__(self, period=300):
        self.index = 0
        self.period = period
        self.phase = 0
    
    def get_dummy_rotation(self, phase, index, period):
        if phase == 0: # Z axis 0 -> 90
            q1 = np.array([1, 0, 0, 0])
            q2 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
            q = rotations.quaternion_slerp(q1, q2, index/period)

        elif phase == 1: # Z axis 90 -> 180
            q2 = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
            q3 = np.array([np.cos(np.pi/2), 0, 0, np.sin(np.pi/2)])
            q = rotations.quaternion_slerp(q2, q3, index/period)

        elif phase == 2: # Z axis 180 -> 270
            q3 = np.array([np.cos(np.pi/2), 0, 0, np.sin(np.pi/2)])
            q4 = np.array([np.cos(np.pi*3/4), 0, 0, np.sin(np.pi*3/4)])
            q = rotations.quaternion_slerp(q3, q4, index/period)

        elif phase == 3: # Z axis 270 -> 360
            q4 = np.array([np.cos(np.pi*3/4), 0, 0, np.sin(np.pi*3/4)])
            q5 = np.array([-1, 0, 0, 0])
            q = rotations.quaternion_slerp(q4, q5, index/period)

        elif phase == 4: # Y axis 0 -> 90
            q1 = np.array([1, 0, 0, 0])
            q2 = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
            q = rotations.quaternion_slerp(q1, q2, index/period)

        elif phase == 5: # Y axis 90 -> 180
            q2 = np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
            q3 = np.array([np.cos(np.pi/2), 0, np.sin(np.pi/2), 0])
            q = rotations.quaternion_slerp(q2, q3, index/period)

        elif phase == 6: # Y axis 180 -> 270
            q3 = np.array([np.cos(np.pi/2), 0, np.sin(np.pi/2), 0])
            q4 = np.array([np.cos(np.pi*3/4), 0, np.sin(np.pi*3/4), 0])
            q = rotations.quaternion_slerp(q3, q4, index/period)

        elif phase == 7: # Y axis 270 -> 360
            q4 = np.array([np.cos(np.pi*3/4), 0, np.sin(np.pi*3/4), 0])
            q5 = np.array([-1, 0, 0, 0])
            q = rotations.quaternion_slerp(q4, q5, index/period)
        
        elif phase == 8: # X axis 0 -> 90
            q1 = np.array([1, 0, 0, 0])
            q2 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
            q = rotations.quaternion_slerp(q1, q2, index/period)

        elif phase == 9: # X axis 90 -> 180
            q2 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
            q3 = np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0, 0])
            q = rotations.quaternion_slerp(q2, q3, index/period)

        elif phase == 10: # X axis 180 -> 270
            q3 = np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0, 0])
            q4 = np.array([np.cos(np.pi*3/4), np.sin(np.pi*3/4), 0, 0])
            q = rotations.quaternion_slerp(q3, q4, index/period)

        elif phase == 11: # X axis 270 -> 0
            q4 = np.array([np.cos(np.pi*3/4), np.sin(np.pi*3/4), 0, 0])
            q5 = np.array([-1, 0, 0, 0])
            q = rotations.quaternion_slerp(q4, q5, index/period)
        return q

    def get_dummy_translation(self, phase, index, period):
        left_p = np.array([0.2, 0.3, 1.6])
        right_p = np.array([0.2, -0.3, 1.6])
        distance = 0.1
        delta = (self.index / self.period) * distance
        if self.phase % 4 == 0:     # 0.0 -> distance
            left_p[0] += delta
            right_p[1] += delta
        elif self.phase % 4 == 1:   # distance -> 0.0    
            left_p[0] += distance - delta
            right_p[1] += distance - delta
        elif self.phase % 4 == 2:   # 0.0 -> -distance   
            left_p[0] += -delta
            right_p[1] += -delta
        else:                       # -distance -> 0.0   
            left_p[0] += -distance + delta
            right_p[1] += -distance + delta
        return left_p, right_p

    def get_action(self):
        if self.index >= self.period:
            self.index = 0
            self.phase += 1
        if self.phase >= 12:
            self.phase = 0
            print("Dummy action end, loop again.")
        
        # q = self.get_dummy_rotation(self.phase, self.index, self.period)
        # head_rmat = rotations.matrix_from_quaternion(q)
        head_rmat = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # left_p = np.array([(self.index/self.period)*0.3, 0.3, 1.5])
        # # right_p = np.array([0.3, -0.3, 1.3 + (self.index/self.period)*0.5])
        # right_p = np.array([0.3, -0.3 + (self.index/self.period)*0.3, 1.5])
        left_p, right_p = self.get_dummy_translation(self.phase, self.index, self.period)

        # q = self.get_dummy_rotation(self.phase, self.index, self.period)
        # left_r = q
        # right_r = q
        # left_r = left_r[[1, 2, 3, 0]]   # [w,x,y,z] -> [x,y,z,w]
        # right_r = right_r[[1, 2, 3, 0]] # [w,x,y,z] -> [x,y,z,w]
        
        # y-axis 88 degree
        left_r = np.array([0, 0.6946584, 0, 0.7193398])
        right_r = np.array([0, 0.6946584, 0, 0.7193398])

        # left_r = np.array([0, 6.7561352e-01,  4.9280497e-06, 7.3725611e-01])
        # right_r = np.array([0, 6.7561352e-01,  4.9280497e-06, 7.3725611e-01])

        left_pose = np.concatenate([left_p, left_r])
        right_pose = np.concatenate([right_p, right_r])

        # left_qpos = np.random.randn(2)
        # right_qpos = np.random.randn(2)
        # left_qpos = np.zeros(2)
        # right_qpos = np.zeros(2)
        # left_qpos = np.array([1, 1])
        # right_qpos = np.array([1, 1])

        left_qpos = np.full(8, self.index/self.period)
        right_qpos = np.full(8, self.index/self.period)

        # left_qpos = np.zeros(8)
        # right_qpos = np.zeros(8)

        self.index += 1
        return head_rmat, left_pose, right_pose, left_qpos, right_qpos

class ReplayActionLoader:
    def __init__(self, data_dir, index=0):
        self.index = index
        self.data_dir = data_dir
        self.json_path = os.path.join(data_dir, "action_1.json")

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"action.json not found: {self.json_path}")
        with open(self.json_path, 'r') as json_file:
            self.data_list = json.load(json_file)
        
        self.data_len = len(self.data_list)
        if self.data_len == 0:
            raise ValueError(f"action.json is empty: {self.json_path}")
        
    def get_action(self):
        if self.index >= self.data_len:
            self.index = 0
            print("Replay action end, loop again.")

        data_dict = self.data_list[self.index]
        
        head_rmat = np.array(data_dict["head_rmat"])
        left_p = np.array(data_dict["left_pose"][:3])
        right_p = np.array(data_dict["left_pose"][:3])
        left_p[0] += 0.8
        right_p[0] += 0.8

        q_offset = np.array([0, 0.6946584, 0, 0.7193398])   # [x,y,z,w]
        left_r = np.array(data_dict["left_pose"][3:])
        right_r = np.array(data_dict["right_pose"][3:])
        left_r = rotations.concatenate_quaternions(left_r, q_offset)
        right_r = rotations.concatenate_quaternions(right_r, q_offset)

        left_pose = np.concatenate([left_p, left_r])
        right_pose = np.concatenate([right_p, right_r])
        # left_pose = np.array(data_dict["left_pose"])
        # right_pose = np.array(data_dict["right_pose"])
        # left_qpos = np.array(data_dict["left_qpos"])
        # right_qpos = np.array(data_dict["right_qpos"])
        # left_qpos = np.random.randn(2)
        # right_qpos = np.random.randn(2)
        left_qpos = np.array([0.01, 0.01])
        right_qpos = np.array([0.03, 0.03])

        self.index += 1
        return head_rmat, left_pose, right_pose, left_qpos, right_qpos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="./data/teleop_aloha_gripper")
    parser.add_argument('--save_img', action="store_true")
    parser.add_argument('--headless', action="store_true")
    parser.add_argument('--multi_asset', action="store_true")
    parser.add_argument('--real_robot', action="store_true")

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(f"{output_dir}/json", exist_ok=True)
    os.makedirs(f"{output_dir}/left", exist_ok=True)
    os.makedirs(f"{output_dir}/right", exist_ok=True)
    # teleoperator = VuerTeleop('../assets/config/aloha_gripper.yml')
    simulator = Sim(args)
    robot = None
    if args.real_robot:
        robot = DualPiperRobot()

    try:
        i = 0
        # dataloader = ReplayActionLoader(output_dir)
        dataloader = DummyActionLoader()
        action_list = []
        while True:
            print(f"{i=}")
            # head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = dataloader.get_action()
            # left_pose = np.array([-0.1, 0.2, 1.6, 0.0, 1.0, 0.0, 0.0])
            # right_pose = np.array([-0.1, -0.2, 1.6, 0.0, -1.0, 0.0, 0.0])
            # left_pose, right_pose = adjust_aloha_gripper_orientation(left_pose, right_pose)
            # left_qpos, right_qpos = remap_aloha_gripper_qpos(left_qpos, right_qpos)

            # TODO: ZED Mini 相机通信
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos, robot)
            # np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
            action_dict = {
                "head_rmat": head_rmat.tolist(),
                "left_pose": left_pose.tolist(),
                "right_pose": right_pose.tolist(),
                "left_qpos": left_qpos.tolist(),
                "right_qpos": right_qpos.tolist(),
            }
            action_list.append(action_dict)
            if args.save_img:
                left_img = Image.fromarray(left_img.astype(np.uint8))
                right_img = Image.fromarray(right_img.astype(np.uint8))
                left_img.save(f"{output_dir}/left/{i}.png")
                right_img.save(f"{output_dir}/right/{i}.png")
            # image saving time > 3dgs render time
            # TODO: collect camera pose, offline rendering
            # t2 = time.time(); print(f"t2={t2-t1:.4f}")
            i += 1
    except KeyboardInterrupt:
        output_file = f"{output_dir}/action.json"
        with open(output_file, "w") as json_file:
            json.dump(action_list, json_file, indent=4)
        simulator.end()
        exit(0)
