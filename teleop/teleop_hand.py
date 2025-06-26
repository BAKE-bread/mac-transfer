from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import pyzed.sl as sl  
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

        wrist_to_head = np.array([-0.6, 0, 1.6])

        left_pose = np.concatenate([left_p + wrist_to_head,
                                    rotations.quaternion_from_matrix(left_r)[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_p + wrist_to_head,
                                     rotations.quaternion_from_matrix(right_r)[[1, 2, 3, 0]]])
        # https://docs.vuer.ai/en/latest/tutorials/physics/mocap_hand_control.html
        # tip_indices = [4, 9, 14, 19, 24]
        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        # right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        # https://github.com/unitreerobotics/avp_teleoperate/issues/30
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

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

        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.dof = self.gym.get_asset_dof_count(left_asset)

        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(
            self.env, 
            table_asset, 
            pose, 
            'table', 
            group=0, 
            filter=1, 
            segmentationId=0)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.3, 0, 1.3)
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
            pose.p = gymapi.Vec3(-0.3, 0.2, 1.3)
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
            pose.p = gymapi.Vec3(-0.1, 0.1, 1.3)
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
            pose.p = gymapi.Vec3(-0.2, -0.2, 1.4)
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
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(
            self.env, 
            left_asset, 
            pose, 
            'left',
            group=0, 
            filter=1,
            segmentationId=1)
        set_asset_rigid_shape_props(self.left_handle)
        # https://forums.developer.nvidia.com/t/carb-gymdofproperties-documentation/196552
        left_dof_props = self.gym.get_actor_dof_properties(self.env, self.left_handle)
        left_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        left_dof_props["stiffness"].fill(1000)
        left_dof_props["damping"].fill(50)
        self.gym.set_actor_dof_properties(self.env, self.left_handle, left_dof_props)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # right_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.right_handle = self.gym.create_actor(
            self.env, 
            right_asset, 
            pose, 
            'right', 
            group=0, 
            filter=1,
            segmentationId=1)
        set_asset_rigid_shape_props(self.right_handle)
        right_dof_props = self.gym.get_actor_dof_properties(self.env, self.right_handle)
        right_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        right_dof_props["stiffness"].fill(1000)
        right_dof_props["damping"].fill(50)
        self.gym.set_actor_dof_properties(self.env, self.right_handle, right_dof_props)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

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
        self.cam_pos = np.array([-0.6, 0, 1.6])
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

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):

        if self.print_freq:
            start = time.time()

        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        left_qpos = left_qpos.astype(np.float32)
        right_qpos = right_qpos.astype(np.float32)

        self.gym.set_actor_dof_position_targets(self.env, self.left_handle, left_qpos)
        self.gym.set_actor_dof_position_targets(self.env, self.right_handle, right_qpos)


        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        # self.gym.set_camera_location(self.left_camera_handle,
        #                              self.env,
        #                              gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
        #                              gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        # self.gym.set_camera_location(self.right_camera_handle,
        #                              self.env,
        #                              gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
        #                              gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
        
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

        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        if not self.args.headless:
            self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        #return left_image, right_image
        # 这里不再获取模拟环境的图像
        return None, None
    
    def end(self):
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class DummyActionLoader:
    def __init__(self, period=120):
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

    def get_action(self):
        if self.index >= self.period:
            self.index = 0
            self.phase += 1
        if self.phase >= 12:
            self.phase = 0
            print("Dummy action end, loop again.")
        
        q = self.get_dummy_rotation(self.phase, self.index, self.period)
        head_rmat = rotations.matrix_from_quaternion(q)
        # head_rmat = np.array([
        #     [1.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0],
        #     [0.0, 0.0, 1.0]
        # ])

        wrist_to_head = np.array([-0.6, 0, 1.6])
        left_p = np.array([0.5, 0.3, 0.0])
        right_p = np.array([0.5, -0.3, 0.0])

        # q = self.get_dummy_rotation(self.phase, self.index, self.period)
        left_r = np.array([0, 0, 0, 1])
        right_r = np.array([0, 0, 0, 1])
        # left_r = left_r[[1, 2, 3, 0]]   # [w,x,y,z] -> [x,y,z,w]
        # right_r = right_r[[1, 2, 3, 0]] # [w,x,y,z] -> [x,y,z,w]

        left_pose = np.concatenate([left_p + wrist_to_head, left_r])
        right_pose = np.concatenate([right_p + wrist_to_head, right_r])

        # left_qpos = np.random.randn(12)
        # right_qpos = np.random.randn(12)
        left_qpos = np.zeros(12)
        right_qpos = np.zeros(12)

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
        left_pose = np.array(data_dict["left_pose"])
        right_pose = np.array(data_dict["right_pose"])
        left_qpos = np.array(data_dict["left_qpos"])
        right_qpos = np.array(data_dict["right_qpos"])
        
        self.index += 1
        return head_rmat, left_pose, right_pose, left_qpos, right_qpos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="./data/teleop_hand")
    parser.add_argument('--save_img', action="store_true")
    parser.add_argument('--headless', action="store_true")
    parser.add_argument('--multi_asset', action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(f"{output_dir}/json", exist_ok=True)
    os.makedirs(f"{output_dir}/left", exist_ok=True)
    os.makedirs(f"{output_dir}/right", exist_ok=True)
    teleoperator = VuerTeleop('../assets/config/inspire_hand.yml')
    simulator = Sim(args)

    # 初始化ZED相机
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()
    image_left = sl.Mat()
    image_right = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    try:
        i = 0
        # dataloader = ReplayActionLoader(output_dir)
        dataloader = DummyActionLoader()
        action_list = []
        while True:
            print(f"{i=}")
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            # head_rmat, left_pose, right_pose, left_qpos, right_qpos = dataloader.get_action()
            # left_qpos, right_qpos = remap_hand_qpos(left_qpos, right_qpos)
            
            #left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            # 调用模拟器的step方法，但不使用返回的图像
            simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)
            
            # 获取ZED相机的图像
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                left_img = image_left.numpy()[:, :, :3]
                right_img = image_right.numpy()[:, :, :3]
            else:
                print("Failed to grab ZED camera image.")
                continue

            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
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
