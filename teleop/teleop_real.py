import math
import numpy as np

np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations

import time
import cv2
from constants_vuer import *
from TeleVision import OpenTeleVision
import pyzed.sl as sl
from dynamixel.active_cam import DynamixelAgent
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

resolution = (360, 640)
# resolution = (720, 1280)

crop_size_w = 1
crop_size_h = 0
resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)

# agent = DynamixelAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8IT033-if00-port0")
agent = DynamixelAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U241-if00-port0")
print(f"{agent=}")
agent._robot.set_torque_mode(True)

# Create a Camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 opr HD1200 video mode, depending on camera type.
init_params.camera_fps = 60  # Set fps at 60

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera Open : " + repr(err) + ". Exit program.")
    exit()

# Capture 50 frames and stop
i = 0
image_left = sl.Mat()
image_right = sl.Mat()
runtime_parameters = sl.RuntimeParameters()

img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
img_height, img_width = resolution_cropped[:2]
shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)
image_queue = Queue()
toggle_streaming = Event()
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming)

while True:
    start = time.time()

    t0 = time.time()

    # 1. Get head pose from Vision Pro
    head_mat = grd_yup2grd_zup[:3, :3] @ tv.head_matrix[:3, :3] @ grd_yup2grd_zup[:3, :3].T
    if np.sum(head_mat) == 0:
        head_mat = np.eye(3)
    head_rot = rotations.quaternion_from_matrix(head_mat[0:3, 0:3])
    ypr = rotations.euler_from_quaternion(head_rot, 2, 1, 0, False)
    print(f"{ypr=}")

    # 2. Control dynamixel
    action = agent.act(1)
    # NOTE: yaw and pitch order is different from oringinal OpenTelevision
    # because we are using differnt dynamixel id
    command = np.array([ypr[1], ypr[0]])
    agent._robot.command_joint_state(command)
    time.sleep(0.01) 
    true_value = agent._robot._driver.get_joints()    
    print("true value                 : ", [f"{x:.3f}" for x in true_value])

    # 3. Get image from ZED camera
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured

        # left_img = image_left.get_data()
        # right_img = image_right.get_data()
        # img_show = np.concatenate((left_img, right_img), axis=1)
        # h, w = img_show.shape[:2]
        # img_show = cv2.resize(img_show, (int(w/1.5), int(h/1.5)))
        # cv2.imshow("Stereo Image", img_show)
        # cv2.waitKey(1)

    # 4. Send image to Vision Pro
    
    bgr = np.hstack((image_left.numpy()[crop_size_h:, crop_size_w:-crop_size_w],
                    image_right.numpy()[crop_size_h:, crop_size_w:-crop_size_w]))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    h, w = img_array.shape[:2]
    rgb = cv2.resize(rgb, (w, h))
    print(f"{img_array.shape=}")
    np.copyto(img_array, rgb)
    
    t1 = time.time()
    fps = 1/(t1-t0)
    print(f"{fps=:.4f}")

zed.close()