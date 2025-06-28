import pyzed.sl as sl
import cv2
import sys

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and configure seconds
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Create a Mat to store images
    left_image = sl.Mat()
    right_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    print("Grabbing one frame...")

    # Grab an image
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        
        # Retrieve the left image
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        # Retrieve the right image
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)

        # Convert sl.Mat to a numpy array (suitable for OpenCV)
        left_image_np = left_image.get_data()
        right_image_np = right_image.get_data()

        # Save the images using OpenCV
        try:
            cv2.imwrite("im0.png", left_image_np)
            cv2.imwrite("im1.png", right_image_np)
            print("Successfully saved im0.png and im1.png.")
        except Exception as e:
            print(f"Error saving images: {e}")

    else:
        print("Failed to grab a frame.")


    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()