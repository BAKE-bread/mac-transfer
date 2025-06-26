import pyzed.sl as sl
import cv2

def main():
    # 创建ZED相机对象
    zed = sl.Camera()

    # 设置相机初始化参数
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 设置分辨率为HD720
    init_params.camera_fps = 60  # 设置帧率为60fps

    # 打开相机
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"相机打开失败: {repr(err)}，退出程序。")
        return

    # 创建用于存储左右图像的Mat对象
    image_left = sl.Mat()
    image_right = sl.Mat()

    # 运行时参数
    runtime_parameters = sl.RuntimeParameters()

    try:
        while True:
            # 抓取一帧图像
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # 从相机中获取左右图像
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_image(image_right, sl.VIEW.RIGHT)

                # 将图像转换为OpenCV格式
                left_img = image_left.get_data()
                right_img = image_right.get_data()

                # 显示左右图像
                cv2.imshow("Left Image", left_img)
                cv2.imshow("Right Image", right_img)

                # 按 'q' 键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("无法抓取图像，请检查相机连接。")
    except KeyboardInterrupt:
        pass

    # 关闭窗口
    cv2.destroyAllWindows()

    # 关闭相机
    zed.close()

if __name__ == "__main__":
    main()