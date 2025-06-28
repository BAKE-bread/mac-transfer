import pyzed.sl as sl
import cv2
import threading
import queue
import time

# 创建一个线程安全的队列来存放数据包
data_queue = queue.Queue(maxsize=30) # 设置一个最大尺寸以防止内存无限增长

def producer_thread(zed_cam, runtime_params):
    """
    生产者线程：只负责从ZED相机抓取数据，打包并放入队列。
    """
    print("Producer thread started...")
    # 为图像数据分配内存 (只需一次)
    left_image = sl.Mat()
    right_image = sl.Mat()

    while True:
        # 抓取一帧
        if zed_cam.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 1. 获取唯一的、权威的时间戳
            timestamp = zed_cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)

            # 2. 获取双目图像
            zed_cam.retrieve_image(left_image, sl.VIEW.LEFT)
            zed_cam.retrieve_image(right_image, sl.VIEW.RIGHT)

            # 3. 创建原子数据包 (注意.copy()，避免内存冲突)
            #    将图像数据从ZED的GPU/CPU内存复制出来，成为独立的numpy数组
            data_packet = {
                "timestamp_ns": timestamp.get_nanoseconds(),
                "left_image": left_image.get_data().copy(),
                "right_image": right_image.get_data().copy()
            }

            # 4. 将数据包放入队列
            try:
                data_queue.put(data_packet, block=False) # 非阻塞放入，如果队列满了就丢弃
            except queue.Full:
                # print("Queue is full, dropping a frame.")
                pass
        
        # 这里可以加一个小小的延时，避免CPU空转，但通常grab()会阻塞足够时间
        # time.sleep(0.001)

def local_consumer_thread():
    """
    消费者线程1：负责本机的、需要严格同步的双目操作。
    """
    print("Local consumer thread started...")
    while True:
        try:
            # 从队列获取数据包
            data_packet = data_queue.get(timeout=1)

            timestamp = data_packet["timestamp_ns"]
            left_img = data_packet["left_image"]
            right_img = data_packet["right_image"]

            # 在这里执行需要严格同步的本机操作
            # 例如：立体匹配、目标检测等
            # 因为left_img和right_img来自同一个包，它们的时间戳是完全同步的
            print(f"[Local OP] Processing frame with timestamp: {timestamp}")
            
            # 示例：显示双目图像
            combined_img = cv2.hconcat([left_img, right_img])
            cv2.imshow("Local Binocular Processing", combined_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except queue.Empty:
            # 如果队列为空，可以稍等或继续
            continue

def network_consumer_thread():
    """
    消费者线程2：负责将左眼视频流和时间戳发送到网络。
    """
    print("Network consumer thread started...")
    while True:
        try:
            # 从队列获取数据包
            data_packet = data_queue.get(timeout=1)
            
            timestamp = data_packet["timestamp_ns"]
            left_img = data_packet["left_image"]

            print(f"[Network OP] Preparing to send left eye frame with timestamp: {timestamp}")

            # ==========================================================
            # 在这里，实现你的网络传输逻辑
            # 1. 视频编码与发送 (HEVC)
            #    - 将 left_img (numpy数组) 喂给你的HEVC编码器。
            #    - 最常见的方式是通过管道(pipe)传给FFmpeg子进程。
            #    - FFmpeg会将编码后的HEVC流通过RTP/SRT等协议发送出去。
            
            # 2. 元数据时间戳发送
            #    - 通过一个独立的网络通道 (如WebSocket, gRPC, ZeroMQ)
            #    - 发送一个JSON消息，例如:
            #      meta_message = {"capture_timestamp": timestamp, "source": "ZED_Mini_Left"}
            #      websocket_connection.send(json.dumps(meta_message))
            # ==========================================================

        except queue.Empty:
            continue


if __name__ == "__main__":
    zed = sl.Camera()

    # 初始化ZED相机
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720 # 根据需求选择
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE # 如果不需要深度，可以关闭以节省资源

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(-1)
    
    runtime_params = sl.RuntimeParameters()

    # 创建并启动线程
    p_thread = threading.Thread(target=producer_thread, args=(zed, runtime_params))
    lc_thread = threading.Thread(target=local_consumer_thread)
    nc_thread = threading.Thread(target=network_consumer_thread)

    p_thread.start()
    lc_thread.start()
    nc_thread.start()

    # 等待线程结束 (在这个例子中是无限循环，但可以加入退出逻辑)
    p_thread.join()
    lc_thread.join()
    nc_thread.join()

    zed.close()
    cv2.destroyAllWindows()