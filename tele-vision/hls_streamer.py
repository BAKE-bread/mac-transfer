# hls_streamer.py

import asyncio
import logging
import subprocess
import os
import shutil
from zed_capture import ZEDCapture

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# HLS文件将在此目录下生成，Web服务器会指向这里
HLS_OUTPUT_DIR = "/tmp/hls_stream"

def setup_hls_output_dir():
    """创建或清空HLS输出目录"""
    if os.path.exists(HLS_OUTPUT_DIR):
        shutil.rmtree(HLS_OUTPUT_DIR)
    os.makedirs(os.path.join(HLS_OUTPUT_DIR, "left"))
    os.makedirs(os.path.join(HLS_OUTPUT_DIR, "right"))
    logging.info(f"HLS output directory created at: {HLS_OUTPUT_DIR}")

async def run_ffmpeg_process(zed: ZEDCapture):
    """启动两个FFmpeg进程，分别处理左右眼的视频流"""
    
    # FFmpeg命令模板
    # -i -: 从标准输入读取视频帧
    # -c:v libx264: 使用H.264编码器
    # -hls_time 2: 每个视频片段时长2秒
    # -hls_list_size 3: 播放列表保留3个片段
    # -hls_flags delete_segments: 自动删除旧的片段
    ffmpeg_cmd_template = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgra', # ZED相机输出的像素格式
        '-s', f'{zed.zed.get_camera_information().camera_configuration.resolution.width}x{zed.zed.get_camera_information().camera_configuration.resolution.height}',
        '-r', str(zed.zed.get_camera_information().camera_configuration.fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast', # 超快预设以降低延迟
        '-tune', 'zerolatency', # 针对零延迟进行优化
        '-pix_fmt', 'yuv420p',
        '-hls_time', '1',
        '-hls_list_size', '3',
        '-hls_flags', 'delete_segments',
        '-start_number', '1'
    ]

    # 为左眼创建FFmpeg进程
    left_playlist = os.path.join(HLS_OUTPUT_DIR, "left/stream.m3u8")
    proc_left = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd_template, left_playlist,
        stdin=asyncio.subprocess.PIPE
    )
    logging.info("FFmpeg process for left eye started.")

    # 为右眼创建FFmpeg进程
    right_playlist = os.path.join(HLS_OUTPUT_DIR, "right/stream.m3u8")
    proc_right = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd_template, right_playlist,
        stdin=asyncio.subprocess.PIPE
    )
    logging.info("FFmpeg process for right eye started.")

    try:
        while True:
            # 从ZED相机抓取数据
            packet = zed.grab_frame_packet()
            if packet:
                # 将图像数据写入两个FFmpeg进程的标准输入
                proc_left.stdin.write(packet.left_image.tobytes())
                await proc_left.stdin.drain()
                proc_right.stdin.write(packet.right_image.tobytes())
                await proc_right.stdin.drain()
            await asyncio.sleep(0.01) # 稍微等待一下
    finally:
        proc_left.stdin.close()
        proc_right.stdin.close()
        await proc_left.wait()
        await proc_right.wait()

async def run_web_server():
    """运行一个简单的Web服务器来提供HLS文件服务"""
    from aiohttp import web
    app = web.Application()
    app.router.add_static('/', HLS_OUTPUT_DIR)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    logging.info(f"HLS web server started at http://0.0.0.0:8080")
    await asyncio.Event().wait()


async def main():
    setup_hls_output_dir()
    zed = None
    try:
        zed = ZEDCapture()
        zed.open()
        
        # 并发运行FFmpeg处理和Web服务器
        await asyncio.gather(
            run_ffmpeg_process(zed),
            run_web_server()
        )
    except Exception as e:
        logging.error(f"Main error: {e}", exc_info=True)
    finally:
        if zed:
            zed.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown.")