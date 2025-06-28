# webrtc_streaming/streaming_server.py (最终正确版，已恢复调试日志)

import asyncio
import json
import logging
import os
from pathlib import Path
import cv2
import numpy as np

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).parent

class ZedCameraTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, zed_capture, stream_type):
        super().__init__()
        self.zed_capture = zed_capture
        self.stream_type = stream_type
        self.time_base = 90000
        self.frame_count = 0

    async def recv(self):
        loop = asyncio.get_event_loop()
        data_packet = await loop.run_in_executor(None, self.zed_capture.grab_synced_frames)

        if data_packet is None:
            await asyncio.sleep(0.01)
            return await self.recv()

        # --- 已恢复的调试日志 ---
        self.frame_count += 1
        if self.frame_count % 60 == 0 and self.stream_type == 'left':
            frame_data_for_debug = data_packet[f'{self.stream_type}_image']
            if frame_data_for_debug is not None and frame_data_for_debug.size > 0:
                mean_pixel_value = np.mean(frame_data_for_debug)
                log_msg = (
                    f"Debug Check (Frame {self.frame_count}): "
                    f"Shape={frame_data_for_debug.shape}, "
                    f"Mean Pixel Value={mean_pixel_value:.2f}"
                )
                if mean_pixel_value < 5.0:
                    log_msg += " -- WARNING: Image is nearly black!"
                logging.info(log_msg)
            else:
                logging.error(f"Debug Check (Frame {self.frame_count}): Received empty data!")
        # ----------------------------

        timestamp_ns = data_packet["timestamp_ns"]
        frame_data_bgra = data_packet[f'{self.stream_type}_image']
        
        if frame_data_bgra is None or frame_data_bgra.shape[2] != 4:
             return await self.recv()

        frame_data_bgr = cv2.cvtColor(frame_data_bgra, cv2.COLOR_BGRA2BGR)
        frame = VideoFrame.from_ndarray(frame_data_bgr, format="bgr24")
        frame.pts = int(timestamp_ns * self.time_base / 1e9)
        frame.time_base = self.time_base
        
        return frame

class WebRTCServer:
    # --- 核心修复：构造函数的参数名是 'zed_capture' ---
    def __init__(self, zed_capture, host='0.0.0.0', port=8080):
        self.pcs = set()
        self.zed_capture = zed_capture
        self.host = host
        self.port = port

    async def offer(self, request):
        try:
            params = await request.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            pc = RTCPeerConnection()
            self.pcs.add(pc)

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logging.info(f"Connection state is {pc.connectionState}")
                if pc.connectionState in ["failed", "closed", "disconnected"]:
                    await pc.close()
                    self.pcs.discard(pc)
            
            pc.addTransceiver(ZedCameraTrack(self.zed_capture, stream_type='left'), direction='sendonly')
            pc.addTransceiver(ZedCameraTrack(self.zed_capture, stream_type='right'), direction='sendonly')
            
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return web.Response(
                content_type="application/json",
                text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
            )
        except Exception:
            logging.error("Error handling offer:", exc_info=True)
            return web.Response(status=500)

    async def run(self):
        app = web.Application()
        app.router.add_get("/", lambda r: web.Response(content_type="text/html", text=open(os.path.join(ROOT, "index.html"), "r", encoding="utf-8").read()))
        app.router.add_get("/client.js", lambda r: web.Response(content_type="application/javascript", text=open(os.path.join(ROOT, "client.js"), "r", encoding="utf-8").read()))
        app.router.add_post("/offer", self.offer)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        logging.info(f"ZED camera WebRTC server started at http://{self.host}:{self.port}")
        await site.start()
        await asyncio.Event().wait()