# webrtc_streaming/streaming_server.py
import asyncio
import json
import logging
import os
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame
from zed_capture import ZEDFramePacket

ROOT = os.path.dirname(os.path.abspath(__file__))

async def handle_index(request):
    try:
        with open(os.path.join(ROOT, "index.html"), "r") as f:
            return web.Response(content_type="text/html", text=f.read())
    except FileNotFoundError: return web.Response(status=404)

async def handle_javascript(request):
    try:
        with open(os.path.join(ROOT, "client.js"), "r") as f:
            return web.Response(content_type="application/javascript", text=f.read())
    except FileNotFoundError: return web.Response(status=404)

class VideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, queue, image_key):
        super().__init__()
        self.queue = queue
        self.image_key = image_key
    async def recv(self):
        packet = await self.queue.get()
        image_bgr = cv2.cvtColor(getattr(packet, self.image_key), cv2.COLOR_BGRA2BGR)
        frame = VideoFrame.from_ndarray(image_bgr, format="bgr24")
        frame.pts = int(packet.timestamp_ns / 1e6) # 使用毫秒级时间戳
        frame.time_base = 1000
        return frame

class WebRTCServer:
    def __init__(self):
        self.subscribers = set()
    async def push_frame_packet_to_all(self, packet):
        for queue in self.subscribers:
            if queue.full(): queue.get_nowait()
            queue.put_nowait(packet)
    async def offer(self, request):
        params, pc = await request.json(), RTCPeerConnection()
        queue = asyncio.Queue(1)
        self.subscribers.add(queue)
        @pc.on("connectionstatechange")
        async def on_conn_change():
            if pc.connectionState in ("closed", "failed"):
                self.subscribers.discard(queue)
        pc.addTrack(VideoTrack(queue, "left_image"))
        pc.addTrack(VideoTrack(queue, "right_image")) # 复用同一个队列
        await pc.setRemoteDescription(RTCSessionDescription(**params))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.Response(content_type="application/json", text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))
    async def run_server(self):
        app = web.Application()
        app.router.add_get("/", handle_index)
        app.router.add_get("/client.js", handle_javascript)
        app.router.add_post("/offer", self.offer)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        logging.info(f"Server started at http://0.0.0.0:8080")
        await asyncio.Event().wait()