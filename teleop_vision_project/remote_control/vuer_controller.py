# teleop_vision_project/remote_control/vuer_controller.py

import asyncio
import json
import logging
import os
import threading
import time
from typing import Callable

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import Scene

# We borrow the VideoStreamTrack class from our original streaming server module
from teleop_vision_project.webrtc_streaming.streaming_server import VideoStreamTrack

class UnifiedTeleopServer:
    """
    A unified server that handles both Vuer for teleoperation
    and WebRTC for video streaming within a single application instance.
    This avoids threading and asyncio event loop conflicts.
    """
    def __init__(self,
                 capture_queue,
                 project_root,
                 host="0.0.0.0",
                 port=8012, # Use a single port for everything
                 camera_move_cb: Callable[[ClientEvent, Scene], None] = None):

        self.app = Vuer(host=host, port=port, cert=None, key=None)
        self.host = host
        self.port = port
        self.project_root = project_root
        self.pcs = set() # Set to store peer connections

        # 1. Set up Vuer handler for teleoperation
        if camera_move_cb:
            self.app.add_handler("CAMERA_MOVE", camera_move_cb)

        # 2. Set up WebRTC video tracks using the shared capture queue
        self.left_track = VideoStreamTrack("left", capture_queue, "left_image")
        self.right_track = VideoStreamTrack("right", capture_queue, "right_image")

        # 3. Add our WebRTC and HTML serving routes to Vuer's underlying aiohttp app
        aio_app = self.app.app
        aio_app.router.add_get("/", self.handle_index)
        aio_app.router.add_get("/client.js", self.handle_javascript)
        aio_app.router.add_post("/offer", self.handle_webrtc_offer)
        
        # 4. Run the entire Vuer application in a background thread
        server_thread = threading.Thread(target=self.run_vuer_server, daemon=True)
        server_thread.start()
        
        # Allow the server a moment to initialize
        time.sleep(1.0)
        
        # FIX: Manually construct the URL instead of relying on self.app.url
        logging.info(f"Unified Server Started. Visit URL: http://{self.host}:{self.port}")
        logging.info("Vuer client page: https://vuer.ai")

    @property
    def url(self):
        # FIX: Provide a stable URL property.
        return f"http://{self.host}:{self.port}"

    def run_vuer_server(self):
        """This function runs in a separate thread and manages its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self.app.run()
        finally:
            loop.close()

    # --- Route Handlers for WebRTC Streaming ---

    async def handle_index(self, request):
        html_path = os.path.join(self.project_root, '3_webrtc_streaming', 'index.html')
        with open(html_path, "r") as f:
            return web.Response(content_type="text/html", text=f.read())

    async def handle_javascript(self, request):
        js_path = os.path.join(self.project_root, '3_webrtc_streaming', 'client.js')
        with open(js_path, "r") as f:
            return web.Response(content_type="application/javascript", text=f.read())

    async def handle_webrtc_offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        # Add video tracks to the peer connection
        pc.addTrack(self.left_track)
        pc.addTrack(self.right_track)
        
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
        )
