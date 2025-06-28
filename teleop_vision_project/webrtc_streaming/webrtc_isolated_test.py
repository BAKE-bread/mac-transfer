# webrtc_isolated_test.py (Final version with timestamp fix)

import asyncio
import json
import logging
import os
from pathlib import Path
import time

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame

# Setup detailed logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("aiortc").setLevel(logging.INFO)

ROOT = Path(__file__).parent

class DummyVideoTrack(MediaStreamTrack):
    """
    A virtual video track that generates black frames with correct timestamps.
    """
    kind = "video"

    def __init__(self, track_id):
        super().__init__()
        self.track_id = track_id
        self._start_time = time.time()
        self._frame_count = 0
        self.time_base = 90000  # A common time base for video

    async def recv(self):
        # --- CORE FIX: Manually calculate the timestamp ---
        # Calculate the presentation timestamp (PTS) based on elapsed time
        elapsed_time = time.time() - self._start_time
        pts = int(elapsed_time * self.time_base)
        
        # Create a black video frame
        frame = VideoFrame(width=640, height=480, format="yuv420p")
        for p in frame.planes:
            p.update(bytes(p.line_size * p.height))
        
        # Set the timestamp on the frame
        frame.pts = pts
        frame.time_base = self.time_base
        
        self._frame_count += 1
        if self._frame_count % 150 == 0: # Print status every ~5 seconds
            logging.info(f"DummyTrack {self.track_id}: sent frame {self._frame_count}")
        
        # Pace the stream to roughly 30fps
        await asyncio.sleep(1/30)

        return frame

# --- The rest of the file remains the same ---

async def index(request):
    with open(os.path.join(ROOT, "index.html"), "r", encoding="utf-8") as f:
        return web.Response(content_type="text/html", text=f.read())

async def javascript(request):
    with open(os.path.join(ROOT, "client.js"), "r", encoding="utf-8") as f:
        return web.Response(content_type="application/javascript", text=f.read())

async def offer(request):
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info(f"PeerConnection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()

        pc.addTransceiver(DummyVideoTrack("left"), direction="sendonly")
        pc.addTransceiver(DummyVideoTrack("right"), direction="sendonly")

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )
    except Exception as e:
        logging.error("Error handling offer:", exc_info=True)
        return web.Response(status=500, text=str(e))

async def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    logging.info("Starting isolated test server at http://0.0.0.0:8080")
    await site.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass