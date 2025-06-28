// client.js (最终修复版)

const pc = new RTCPeerConnection({
    sdpSemantics: 'unified-plan'
});

const statusDiv = document.getElementById('status');
const leftVideo = document.getElementById('left-video');
const rightVideo = document.getElementById('right-video');

// 创建两个独立的MediaStream对象，分别用于左右眼
// 这是为了避免某些浏览器在处理单个多轨道流时的怪癖
const leftStream = new MediaStream();
const rightStream = new MediaStream();

leftVideo.srcObject = leftStream;
rightVideo.srcObject = rightStream;

let tracksReceived = 0;

pc.ontrack = function (event) {
    statusDiv.textContent = 'Status: Receiving streams...';
    console.log(`Received track: kind=${event.track.kind}, id=${event.track.id}`);

    // --- 核心修复：更健壮地分配轨道 ---
    // 将收到的第一个视频轨道添加到左眼流
    if (event.track.kind === 'video' && tracksReceived === 0) {
        console.log('Adding track to LEFT stream.');
        leftStream.addTrack(event.track);
        tracksReceived++;
    } 
    // 将收到的第二个视频轨道添加到右眼流
    else if (event.track.kind === 'video' && tracksReceived === 1) {
        console.log('Adding track to RIGHT stream.');
        rightStream.addTrack(event.track);
        tracksReceived++;
    }
    // ------------------------------------

    // 当两个视频轨道都收到后，打印成功信息
    if (tracksReceived === 2) {
        console.log('Successfully set up both left and right video streams.');
    }
};

pc.oniceconnectionstatechange = e => {
    console.log(`ICE connection state: ${pc.iceConnectionState}`);
    statusDiv.textContent = `Status: ${pc.iceConnectionState}`;
    if (pc.iceConnectionState === 'connected') {
        statusDiv.textContent = 'Status: Connected and streaming';
    }
};

async function start() {
    statusDiv.textContent = 'Status: Connecting...';
    try {
        // 告诉PeerConnection我们想要接收两个视频轨道
        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.addTransceiver('video', { direction: 'recvonly' });

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const response = await fetch('/offer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server returned error: ${response.status} ${await response.text()}`);
        }

        const answer = await response.json();
        await pc.setRemoteDescription(answer);

    } catch (e) {
        alert(`Connection failed: ${e}`);
        console.error(e);
        statusDiv.textContent = 'Status: Connection failed';
    }
}

// 自动开始连接
start();