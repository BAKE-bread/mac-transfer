// webrtc_streaming/client.js
var pc = null;

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('video', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                pc.addEventListener('icegatheringstatechange', function() {
                    if (pc.iceGatheringState === 'complete') {
                        resolve();
                    }
                });
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        console.log("Sending offer to server...");
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: { 'Content-Type': 'application/json' },
            method: 'POST'
        });
    }).then((response) => {
        console.log("Received answer from server.");
        return response.json();
    }).then((answer) => {
        console.log("Setting remote description.");
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        alert(e);
    });
}

function start() {
    console.log("Starting WebRTC negotiation...");
    var config = { sdpSemantics: 'unified-plan' };
    pc = new RTCPeerConnection(config);

    var trackCount = 0;
    pc.addEventListener('track', (evt) => {
        trackCount++;
        console.log(`Track #${trackCount} received.`);
        if (trackCount === 1) {
            document.getElementById('left-video').srcObject = evt.streams[0];
        } else {
            document.getElementById('right-video').srcObject = evt.streams[0];
        }
    });
    
    pc.addEventListener('connectionstatechange', () => {
        console.log("Connection state:", pc.connectionState);
    });

    negotiate();
}

window.onload = start;