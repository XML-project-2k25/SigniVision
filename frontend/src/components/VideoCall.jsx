import React, { useEffect, useRef, useState } from "react";

function generateRoomId() {
  return Math.random().toString(36).substring(2, 10);
}

const SIGNAL_SERVER_URL = "ws://localhost:8000/ws/";

export default function VideoCall() {
  const [roomId, setRoomId] = useState("");
  const [inputRoomId, setInputRoomId] = useState("");
  const [callActive, setCallActive] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [roomJoined, setRoomJoined] = useState(false);
  const [remoteStreamAvailable, setRemoteStreamAvailable] = useState(false);
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const wsRef = useRef(null);
  const pcRef = useRef(null);
  const localStreamRef = useRef(null);
  const isCallerRef = useRef(false);

  const iceServers = {
    iceServers: [
      { urls: "stun:stun.l.google.com:19302" },
    ],
  };

  // Join a room (either create or enter)
  const joinRoom = async (id) => {
    setRoomId(id);
    setRoomJoined(true);
    setStatus("Room joined. Connecting to signaling server...");
    // Connect to signaling server immediately
    wsRef.current = new WebSocket(SIGNAL_SERVER_URL + id);
    wsRef.current.onopen = () => {
      setStatus("Connected to signaling server. Ready to start call.");
    };
    wsRef.current.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "offer") {
        setStatus("Received offer. Creating answer...");
        await handleOffer(data.offer);
      } else if (data.type === "answer") {
        setStatus("Received answer. Connecting media...");
        await handleAnswer(data.answer);
      } else if (data.type === "ice-candidate") {
        await handleRemoteICE(data.candidate);
      }
    };
  };

  // Start the call: get media, create peer connection, create/send offer
  const startCall = async () => {
    setStatus("Starting call and getting local media...");
    isCallerRef.current = true;
    setCallActive(true);
    setRemoteStreamAvailable(false);
    // 1. Get local media
    const localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    localStreamRef.current = localStream;
    if (localVideoRef.current) {
      localVideoRef.current.srcObject = localStream;
    }
    // 2. Create peer connection
    createPeerConnection();
    // 3. Add local tracks
    localStream.getTracks().forEach((track) => {
      pcRef.current.addTrack(track, localStream);
    });
    // 4. Create and send offer
    setStatus("Creating and sending offer...");
    const offer = await pcRef.current.createOffer();
    await pcRef.current.setLocalDescription(offer);
    wsRef.current.send(JSON.stringify({ type: "offer", offer }));
    setStatus("Offer sent. Waiting for answer...");
  };

  // Create peer connection and set up handlers
  const createPeerConnection = () => {
    if (pcRef.current) return;
    pcRef.current = new RTCPeerConnection(iceServers);
    pcRef.current.ontrack = (event) => {
      if (remoteVideoRef.current) {
        remoteVideoRef.current.srcObject = event.streams[0];
        setRemoteStreamAvailable(true);
      }
    };
    pcRef.current.onicecandidate = (event) => {
      if (event.candidate && wsRef.current) {
        wsRef.current.send(JSON.stringify({ type: "ice-candidate", candidate: event.candidate }));
      }
    };
  };

  // Handle incoming offer (for callee)
  const handleOffer = async (offer) => {
    isCallerRef.current = false;
    setCallActive(true);
    setRemoteStreamAvailable(false);
    // 1. Get local media
    const localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    localStreamRef.current = localStream;
    if (localVideoRef.current) {
      localVideoRef.current.srcObject = localStream;
    }
    // 2. Create peer connection
    createPeerConnection();
    // 3. Add local tracks
    localStream.getTracks().forEach((track) => {
      pcRef.current.addTrack(track, localStream);
    });
    // 4. Set remote description and create/send answer
    await pcRef.current.setRemoteDescription(new window.RTCSessionDescription(offer));
    const answer = await pcRef.current.createAnswer();
    await pcRef.current.setLocalDescription(answer);
    wsRef.current.send(JSON.stringify({ type: "answer", answer }));
    setStatus("Answer sent. Waiting for connection...");
  };

  // Handle incoming answer (for caller)
  const handleAnswer = async (answer) => {
    await pcRef.current.setRemoteDescription(new window.RTCSessionDescription(answer));
    setStatus("Call connected!");
  };

  // Handle remote ICE candidates
  const handleRemoteICE = async (candidate) => {
    try {
      await pcRef.current.addIceCandidate(new window.RTCIceCandidate(candidate));
    } catch (e) {
      console.error("Error adding remote ICE candidate", e);
    }
  };

  // End call and cleanup
  const endCall = () => {
    setCallActive(false);
    setStatus("Call ended");
    setRemoteStreamAvailable(false);
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    }
    if (localVideoRef.current) localVideoRef.current.srcObject = null;
    if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
    setRoomJoined(false);
    setRoomId("");
    setInputRoomId("");
  };

  useEffect(() => {
    return () => {
      endCall();
    };
    // eslint-disable-next-line
  }, []);

  return (
    <div className="card" style={{ marginTop: 20 }}>
      <h2>Video Call</h2>
      {!roomJoined ? (
        <div style={{ marginBottom: 16 }}>
          <button
            className="button sky-blue"
            onClick={() => joinRoom(generateRoomId())}
            style={{ marginRight: 8 }}
          >
            Create Room
          </button>
          <input
            type="text"
            placeholder="Enter Room ID"
            value={inputRoomId}
            onChange={e => setInputRoomId(e.target.value)}
            style={{ marginRight: 8, padding: 4 }}
          />
          <button
            className="button outline"
            onClick={() => inputRoomId && joinRoom(inputRoomId)}
            disabled={!inputRoomId}
          >
            Join Room
          </button>
        </div>
      ) : (
        <>
          <p>Room ID: <b>{roomId}</b></p>
          {!callActive ? (
            <button className="button sky-blue" onClick={startCall}>
              Start Call
            </button>
          ) : (
            <button className="button danger" onClick={endCall}>
              End Call
            </button>
          )}
        </>
      )}
      {callActive && (
        <div style={{ display: "flex", gap: 20, justifyContent: "center", marginTop: 16 }}>
          <div>
            <video ref={localVideoRef} autoPlay playsInline muted style={{ width: 240, height: 180, background: "#222" }} />
            <div style={{ textAlign: "center" }}>You</div>
          </div>
          <div>
            <video ref={remoteVideoRef} autoPlay playsInline style={{ width: 240, height: 180, background: "#222" }} />
            <div style={{ textAlign: "center" }}>Remote</div>
            {!remoteStreamAvailable && <div style={{ color: '#888', fontSize: 14 }}>Waiting for remote...</div>}
          </div>
        </div>
      )}
      <div style={{ marginTop: 10, color: '#555' }}>{status}</div>
    </div>
  );
} 