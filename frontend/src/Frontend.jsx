import { useState, useRef, useEffect } from "react";
import "./styles.css";
import { Video, Image, User, LogOut, Camera } from "lucide-react";
import { getPredictions } from "./api";

export default function Frontend() {
  const [image, setImage] = useState(null);
  const [videoChatActive, setVideoChatActive] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [Text, setText] = useState(undefined);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const sendBlobToPrediction = (blob) => {
    const formData = new FormData();
    // Append the blob as a file to FormData
    formData.append("file", blob, "captured-image.jpeg");

    // Call getPredictions with formData
    getPredictions(formData).then((data) => {
      setText(data);
    });
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraActive(true);
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setCameraActive(false);
    }
  };

  useEffect(() => {
    let interval;
    if (cameraActive) {
      interval = setInterval(() => {
        if (videoRef.current && canvasRef.current) {
          const context = canvasRef.current.getContext("2d");
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
          context.drawImage(videoRef.current, 0, 0);
          setImage(canvasRef.current.toDataURL("image/jpeg"));
          // Convert canvas to Blob and send to server
          canvasRef.current.toBlob((blob) => {
            if (blob) {
              sendBlobToPrediction(blob);
            }
          }, "image/jpeg");
        }
      }, 500);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [cameraActive]);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target.result);
      reader.readAsDataURL(file);
      sendBlobToPrediction(file);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1 className="title">Symbol Detection</h1>
        <div className="profile-options">
          <button className="button outline">
            <User size={20} className="icon" /> View Profile
          </button>
          <button className="button danger">
            <LogOut size={20} className="icon" /> Logout
          </button>
        </div>
      </div>

      <div className="center-container">
        <div className="card camera-card">
          <div className="card-content">
            <Image size={48} />
            <button
              className="button sky-blue"
              onClick={startCamera}
              disabled={cameraActive}
            >
              Open Camera
            </button>
            {cameraActive && (
              <button className="button creamish" onClick={stopCamera}>
                Close Camera
              </button>
            )}
            <div
              className="video-container"
              style={{
                width: "100%",
                height: "300px",
                display: cameraActive ? "block" : "none",
              }}
            >
              <video
                ref={videoRef}
                className="media"
                autoPlay
                playsInline
                style={{ width: "100%", height: "100%" }}
              />
            </div>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="file-input"
            />
            <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
            {image && (
              <>
                <img
                  src={image}
                  alt="Captured"
                  className="media"
                  id="captured-img"
                />
                <h3
                  style={{
                    backgroundColor: "black",
                    padding: "10px 20px",
                  }}
                >
                  {Text ? Text : ""}
                </h3>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="video-chat-container">
        <button
          className="button sky-blue video-chat"
          onClick={() => setVideoChatActive(!videoChatActive)}
        >
          {videoChatActive ? "End Video Chat" : "Start Video Chat"}
        </button>
        {videoChatActive && (
          <p className="status-text">Video Chat is Active...</p>
        )}
      </div>
    </div>
  );
}
