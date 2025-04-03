import base64
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wavfile


app = FastAPI()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="../model/best.pt")
model.to(device)
# add cors policy
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class VitsTTS:
    def _init_(self, model_name="kakao-enterprise/vits-ljs"):
        """Initialize the VITS model and tokenizer."""
        print("Loading model and tokenizer...")
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_rate = self.model.config.sampling_rate
        print("Model loaded successfully.")

    def text_to_speech(self, text: str) -> str:
        """Convert text to speech and return Base64-encoded WAV audio."""
        try:
            if not text.strip():
                raise ValueError("Transcription resulted in empty text") 

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,        # Ensure uniform input length
                truncation=True,      # Prevent excessively long inputs
                max_length=256,  # Explicitly set max_length
            )

            with torch.no_grad():
                waveform = self.model(**inputs).waveform

            # Save waveform to an in-memory buffer
            buffer = BytesIO()
            wavfile.write(buffer, rate=self.sampling_rate, data=waveform.squeeze().numpy())
            buffer.seek(0)

            # Convert audio bytes to Base64 string
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return audio_base64
        except Exception as e:
            print(f"An error occurred during vits text-to-speech: {e}")
            return None

tts_model = VitsTTS()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    print(img.size)
    results = model(img)
    sign = results.pandas().xyxy[0].to_dict(orient="records")

    if (len(sign) > 0):
        sign[0]["audio"] = tts_model.text_to_speech(sign[0]["name"])
    return sign




if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)