import base64
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO


app = FastAPI()

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="../model/best.pt")
# add cors policy
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    print(img.size)
    results = model(img)
    return results.pandas().xyxy[0].to_dict(orient="records")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
