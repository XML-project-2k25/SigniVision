
# ✋ SigniVision: Real-Time Indian Sign Language Detection with Audio Output

SigniVision is an AI-powered web application that detects **Indian Sign Language (ISL)** signs in real-time using a webcam and generates **audio feedback** using state-of-the-art **transformer-based text-to-speech (TTS)**.

This project bridges the communication gap between the hearing-impaired and the general public using deep learning models such as YOLOv5 (for hand sign detection) and VITS (for audio synthesis).


---

## 🚀 Features

- 🖐️ **Real-time detection** of Indian Sign Language words
- 🔍 **YOLOv5 custom-trained model** for sign recognition
- 🔊 **Transformer-based VITS model** for generating speech
- 🧠 **FastAPI backend** with easy RESTful endpoint
- 🌐 **React frontend** with live webcam capture (see `/frontend`)
- 🧪 CORS-enabled backend to allow frontend integration

---

## 🛠 Tech Stack

| Layer     | Technology                              |
|-----------|------------------------------------------|
| Frontend  | React, Tailwind CSS, Lucide Icons        |
| Backend   | FastAPI, Python, Torch, OpenCV           |
| ML Model  | YOLOv5 for detection, VITS for TTS       |
| Utilities | PIL, NumPy, `transformers`, `torch.hub`  |

---

## 📂 Folder Structure

```

SigniVision/
│
├── model_backend/               # FastAPI backend with YOLOv5 + VITS integration
│   ├── main.py
│   └── ...
│
├── model/                 # YOLOv5 model directory
│      └── best.pt
|      └── signlang.ipynb  # for model fine tuning 
├── frontend/              # React frontend with webcam and UI
│
├── f2/                    # training dataset
|
├── assets/                # Assets like images, PDFs, and demos
│   └── presentation.pdf
│
├── README.md
└── requirements.txt

````

---

## ⚙️ Setup Instructions

### 🔁 Clone the Repository

```bash
git clone https://github.com/XML-project-2k25/SigniVision.git
cd SigniVision
````

---

### 🧠 Backend Setup (FastAPI)

#### 1. Create and activate environment

```bash
cd model_backend
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run FastAPI server

```bash
uvicorn main:app --reload
```

API will be live at `http://localhost:8000`

#### 🧪 Test API with `/predict/`

Send a POST request to `/predict/` with an image file:

```bash
curl -X POST http://localhost:8000/predict/ -F "file=@image.jpg"
```

---

### 🖼 Frontend Setup (React)

```bash
cd frontend
npm install
npm run dev
```

Frontend will run at `http://localhost:5173`

---

## 🔊 How Text-to-Speech Works

* Detected sign class name (e.g., "Thank You") is passed to the `VitsTTS` class.
* VITS generates speech using pretrained model `kakao-enterprise/vits-ljs`.
* Output audio is encoded in **Base64 WAV** and sent back via API.
* Frontend decodes and plays the speech audio in the browser.

---

## 📄 Resources

* [YOLOv5 - Ultralytics](https://github.com/ultralytics/yolov5)
* [VITS - kakao-enterprise](https://huggingface.co/kakao-enterprise/vits-ljs)
* [FastAPI Docs](https://fastapi.tiangolo.com/)
* [Transformers by HuggingFace](https://huggingface.co/transformers/)

---

## 📥 Project Presentation

📄 [Download PDF Presentation](https://github.com/XML-project-2k25/SigniVision/blob/main/assets/presentation.pdf?raw=true)

---

## ✅ Future Scope

* Video call feature
* Support for dynamic ISL gestures
* Multilingual audio output
* Mobile version using React Native
* Improved UI/UX with gesture history and chat overlay

---



