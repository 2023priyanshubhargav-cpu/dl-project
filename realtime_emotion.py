import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json

# ── Config ──────────────────────────────────────────────
with open("emotion_model_config.json") as f:
    config = json.load(f)

CLASSES = config["classes"]
MEAN    = config["mean"]
STD     = config["std"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_COLORS = {
    'angry':    (0,   0,   255),
    'disgust':  (0,   140, 255),
    'fear':     (0,   255, 255),
    'happy':    (0,   255, 0),
    'neutral':  (255, 255, 0),
    'sad':      (255, 0,   0),
    'surprise': (255, 0,   255),
}

# ── Load TorchScript model ─────────────────────────────
model = torch.jit.load("emotion_model_scripted.pt", map_location=device)
model.eval()
print("Model loaded (TorchScript)")

# ── Transform ───────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Face detector ───────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Webcam ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
else:
    print("Webcam started — Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_pil  = Image.fromarray(
                cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            )

            tensor = infer_transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx  = np.argmax(probs)
            pred_name = CLASSES[pred_idx]
            conf      = probs[pred_idx]
            color     = EMOTION_COLORS[pred_name]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            label = f"{pred_name.upper()} {conf*100:.1f}%"
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
            cv2.putText(frame, label, (x+5, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            bx, by = 20, 50
            cv2.putText(frame, "PROBABILITIES", (bx, by-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

            for i, (cls, prob) in enumerate(zip(CLASSES, probs)):
                yp      = by + i * 35
                bar_len = int(prob * 200)
                clr     = EMOTION_COLORS[cls]

                cv2.rectangle(frame, (bx, yp),
                              (bx+200, yp+20), (50, 50, 50), -1)
                cv2.rectangle(frame, (bx, yp),
                              (bx+bar_len, yp+20), clr, -1)
                cv2.putText(frame,
                            f"{cls:8s} {prob*100:5.1f}%",
                            (bx+205, yp+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        cv2.putText(frame, "Press Q to quit",
                    (20, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done")
