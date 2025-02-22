from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
import pytesseract
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for Next.js frontend

# Load YOLOv8 model
model = YOLO("best.pt")  # Your trained model

# Load OCR Engine
reader = easyocr.Reader(['en'])

@app.route("/detect", methods=["POST"])
def detect_license_plate():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Read Image
    file = request.files["image"]
    image = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Run YOLOv8 detection
    results = model(img)

    # Process detections
    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            cropped_plate = img[y1:y2, x1:x2]

            # OCR on the detected plate
            text = reader.readtext(cropped_plate, detail=0)  # Extract text
            extracted_text = " ".join(text) if text else "No text found"

            plates.append({
                "bbox": [x1, y1, x2, y2],
                "text": extracted_text
            })

    return jsonify({"plates": plates})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)