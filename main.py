import os
import uuid
import shutil
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Flask, request


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Utility functions (color, brightness, eye openness) ...
def get_color_mean(image, landmarks, indices):
    h, w, _ = image.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    colors = [image[y, x] for x, y in points if 0 <= x < w and 0 <= y < h]
    return np.mean(colors, axis=0) if colors else np.array([0, 0, 0])

def get_brightness(image, landmarks, indices):
    h, w, _ = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    brightness = [hsv[y, x][2] for x, y in points if 0 <= x < w and 0 <= y < h]
    return np.mean(brightness) if brightness else 0

def eye_openness(landmarks, top_idx, bottom_idx):
    return abs(landmarks[top_idx].y - landmarks[bottom_idx].y)


def detect_symptoms(filepath):
    image = cv2.imread(filepath)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return [], ""

    landmarks = results.multi_face_landmarks[0].landmark
    symptoms = []
    treatments = []

    # Redness in eyes
    eye_color = get_color_mean(image, landmarks, [33, 133])
    if eye_color[2] > 160:
        symptoms.append("Redness in eyes")
        treatments.append ("Use preservative-free artificial eye drops")

    # Dry lips
    lip_color = get_color_mean(image, landmarks, list(range(61, 88)))
    if lip_color[2] < 130 and lip_color[0] > 90:
        symptoms.append("Dry lips")
        treatments.append ("Use a fragrance-free lip balm containing ingredients like beeswax, petroleum jelly, or shea butter")

    # Swelling and fatigue
    eye_height = eye_openness(landmarks, 159, 145)
    if eye_height < 0.015:
        symptoms.append("Swelling")
        treatments.append ("Apply a cold compress and elevate the affected area")
    if eye_height < 0.018:
        symptoms.append("Fatigue")
        treatments.append ("Sleep well, limit salt intake")

    # Dry skin
    forehead_color = get_color_mean(image, landmarks, [10, 338, 297])
    saturation = cv2.cvtColor(np.uint8([[forehead_color]]), cv2.COLOR_BGR2HSV)[0][0][1]
    if saturation < 40:
        symptoms.append("Dry skin")
        treatments.append ("Apply a thick, fragrance-free moisturizer after taking a bath")

    # Runny nose
    brightness = get_brightness(image, landmarks, [195, 5, 4])
    if brightness > 180:
        symptoms.append("Runny Nose")
        treatments.append ("Use saline nasal spray")

    # Redness
    cheek_color = get_color_mean(image, landmarks, [205, 425])
    if cheek_color[2] > 160 and cheek_color[0] < 130:
        symptoms.append("Redness")
        treatments.append ("Use mild, hypoallergenic moisturizer if it's skin-related")

    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    unique_filename = f"result_{uuid.uuid4().hex}.jpg"
    result_path = os.path.join("static", unique_filename)

    print(f"Saving result image to: {result_path}")
    success = cv2.imwrite(result_path, image)
    if not success:
        print(f"Error: Failed to save result image at {result_path}")
        return symptoms, None

    return symptoms, unique_filename, treatments

# Flask app setup
app = Flask(__name__, static_url_path='/static', static_folder='static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    print("Index hit")
    return 'index.html'


@app.route('/analyze', methods=['POST'])
def analyze():
    print("Analyze hit")
    if 'image' not in request.files or request.files['image'].filename == '':
        return "No file uploaded", 400

    file = request.files['image']

    original_filename = f"original_{uuid.uuid4().hex}_{file.filename}"
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

    file.save(original_path)

    symptoms, result_filename, treatments = detect_symptoms(original_path)

    if not result_filename:
        return "Error processing image.", 500

    return ("results.html", symptoms=symptoms, treatments=treatments, file=result_filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=(int(os.environ.get("PORT",5000))))
    print("Flask app created")
